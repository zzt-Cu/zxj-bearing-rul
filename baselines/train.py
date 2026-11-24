"""Train or continue a CPU baseline from public train/validation JSONL only."""
import argparse
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
import tempfile
import time

try:
    from .model import Model, checkpoint_digest, _THREADS, INPUT_MODE
    from .data import read_dataset, sha256
except ImportError:
    from model import Model, checkpoint_digest, _THREADS, INPUT_MODE
    from data import read_dataset, sha256
import numpy as np

ROOT = Path(__file__).resolve().parent


def metrics(model, data):
    predicted = model.predict(data["x"])
    expected = data["y"]
    if model.meta["task"] == "classification":
        labels = predicted.argmax(axis=1)
        observed = sorted(set(expected.tolist()) | set(labels.tolist()))
        f1 = [2 * np.sum((labels == label) & (expected == label)) /
              (np.sum(labels == label) + np.sum(expected == label)) for label in observed]
        result = {"accuracy": float(np.mean(labels == expected)), "macro_f1": float(np.mean(f1))}
        if "SevereFault" in model.meta["classes"]:
            severe = model.meta["classes"].index("SevereFault")
            count = int(np.sum(expected == severe))
            result["severe_miss_rate"] = float(np.sum((expected == severe) & (labels != severe)) / count) if count else None
        return result
    error = predicted - expected
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}


def train_run(profile, train_path, validation_path, output, *, resume=None, epochs=None,
              learning_rate=None, batch_size=None, seed=None, stage_id="baseline", label_provenance=None,
              ensemble=None):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("output must be a new directory; previous rounds are never overwritten")
    config_path = Path(profile) if str(profile).endswith(".json") else ROOT / "configs" / (str(profile) + ".json")
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    if config.get("profile") not in ("A", "B", "C") or config.get("task") not in ("classification", "regression"):
        raise ValueError("unsupported baseline profile")
    if not isinstance(stage_id, str) or not stage_id.strip() or len(stage_id) > 128:
        raise ValueError("stage ID must be nonempty and bounded")
    provenance = None
    bearing_rul = config["profile"] == "B" and config["task"] == "regression"
    if bearing_rul and config.get("target_unit") != "minutes":
        raise ValueError("B RUL requires target_unit=minutes")
    if config["profile"] == "B" and config["task"] == "classification":
        if label_provenance is None:
            raise ValueError("B requires explicit label provenance; do not silently derive official health labels from RUL")
        raw = Path(label_provenance).read_bytes()
        declared = json.loads(raw)
        if not isinstance(declared, dict) or declared.get("status") not in ("draft", "approved") or not isinstance(declared.get("method"), str) or not declared["method"].strip():
            raise ValueError("label provenance requires status=draft|approved and a nonempty method")
        provenance = {"sha256": sha256(raw), "declared_status": declared["status"], "method": declared["method"]}
    classes = config.get("classes", [])
    train = read_dataset(train_path, config["task"], classes)
    validation = read_dataset(validation_path, config["task"], classes, train["paths"])
    if set(train["ids"]) & set(validation["ids"]):
        raise ValueError("training and validation sample IDs overlap")
    parent = checkpoint_digest(resume) if resume else None
    model = Model.load(resume) if resume else Model(config["task"], train["paths"], classes,
        train["x"], train["y"], hidden=config["hidden"], seed=config["seed"] if seed is None else seed,
        learning_rate=config["learning_rate"], batch_size=config["batch_size"])
    if (model.meta["task"] != config["task"] or model.meta["paths"] != train["paths"] or model.meta["classes"] != classes
            or (resume and model.meta.get("profile") != config["profile"])):
        raise ValueError("checkpoint task, profile, feature schema or classes do not match")
    if bearing_rul and resume and model.meta.get("target_unit") != "minutes":
        raise ValueError("B RUL checkpoint must use minutes")
    if resume and seed is not None and seed != model.meta["seed"]:
        raise ValueError("resume must preserve the checkpoint's random seed")
    if resume and config["hidden"] != model.meta["hidden"]:
        raise ValueError("resume hidden width must match checkpoint; changing the profile does not migrate weights")
    if resume and config["seed"] != model.meta["seed"]:
        raise ValueError("resume profile must preserve the checkpoint's random seed")
    model.meta["profile"] = config["profile"]
    if bearing_rul:
        model.meta["target_unit"] = "minutes"
    # 参考起点的输入宽度是原始列数；abs8log 需要追加 14 列对数绝对量，
    # 这里拿到训练集后一次性扩展（新增列权重为 0，统计量取本训练集）。
    if INPUT_MODE == "abs8log" and model.params["w1"].shape[0] == len(train["paths"]):
        model.expand_inputs(train["x"])
    if learning_rate is not None:
        model.meta["learning_rate"] = learning_rate
    if batch_size is not None:
        model.meta["batch_size"] = batch_size
    model.validate()
    count = config["epochs"] if epochs is None else epochs
    if type(count) is not int or not 1 <= count <= 10000:
        raise ValueError("epochs must be in 1..10000")
    initial_steps, initial_epochs = model.meta["steps"], model.meta["epochs"]
    validation_before = metrics(model, validation)
    started = time.monotonic()
    records = []
    ensemble_report = None
    if ensemble:
        # 多轨迹：同一续训起点、同一轮数，仅 batch_size 不同；结束后权重与 Adam 矩取算术平均。
        # 目的不是增加容量，而是压掉单条轨迹对训练集偶然结构的过拟合——
        # 各轨迹共享同一信号、各自噪声不同，平均后噪声互相抵消。
        sizes = [int(b) for b in ensemble]
        if len(sizes) < 2 or any(b < 1 or b > 65536 for b in sizes):
            raise ValueError("ensemble requires at least two valid batch sizes")
        trajectories, curves = [], []
        for size in sizes:
            clone = Model.load(resume) if resume else Model(
                config["task"], train["paths"], classes, train["x"], train["y"], hidden=config["hidden"],
                seed=config["seed"] if seed is None else seed, learning_rate=config["learning_rate"],
                batch_size=config["batch_size"])
            if INPUT_MODE == "abs8log" and clone.params["w1"].shape[0] == len(train["paths"]):
                clone.expand_inputs(train["x"])
            clone.meta["profile"] = config["profile"]
            if bearing_rul:
                clone.meta["target_unit"] = "minutes"
            clone.meta["learning_rate"] = model.meta["learning_rate"]
            clone.meta["batch_size"] = size
            clone.validate()
            curve = []
            for _ in range(count):
                loss = clone.fit(train["x"], train["y"], 1)[0]
                curve.append({"epoch": clone.meta["epochs"], "step": clone.meta["steps"], "loss": loss,
                              "validation": metrics(clone, validation)})
            trajectories.append(clone)
            curves.append(curve)
        for name in model.params:
            model.params[name] = np.mean([t.params[name] for t in trajectories], axis=0)
            model.first[name] = np.mean([t.first[name] for t in trajectories], axis=0)
            model.second[name] = np.mean([t.second[name] for t in trajectories], axis=0)
        # 目标空间必须跟着轨迹走：起点检查点的 target_mean/scale/power 属于原始目标空间，
        # 若不覆盖，保存下来的权重与还原参数会错位（预测看似可用，实际是错的变换）。
        for key in ("target_power", "target_mean", "target_scale"):
            model.meta[key] = trajectories[0].meta[key]
        model.meta["steps"] = int(round(float(np.mean([t.meta["steps"] for t in trajectories]))))
        model.meta["epochs"] = int(round(float(np.mean([t.meta["epochs"] for t in trajectories]))))
        model.meta["batch_size"] = sizes[0]
        model.validate()
        merged = metrics(model, validation)
        records = curves[0][:-1] + [{"epoch": model.meta["epochs"], "step": model.meta["steps"],
                                     "loss": curves[0][-1]["loss"], "validation": merged}]
        ensemble_report = {"batch_sizes": sizes,
                           "member_rmse": [metrics(t, validation)["rmse"] for t in trajectories],
                           "merged_rmse": merged["rmse"]}
        # 标记这是「权重平均」产物：它的低误差来自 6 个函数平均，不是权重本身落在好解上。
        # harness 据此禁止把它当作下一轮续训起点（证据见 P3_集成点不可续训_诊断.md）。
        model.meta["aggregate"] = "weight_average"
        model.meta["ensemble"] = sizes
    else:
        for _ in range(count):
            loss = model.fit(train["x"], train["y"], 1)[0]
            records.append({"epoch": model.meta["epochs"], "step": model.meta["steps"], "loss": loss,
                            "validation": metrics(model, validation)})
    duration = time.monotonic() - started
    if not records:
        raise ValueError("at least one training epoch is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".baseline-", dir=output.parent) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir(mode=0o700)
        checkpoint = staging / "checkpoint.npz"
        model.save(checkpoint)
        report = {"format": "contest.cpu-baseline-report.v1", "profile": config["profile"], "task": config["task"],
                  "purpose": "development_reference", "official_scoring_ready": False, "device": "cpu", "cpu_threads_requested": _THREADS,
                  "numpy_version": np.__version__, "python_version": platform.python_version(), "machine": platform.machine(), "stage_id": stage_id, "seed": model.meta["seed"],
                  "parameter_count": model.meta["parameter_count"], "input_features": len(model.meta["paths"]),
                  "train_rows": len(train["ids"]), "validation_rows": len(validation["ids"]),
                  "train_sha256": train["sha256"], "validation_sha256": validation["sha256"],
                  "profile_sha256": sha256(config_raw), "parent_checkpoint_sha256": parent,
                  "checkpoint_sha256": checkpoint_digest(checkpoint), "initial_steps": initial_steps,
                  "initial_epochs": initial_epochs, "final_steps": model.meta["steps"], "final_epochs": model.meta["epochs"],
                  "training_seconds": duration, "learning_rate": model.meta["learning_rate"], "batch_size": model.meta["batch_size"],
                  "input_mode": model.meta.get("input_mode"), "input_width": model.meta.get("input_width"),
                  "ensemble": ensemble_report,
                  "validation_before": validation_before, "validation_after": records[-1]["validation"],
                  "configuration": {"hidden": model.meta["hidden"], "activation": "tanh", "optimizer": "Adam",
                      "adam_beta1": .9, "adam_beta2": .999, "adam_epsilon": 1e-8, "gradient_clip_norm": 5.0,
                      "feature_clip": [-10.0, 10.0], "normalization": "initial_training_mean_std",
                      "learning_rate": model.meta["learning_rate"], "batch_size": model.meta["batch_size"],
                      "requested_epochs": count, "classes": model.meta["classes"]},
                  "source_sha256": {name: sha256((ROOT / name).read_bytes()) for name in ("model.py", "data.py", "train.py", "predict.py")},
                  "label_provenance": provenance, "created_at": datetime.now(timezone.utc).isoformat()}
        if bearing_rul:
            report.update(target_unit="minutes", target_origin="remaining_time_from_lifecycle",
                target_definition="(last_snapshot - current_snapshot) * 1 minute; last observed snapshot has RUL 0")
        (staging / "training.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        with (staging / "history.jsonl").open("w") as stream:
            for record in records:
                stream.write(json.dumps({"stage_id": stage_id, "seed": model.meta["seed"], **record}, allow_nan=False) + "\n")
        os.replace(staging, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="A, B, C or explicit profile JSON")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume", help="Previous round checkpoint; preserve weights, Adam, normalization and epoch")
    continuation.add_argument("--from-baseline", help="Official/reference starting checkpoint; identical continuation semantics")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stage-id", default="baseline")
    parser.add_argument("--label-provenance", help="Required only for historical B classification profiles: JSON with status and method")
    parser.add_argument("--ensemble", help="Comma separated batch sizes; train each from the same start and average the weights")
    args = parser.parse_args()
    if args.epochs is not None and not 1 <= args.epochs <= 10000:
        parser.error("epochs must be in 1..10000")
    ensemble = None
    if args.ensemble:
        try:
            ensemble = [int(part) for part in args.ensemble.split(",") if part.strip()]
        except ValueError:
            parser.error("--ensemble expects comma separated integers")
        if len(ensemble) < 2:
            parser.error("--ensemble needs at least two batch sizes")
    report = train_run(args.profile, args.train, args.validation, args.output, resume=args.resume or args.from_baseline,
        epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, seed=args.seed,
        stage_id=args.stage_id, label_provenance=args.label_provenance, ensemble=ensemble)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
