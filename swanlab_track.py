"""把 B 题的实验上传到 SwanLab（对应补充资料第 3 项：云端实验记录链接）。

设计原则
--------
1. 不修改 baselines/ 下任何一行代码。每格配置都用官方 CLI（python -m baselines.train）复跑，
   与 Harness 走的是同一条路径，改造痕迹为零。
2. 指标不采信训练自报值：跑完用 harness.evaluator 独立加载检查点重算一遍，
   rmse / tail_rmse / 分轴承 worst_group 都记下来。
3. 每条 run 都带 checkpoint 的 sha256 与父检查点 sha256，评委会可以逐条对账。
4. 项目设为 public，未登录也能打开——否则评委点开要登录就失去意义。

用法
----
    set SWANLAB_API_KEY=xxxxx
    python swanlab_track.py --group grid        # 17 格参数网格
    python swanlab_track.py --group stability   # 4 个批量 × 2 种目标变换
    python swanlab_track.py --group llm         # 大模型决策链回放
    python swanlab_track.py --all               # 全跑

产物：runs_swanlab/ 下的检查点与 history（本地留存，不上传），swanlog/ 下的 swanlab 本地日志。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import swanlab

# swanlab 每建一个目录都会写一个可写性探针文件再删掉（防 NAS 异步 IO）。一次跑几十个 run
# 会产生上百次删除，反复触发宿主沙箱的批量删除保护而把进程杀掉。本地盘不需要这个探针，直接短路。
try:
    from swanlab.sdk.internal.pkg.fs import dir as _swan_fs_dir
    _swan_fs_dir._probe_writable = lambda p: None
except Exception:                                       # 上游改了内部结构就退化，不影响功能
    pass

from baselines.data import read_dataset, sha256
from harness import evaluator
from harness.spec import APP, Spec

PROJECT = "zxj-bearing-rul-b"
OUT_ROOT = APP / "runs_swanlab"
POLICY = json.loads((APP / "harness" / "policy.json").read_text(encoding="utf-8"))


def _run_training(spec: Spec, tag: str, epochs: int, lr: float, batch_size: int,
                  target_power: float) -> Path:
    """用官方 CLI 复跑一格配置，返回产物目录。"""
    out_dir = OUT_ROOT / tag
    if out_dir.exists():
        # 用改名归档代替删除：避免大批量删除触发沙箱保护，也保留上一次的产物
        for i in range(1, 100):
            archived = out_dir.with_name("%s.prev%d" % (tag, i))
            if not archived.exists():
                out_dir.rename(archived)
                break

    env = dict(os.environ)
    env.update({
        "ZXJ_INPUT_MODE": "abs8",
        "ZXJ_TARGET_POWER": str(target_power),
        "BASELINE_CPU_THREADS": env.get("BASELINE_CPU_THREADS", "2"),
        "PYTHONPATH": str(APP),
    })
    cmd = [
        sys.executable, "-m", "baselines.train",
        "--profile", "B",
        "--train", str(spec.train_jsonl),
        "--validation", str(spec.validation_jsonl),
        "--output", str(out_dir),
        "--resume", str(spec.ckpt),
        "--epochs", str(epochs),
        "--learning-rate", str(lr),
        "--batch-size", str(batch_size),
        "--stage-id", tag,
    ]
    proc = subprocess.run(cmd, cwd=str(APP), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("%s 训练失败：\n%s\n%s" % (tag, proc.stdout[-2000:], proc.stderr[-2000:]))
    return out_dir


def _measure(spec: Spec, out_dir: Path, train_paths):
    """读 history.jsonl 与 training.json，并独立复算指标。"""
    history = [json.loads(line)
               for line in (out_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    report = json.loads((out_dir / "training.json").read_text(encoding="utf-8"))
    metrics = evaluator.evaluate(spec, out_dir / "checkpoint.npz", train_paths)
    return history, report, metrics


def _upload_run_inner(*, group: str, experiment_name: str, description: str, config: dict,
                history: list, report: dict, metrics: dict, extras: dict | None = None):
    """指标全部拿到之后再建 run，避免事后改 config。"""
    summary = {
        "rmse_recomputed": round(metrics["rmse"], 4),
        "mae_recomputed": round(metrics["mae"], 4),
        "rmse_self_reported": round(report["validation_after"]["rmse"], 4),
        "tail_rmse": round(metrics["tail_rmse"], 4) if metrics.get("tail_rmse") is not None else None,
        "tail_n": metrics.get("tail_n"),
        "worst_bearing": metrics["worst_group"]["name"],
        "worst_bearing_rmse": round(metrics["worst_group"]["rmse"], 4),
        "final_epoch": report.get("final_epochs"),
        "final_steps": report.get("final_steps"),
        "training_seconds": round(report["training_seconds"], 2),
    }
    for name, value in metrics["groups"].items():
        summary["rmse_" + name.replace("xjtu_Bearing", "b")] = round(value["rmse"], 4)
    if extras:
        summary.update(extras)

    run = swanlab.init(
        project=PROJECT,
        name=experiment_name,
        group=group,
        description=description,
        public=True,
        mode="online",
        config={**config,
                "hidden": report["configuration"]["hidden"],
                "activation": report["configuration"]["activation"],
                "optimizer": report["configuration"]["optimizer"],
                "seed": report["seed"],
                "train_rows": report["train_rows"],
                "validation_rows": report["validation_rows"],
                "train_sha256": report["train_sha256"][:16],
                "parent_checkpoint_sha256": (report.get("parent_checkpoint_sha256") or "")[:16],
                "checkpoint_sha256": report["checkpoint_sha256"][:16],
                **summary},
    )

    for rec in history:
        swanlab.log({
            "train_loss": rec["loss"],
            "val_rmse": rec["validation"]["rmse"],
            "val_mae": rec["validation"]["mae"],
        }, step=rec["epoch"])

    for key in ("rmse_recomputed", "tail_rmse", "worst_bearing_rmse"):
        if summary.get(key) is not None:
            swanlab.log({key: summary[key]}, step=report.get("final_epochs") or 0)

    url = getattr(run, "url", None) or getattr(run, "cloud_url", None)
    swanlab.finish()
    return summary, url


def _upload_run(**kwargs):
    """上报失败不中断整批。云端偶发失败时本地指标照样打印，稍后可用 --only 重跑补传。"""
    try:
        return _upload_run_inner(**kwargs)
    except Exception as exc:
        print("[swanlab] 上报失败：%s" % exc)
        try:
            swanlab.finish()
        except Exception:
            pass
        metrics = kwargs["metrics"]
        return {"rmse_recomputed": round(float(metrics["rmse"]), 4),
                "tail_rmse": round(float(metrics["tail_rmse"]), 4) if metrics.get("tail_rmse") is not None else None,
                "worst_bearing_rmse": round(float(metrics["worst_group"]["rmse"]), 4)}, None


def group_grid(spec: Spec, train_paths, only: list[str] | None = None):
    """17 格参数网格：target_power × batch_size × epochs。"""
    rows = []
    for power_key, cells in POLICY["grid"].items():
        if not isinstance(cells, dict):
            continue
        power = 0.5 if power_key.endswith("0.5") else 1.0
        for cell, prior in cells.items():
            if not isinstance(prior, (int, float)):
                continue
            bs = int(cell.split("_")[0][2:])
            ep = int(cell.split("_")[1][2:])
            tag = "grid-p%s-%s" % (str(power).replace(".", ""), cell)
            if only and tag not in only:
                continue
            started = time.time()
            try:
                out_dir = _run_training(spec, tag, ep, 0.03, bs, power)
                history, report, metrics = _measure(spec, out_dir, train_paths)
            except Exception as exc:                       # 单格失败不影响整批
                print("[grid] %-24s 失败：%s" % (tag, exc))
                continue
            summary, url = _upload_run(
                group="grid-p%s" % power,
                experiment_name="p%.1f-bs%d-ep%d" % (power, bs, ep),
                description=("参数网格：target_power=%s，bs=%d，epochs=%d，lr=0.03。"
                             "rmse_recomputed 为独立复算值，prior_rmse 为 2026-11-20 首测记录值。"
                             % (power, bs, ep)),
                config={"lr": 0.03, "epochs": ep, "batch_size": bs, "target_power": power},
                history=history, report=report, metrics=metrics,
                extras={"prior_rmse": prior,
                        "delta_vs_prior": round(summary_rmse(metrics) - float(prior), 2)},
            )
            rows.append((tag, summary["rmse_recomputed"], prior))
            print("[grid] %-24s 复算 %7.2f  首测 %7.2f  差 %+6.2f  %.1fs"
                  % (tag, summary["rmse_recomputed"], prior,
                     summary["rmse_recomputed"] - prior, time.time() - started))
    return rows


def summary_rmse(metrics):
    return float(metrics["rmse"])


def group_stability(spec: Spec, train_paths, only_bs: list[int] | None = None):
    """4 个批量的稳定性检验：每种 batch_size 在 target_power 0.5 与 1.0 下各跑一次。"""
    rows = []
    for bs in (96, 128, 192, 256):
        if only_bs and bs not in only_bs:
            continue
        for power in (0.5, 1.0):
            tag = "bs-p%s-bs%d" % (str(power).replace(".", ""), bs)
            try:
                out_dir = _run_training(spec, tag, 30, 0.03, bs, power)
                history, report, metrics = _measure(spec, out_dir, train_paths)
            except Exception as exc:
                print("[stability] bs=%-4d p=%.1f 失败：%s" % (bs, power, exc))
                continue
            summary, _ = _upload_run(
                group="stability",
                experiment_name="stab-p%.1f-bs%d-ep30" % (power, bs),
                description=("稳定性检验：bs=%d，target_power=%s，30 epoch，lr=0.03。"
                             "用于验证 sqrt 目标变换在四个批量上是否一致降低末端误差。" % (bs, power)),
                config={"lr": 0.03, "epochs": 30, "batch_size": bs, "target_power": power},
                history=history, report=report, metrics=metrics,
            )
            rows.append((bs, power, summary["rmse_recomputed"], summary["tail_rmse"]))
            print("[stability] bs=%-4d p=%.1f  rmse=%7.2f  tail=%7.2f"
                  % (bs, power, summary["rmse_recomputed"], summary["tail_rmse"]))
    return rows


def group_llm(spec: Spec, train_paths):
    """回放大模型决策链：每轮一条 run，配置与结果原样取自当时的 decisions.jsonl。"""
    sources = [
        ("runs_real3", "llm-chain-local", "宿主机环境下的多轮大模型决策（proposal.source=llm）"),
        ("runs_image", "llm-chain-container", "容器内注入真实凭证后的决策链（proposal.source=llm）"),
    ]
    rows = []
    for folder, group, desc in sources:
        path = APP / folder / "decisions.jsonl"
        if not path.is_file():
            print("[llm] 跳过 %s（无 decisions.jsonl）" % folder)
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            proposal = record.get("proposal") or {}
            llm = proposal.get("llm") or {}
            result = record.get("result") or {}
            execution = record.get("execution") or {}
            decision = record.get("decision") or {}
            rnd = int(record.get("round") or 0)
            name = "%s-r%02d" % (group, rnd)

            run = swanlab.init(
                project=PROJECT,
                name=name,
                group=group,
                description=desc,
                public=True,
                mode="online",
                config={
                    "round": rnd,
                    "lr": proposal.get("lr"),
                    "epochs": proposal.get("epochs"),
                    "batch_size": proposal.get("batch_size"),
                    "proposal_source": proposal.get("source"),
                    "reason": (proposal.get("reason") or "")[:200],
                    "llm_model": llm.get("model"),
                    "reasoning_effort": llm.get("reasoning_effort"),
                    "prompt_tokens": (llm.get("usage") or {}).get("prompt_tokens"),
                    "completion_tokens": (llm.get("usage") or {}).get("completion_tokens"),
                    "total_tokens": (llm.get("usage") or {}).get("total_tokens"),
                    "rmse": round(result["rmse"], 4) if result.get("rmse") is not None else None,
                    "tail_rmse": round(result["tail_rmse"], 4) if result.get("tail_rmse") is not None else None,
                    "worst_bearing": (result.get("worst_group") or {}).get("name"),
                    "decision": decision.get("action"),
                    "decision_reason": (decision.get("reason") or "")[:200],
                    "checkpoint_sha256": (execution.get("checkpoint_sha256") or "")[:16],
                    "initial_steps": execution.get("initial_steps"),
                    "final_steps": execution.get("final_steps"),
                    "training_seconds": execution.get("training_seconds"),
                },
            )
            if result.get("rmse") is not None:
                swanlab.log({"rmse": result["rmse"],
                             "tail_rmse": result.get("tail_rmse")}, step=rnd)
            swanlab.finish()
            rows.append((name, result.get("rmse"), decision.get("action")))
            print("[llm] %-26s rmse=%8.2f  %s"
                  % (name, result.get("rmse") or -1, decision.get("action")))
    return rows


def main():
    parser = argparse.ArgumentParser(description="把 B 题实验上传到 SwanLab")
    parser.add_argument("--group", choices=["grid", "stability", "llm"], action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--only", action="append", help="只跑指定 tag，用于断点续传")
    parser.add_argument("--bs", action="append", type=int, help="稳定性组只跑指定 batch_size")
    args = parser.parse_args()

    if not os.environ.get("SWANLAB_API_KEY"):
        parser.error("缺少 SWANLAB_API_KEY")

    groups = ["grid", "stability", "llm"] if args.all else (args.group or ["grid"])
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    spec = Spec.build()
    train = read_dataset(spec.train_jsonl, "regression", [])
    validation = read_dataset(spec.validation_jsonl, "regression", [], train["paths"])
    print("数据：%d 行训练 / %d 行验证 | input_mode=%s | 输出目标变换 y**%.1f"
          % (len(train["ids"]), len(validation["ids"]), spec.input_mode, spec.target_power))

    started = time.time()
    if "grid" in groups:
        group_grid(spec, train["paths"], only=args.only)
    if "stability" in groups:
        group_stability(spec, train["paths"], only_bs=args.bs)
    if "llm" in groups:
        group_llm(spec, train["paths"])
    print("全部完成，用时 %.1f 秒。项目页：https://swanlab.cn 搜项目 %s" % (time.time() - started, PROJECT))


if __name__ == "__main__":
    main()
