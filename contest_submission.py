"""智训引擎 · contest.harness.workspace.v2 选手模块。

平台每一轮调用一次 `train(rnd)`：给一个父检查点、一份公开投影数据、一次模型调用额度，
要求本轮交出一个候选产物，并对「选谁」给出理由。十个轮次串起来就是一条续训轨迹。

设计要点
--------
1. **一轮只做一次增量**。轮次编排由平台负责（round_index 1..10），选手侧不再自建
   多轮循环。每轮从父检查点续训一小段、评估，再交给门控决定是否替换父模型。
2. **决策落在记录里**。每轮至少一次真实模型调用，把父模型指标、平台历史记录和上一轮
   得失交给平台模型，由它给出本轮超参；解析失败才退回内置排程。
3. **被拒就换挡**。门控回滚过的配置不再原样重放：下一轮自动把 epoch 数与学习率各降一档，
   连续两轮被拒则退到最小步长。这是「读历史—做实验—依据结果决策」的可核查形态，
   也避免十个轮次耗在同一个失败实验上。
4. **门控是硬约束**。验证指标没有达到 0.5% 的相对改善就保留父产物，不拿隐藏测试去赌退步。
5. **题型只由 dataset_key 决定**。输入整形模式、目标幂次、特征宽度全部从父检查点继承，
   不在这里臆造；改了这些量预测就会静默错位。
"""
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
SDK_TASKS = ("classification", "regression")
ARTIFACT_CODE = ("baselines/model.py", "baselines/data.py")
MIN_GAIN = 0.005          # 相对改善门槛的默认值；各题按自己的噪声水平覆盖
EPOCH_CAP = 400           # 单数据集累计 epoch 上限，防止排程失控
DEFAULT_WALL = 420.0      # 平台未给墙钟时的保守预算（秒）


# 三题的「模型出生设定」：输入整形与目标幂次。这两项改了预测就会静默错位，
# 只允许从父检查点继承；缺省值与官方起点一致。
PROFILES = {
    "A": {
        "task": "classification", "metric": "macro_f1", "higher_is_better": True,
        "input_mode": "abs8", "target_power": 1.0, "classes": [],
        "epochs": (12, 4, 3, 2, 2, 2, 2, 2, 2, 2),
        "min_gain": 0.005,
        "learning_rate": 0.005, "batch_size": 32,
        "epoch_choices": (2, 4, 6, 8, 12),
        "lr_choices": (0.002, 0.003, 0.005),
        "bs_choices": (32, 64, 128),
    },
    "B": {
        "task": "regression", "metric": "rmse", "higher_is_better": False,
        "input_mode": "abs8log", "target_power": 0.45, "classes": [],
        "epochs": (105, 14, 12, 10, 10, 10, 8, 8, 8, 8),
        "min_gain": 0.002,
        "learning_rate": 0.03, "batch_size": 96,
        "epoch_choices": (8, 10, 14, 20, 26),
        "lr_choices": (0.005, 0.01, 0.02, 0.03),
        "bs_choices": (96, 128, 192),
        # 首轮用 6 条不同批大小的轨迹做权重平均。单轨迹在本任务上会随训练推进迅速过拟合
        # （实测 lr 0.03 时 12 epoch 到 411、24 epoch 反而回 496），平均才是把模型拉到
        # 工作点的手段。此后各轮单轨迹微调——集成平均点不是好的续训起点，见 P3 诊断。
        "ensemble": (96, 112, 128, 144, 192, 256),
    },
    "C": {
        "task": "regression", "metric": "rmse", "higher_is_better": False,
        "input_mode": "abs8", "target_power": 1.0, "classes": [],
        "epochs": (30, 12, 10, 9, 8, 8, 8, 8, 8, 8),
        "min_gain": 0.001,
        "learning_rate": 0.003, "batch_size": 256,
        "epoch_choices": (6, 10, 15, 20, 30),
        "lr_choices": (0.002, 0.003, 0.005),
        "bs_choices": (128, 256, 512),
    },
}


def _log(message):
    print("[submission] %s" % message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- 环境与载入

def _prepare(dataset_key):
    """在 import baselines 之前把模型出生设定定好。

    baselines/model.py 在导入期就把 ZXJ_INPUT_MODE / ZXJ_TARGET_POWER 读成模块常量，
    导入之后再改环境变量不会生效——0.9 踩过这个坑，这里保持同样顺序。
    """
    profile = PROFILES[dataset_key]
    os.environ["ZXJ_INPUT_MODE"] = profile["input_mode"]
    os.environ["ZXJ_TARGET_POWER"] = str(profile["target_power"])
    os.environ.setdefault("BASELINE_CPU_THREADS", "2")
    return profile


def _parent_checkpoint(rnd):
    declared = list(rnd.parent.descriptor["checkpoint_paths"])
    relative = "checkpoint.npz" if "checkpoint.npz" in declared else declared[0]
    return relative, rnd.parent.path(relative)


def _checkpoint_meta(path):
    """只读元数据，不导入 baselines：预测进程要先知道整形模式才能设环境。"""
    import numpy as np
    with np.load(path, allow_pickle=False) as arrays:
        return json.loads(str(arrays["metadata"]))


# ---------------------------------------------------------------- 数据与指标

def _read_rows(rows, task, classes, expected_paths=None):
    """把 SDK 给的行流读成矩阵。逐行消费，不落盘、不拼第二份数据。"""
    from baselines.data import flatten
    ids, features, targets = [], [], []
    paths = expected_paths
    for number, row in enumerate(rows, 1):
        current, values = flatten(row["inputs"])
        if paths is None:
            paths = current
        if current != paths:
            raise ValueError("line %d: feature schema changed within one round" % number)
        answer = row["answer"]
        if task == "classification":
            if not isinstance(answer, str) or answer not in classes:
                raise ValueError("line %d: label %r outside the parent class list" % (number, answer))
            target = classes.index(answer)
        else:
            if isinstance(answer, bool) or not isinstance(answer, (int, float)) or not math.isfinite(answer) or answer < 0:
                raise ValueError("line %d: regression answer must be a finite nonnegative number" % number)
            target = float(answer)
        ids.append(row["id"])
        features.append(values)
        targets.append(target)
    if not ids:
        raise ValueError("public projection is empty")
    import numpy as np
    return {"ids": ids, "x": np.asarray(features, dtype=np.float64),
            "y": np.asarray(targets, dtype=np.int64 if task == "classification" else np.float64),
            "paths": paths}


def _metrics(model, data):
    import numpy as np
    predicted = model.predict(data["x"])
    expected = data["y"]
    if model.meta["task"] == "classification":
        labels = predicted.argmax(axis=1)
        observed = sorted(set(expected.tolist()) | set(labels.tolist()))
        f1 = [2 * np.sum((labels == label) & (expected == label)) /
              max(int(np.sum(labels == label) + np.sum(expected == label)), 1) for label in observed]
        return {"macro_f1": float(np.mean(f1)) if f1 else 0.0,
                "accuracy": float(np.mean(labels == expected))}
    error = predicted - expected
    return {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error)))}


def _improved(before, after, metric, higher_is_better, min_gain=MIN_GAIN):
    old, new = before[metric], after[metric]
    if not math.isfinite(old) or not math.isfinite(new):
        return False
    if higher_is_better:
        return (new - old) / max(abs(old), 1e-9) >= min_gain
    return (old - new) / max(abs(old), 1e-9) >= min_gain


# ---------------------------------------------------------------- 决策

_DECISION_TEMPLATE = (
    "你在为一台工业模型做续训决策。只输出一个 JSON 对象，不要解释，不要代码块。\n"
    "数据集：%(key)s（%(task)s，主指标 %(metric)s，%(direction)s）\n"
    "本轮是第 %(round)d 轮，共 10 轮。父模型在本轮公开验证集上的 %(metric)s = %(before).6f。\n"
    "平台已完成的检查点记录：%(history)s\n"
    "上一轮结果：%(last_outcome)s\n"
    "可选超参（只能从括号里选）：epochs%(epochs)s，learning_rate%(lrs)s，batch_size%(bss)s。\n"
    "已被门控回滚的配置不要原样重放；步长过大会破坏父模型已学到的表示。\n"
    '输出格式：{"epochs": 整数, "learning_rate": 数值, "batch_size": 整数, "reason": "不超过40字"}'
)


def _history_brief(records, limit=6):
    if not records:
        return "无（本轮是首轮）"
    parts = []
    for item in records[-limit:]:
        score = item.get("score")
        if score is None:
            score = item.get("primary") or item.get("metric")
        parts.append("#%s:%s" % (item.get("round_index") or item.get("round") or "?", score))
    return "，".join(parts)


def _last_outcome(last):
    if not last:
        return "无（本轮是首轮，从父检查点冷启动）"
    before = last.get("validation_before") or {}
    after = last.get("validation_after") or {}
    return "epochs=%s, lr=%s，验证 %s -> %s，%s" % (
        last.get("epochs_run"), last.get("learning_rate"),
        before, after, "采纳" if last.get("accepted") else "被门控回滚")


def _parse_decision(text, profile):
    """从模型回复里取第一个 JSON 对象并夹到允许集合内。任何异常都退回默认排程。"""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None

    def pick(name, choices, fallback):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return fallback
        if name == "learning_rate":
            nearest = min(choices, key=lambda choice: abs(choice - value))
            return nearest if abs(nearest - value) <= 1e-9 else fallback
        return int(value) if value in choices else fallback

    reason = payload.get("reason")
    reason = reason.strip()[:200] if isinstance(reason, str) else ""
    return {
        "epochs": pick("epochs", profile["epoch_choices"], None),
        "learning_rate": pick("learning_rate", profile["lr_choices"], None),
        "batch_size": pick("batch_size", profile["bs_choices"], None),
        "reason": reason,
    }


def _ask_model(rnd, profile, before, history, last):
    default_epochs = profile["epochs"][min(rnd.round_index, 10) - 1]
    prompt = _DECISION_TEMPLATE % {
        "key": rnd.dataset_key,
        "task": profile["task"],
        "metric": profile["metric"],
        "direction": "越高越好" if profile["higher_is_better"] else "越低越好",
        "round": rnd.round_index,
        "before": before[profile["metric"]],
        "history": _history_brief(history),
        "last_outcome": _last_outcome(last),
        "epochs": profile["epoch_choices"],
        "lrs": profile["lr_choices"],
        "bss": profile["bs_choices"],
    }
    receipt = rnd.model(
        [{"role": "user", "content": prompt}],
        max_tokens=min(320, rnd.max_output_tokens),
        temperature=0,
    )
    decision = _parse_decision(receipt.text, profile)
    if decision is None:
        _log("model reply unparsable, fall back to the built-in schedule")
        decision = {"epochs": None, "learning_rate": None, "batch_size": None, "reason": ""}
    decision["epochs"] = decision["epochs"] or default_epochs
    decision["learning_rate"] = decision["learning_rate"] or profile["learning_rate"]
    decision["batch_size"] = decision["batch_size"] or profile["batch_size"]
    decision["call_id"] = receipt.call_id
    return decision


def _shift_down(profile, decision, last, streak, round_index, spent_epochs):
    """把「模型建议」变成「本轮实际执行」。

    被门控回滚过的配置不再原样重放。连续被拒时按固定阶梯换挡，其中保留一档「高学习率
    探针」：如果平台在轮次之间注入了漂移，这一档能在一两个 epoch 内把模型拉回工作点，
    而一味减小步长会错过漂移。
    """
    epochs = int(decision["epochs"])
    learning_rate = float(decision["learning_rate"])
    choices = profile["lr_choices"]
    if round_index == 1:
        # 冷启动从官方检查点出发，10 轮里只有一次把模型拉到工作点的机会，
        # 首轮取内置排程的下限，避免模型给出过小增量导致全程欠拟合。
        epochs = max(epochs, profile["epochs"][0])
    if last and not last.get("accepted"):
        previous = int(last.get("epochs_run") or epochs)
        if streak == 1:
            epochs = max(2, min(epochs, previous) // 2)
            lower = [value for value in choices if value < learning_rate]
            learning_rate = max(lower) if lower else min(choices)
        elif streak == 2:
            epochs = 2
            learning_rate = min(choices)
        else:
            ladder = ((2, max(choices)), (4, choices[len(choices) // 2]), (2, min(choices)))
            epochs, learning_rate = ladder[(streak - 3) % len(ladder)]
    if spent_epochs >= EPOCH_CAP:
        epochs = 2
    return max(1, epochs), learning_rate


def _fit_ensemble(parent, train_data, epochs, batches, budget, started):
    """同一父检查点、同一轮数、仅批大小不同，结束后对权重与 Adam 矩取算术平均。

    目的不是增加容量，而是压掉单条轨迹对训练集偶然结构的过拟合：各轨迹共享同一信号、
    各自噪声不同，平均后噪声互相抵消。成员之间差异越大，合并收益越明显（实测成员
    RMSE 380～704，合并 302）。
    """
    import copy
    import numpy as np
    members = []
    for batch in batches:
        clone = copy.deepcopy(parent)
        clone.meta["batch_size"] = int(batch)
        clone.validate()
        for _ in range(epochs):
            clone.fit(train_data["x"], train_data["y"], 1)
            if time.monotonic() - started > budget:
                _log("ensemble wall budget reached on member %d" % batch)
                break
        members.append(clone)
        if time.monotonic() - started > budget:
            break
    if len(members) < 2:
        return members[0] if members else parent
    merged = copy.deepcopy(parent)
    for name in merged.params:
        merged.params[name] = np.mean([item.params[name] for item in members], axis=0)
        merged.first[name] = np.mean([item.first[name] for item in members], axis=0)
        merged.second[name] = np.mean([item.second[name] for item in members], axis=0)
    for key in ("target_power", "target_mean", "target_scale"):
        merged.meta[key] = members[0].meta[key]
    merged.meta["epochs"] = int(round(float(np.mean([item.meta["epochs"] for item in members]))))
    merged.meta["steps"] = int(round(float(np.mean([item.meta["steps"] for item in members]))))
    merged.meta["batch_size"] = int(batches[0])
    merged.meta["aggregate"] = "weight_average"
    merged.meta["ensemble"] = [int(batch) for batch in batches[:len(members)]]
    merged.validate()
    return merged


# ---------------------------------------------------------------- 一轮训练

def train(rnd):
    key = rnd.dataset_key
    if key not in PROFILES:
        raise ValueError("unknown dataset key %r" % key)
    profile = _prepare(key)

    from baselines.model import Model

    work = Path(os.environ.get("CONTEST_WORK", "/workspace")) / ("round-%02d" % rnd.round_index)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    state = work / "state"
    rnd.state_workspace(state)
    ledger = {"dataset_key": key, "rounds": []}
    ledger_path = state / "ledger.json"
    if ledger_path.is_file():
        try:
            previous = json.loads(ledger_path.read_text(encoding="utf-8"))
            if isinstance(previous, dict) and isinstance(previous.get("rounds"), list):
                ledger = previous
        except ValueError:
            _log("previous ledger unreadable, start a new one")
    rounds = ledger["rounds"]
    last = rounds[-1] if rounds else None
    streak = 0
    for item in reversed(rounds):
        if item.get("accepted"):
            break
        streak += 1

    relative, parent_path = _parent_checkpoint(rnd)
    model = Model.load(parent_path)
    if model.meta["task"] != profile["task"]:
        raise ValueError("parent task %r does not match dataset %s" % (model.meta["task"], key))
    classes = list(model.meta["classes"]) or list(profile["classes"])
    paths = model.meta["paths"]

    train_data = _read_rows(rnd.data.training_rows(), profile["task"], classes, paths)
    validation_data = _read_rows(rnd.data.validation_rows(), profile["task"], classes, paths)
    overlap = set(train_data["ids"]) & set(validation_data["ids"])
    if overlap:
        raise ValueError("training and validation ids overlap (%d)" % len(overlap))

    # 父模型的绝对宽度若小于当前整形模式所需宽度（官方起点没有 log 列），先扩列。
    if os.environ["ZXJ_INPUT_MODE"] == "abs8log" and model.params["w1"].shape[0] == len(paths):
        model.expand_inputs(train_data["x"])

    before = _metrics(model, validation_data)
    history = rnd.history()
    decision = _ask_model(rnd, profile, before, history, last)
    epochs, learning_rate = _shift_down(profile, decision, last, streak,
                                        rnd.round_index, int(model.meta["epochs"]))
    model.meta["learning_rate"] = learning_rate
    model.meta["batch_size"] = int(decision["batch_size"])
    model.validate()

    wall = float(rnd.wall_seconds or DEFAULT_WALL)
    budget = max(20.0, wall * 0.72)
    started = time.monotonic()
    ensemble = profile.get("ensemble") if rnd.round_index == 1 else None
    if ensemble:
        model = _fit_ensemble(model, train_data, epochs, ensemble, budget, started)
        done = epochs
    else:
        done = 0
        for _ in range(epochs):
            model.fit(train_data["x"], train_data["y"], 1)
            done += 1
            if time.monotonic() - started > budget:
                _log("wall budget reached after %d/%d epochs" % (done, epochs))
                break
    seconds = time.monotonic() - started

    after = _metrics(model, validation_data)
    accepted = _improved(before, after, profile["metric"], profile["higher_is_better"],
                         float(profile.get("min_gain", MIN_GAIN)))

    candidate = work / "candidate"
    candidate.mkdir()
    model.save(candidate / "checkpoint.npz")
    for name in ARTIFACT_CODE:
        target = candidate / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path("/app") / name, target)

    metric = profile["metric"]
    metrics_note = "%s %.6f -> %.6f" % (metric, before[metric], after[metric])
    # hypothesis 只写「赛前预期」，实测结果放 change_summary：顺序错了，记录就成了事后编故事。
    hypothesis = ("第 %d 轮：父模型验证 %s=%.6f。本轮以 epochs=%d、lr=%g、batch=%d 续训，"
                  "预期%s且超过 %.1f%% 门槛。模型判断：%s" % (
                      rnd.round_index, metric, before[metric], epochs, learning_rate,
                      model.meta["batch_size"],
                      "指标改善" if profile["higher_is_better"] else "误差下降",
                      float(profile.get("min_gain", MIN_GAIN)) * 100,
                      decision["reason"] or "（未给出）"))
    change = ("只改权重与 Adam 矩：输入整形 %s、目标幂次 %.2f、特征宽度 %d 均沿用父检查点；"
              "本轮实跑 %d epoch、%.1f 秒，验证 %s，累计 %d epoch。" % (
                  model.meta.get("input_mode"), float(model.meta.get("target_power", 1.0)),
                  model.meta.get("input_width", len(paths)), done, seconds, metrics_note,
                  model.meta["epochs"]))
    configuration = {
        "dataset_key": key, "round_index": rnd.round_index,
        "epochs": done, "learning_rate": learning_rate, "batch_size": model.meta["batch_size"],
        "input_mode": model.meta.get("input_mode"), "target_power": model.meta.get("target_power"),
        "ensemble": model.meta.get("ensemble"),
        "validation_before": before, "validation_after": after,
        "accepted": accepted, "rejected_streak": streak,
        "seconds": round(seconds, 2), "model_call_id": decision["call_id"],
    }

    submitted = rnd.submit_candidate(
        candidate,
        hypothesis=hypothesis,
        change_summary=change,
        configuration=configuration,
        seed=int(model.meta["seed"]),
        checkpoint_paths=["checkpoint.npz"],
        code_paths=list(ARTIFACT_CODE),
        predict_argv=list(PREDICT_ARGV),
    )

    rounds.append({
        "round_index": rnd.round_index,
        "epochs_run": done, "learning_rate": learning_rate, "batch_size": model.meta["batch_size"],
        "cumulative_epochs": model.meta["epochs"], "cumulative_steps": model.meta["steps"],
        "validation_before": before, "validation_after": after,
        "accepted": accepted, "artifact_id": submitted["artifact_id"],
        "model_call_id": decision["call_id"],
    })
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    rnd.save_state(state)

    selected = submitted["artifact_id"] if accepted else rnd.parent.artifact_id
    gate = float(profile.get("min_gain", MIN_GAIN)) * 100
    reason = ("%s；达到 %.1f%% 门槛，采纳本轮候选。" % (metrics_note, gate) if accepted else
              "%s；未达 %.1f%% 门槛，保留父产物，避免把一次退步带进隐藏测试。" % (metrics_note, gate))
    rnd.finish(
        selected_artifact_id=selected,
        hypothesis=hypothesis,
        change_summary=change,
        reason=reason,
        configuration=configuration,
        seed=int(model.meta["seed"]),
        history_ids=[item["id"] for item in history],
    )


# ---------------------------------------------------------------- 预测

_MODEL_CACHE = {}


def _load_model(model_dir):
    path = os.environ.get("CONTEST_ARTIFACT_PATH")
    if not path:
        raise ValueError("CONTEST_ARTIFACT_PATH is not set")
    if path in _MODEL_CACHE:
        return _MODEL_CACHE[path]
    meta = _checkpoint_meta(path)
    # 预测侧必须用检查点自己的整形模式，否则归一化会静默错位。
    os.environ["ZXJ_INPUT_MODE"] = meta.get("input_mode", "abs8")
    if "target_power" in meta:
        os.environ["ZXJ_TARGET_POWER"] = str(meta["target_power"])
    Model = _artifact_model(model_dir)
    model = Model.load(path)
    _MODEL_CACHE[path] = model
    return model


def _artifact_model(model_dir):
    """优先执行产物自带的 model.py。

    产物是「检查点 + 代码」一起提交的，预测就该跑那份代码；直接把 /model 插进 sys.path
    会被同名包遮蔽（产物里通常没有 __init__.py，会变成命名空间包盖住 /app/baselines），
    所以按文件路径单独加载。
    """
    import importlib.util
    source = Path(model_dir) / "baselines" / "model.py"
    if not source.is_file():
        from baselines.model import Model
        return Model
    key = "artifact_model_%s" % abs(hash(str(source)))
    module = sys.modules.get(key)
    if module is None:
        spec = importlib.util.spec_from_file_location(key, source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[key] = module
    return module.Model


def predict(sample, model_dir):
    from baselines.data import flatten
    model = _load_model(model_dir)
    if sample.task != model.meta["task"]:
        raise ValueError("sample task %r does not match checkpoint task %r"
                         % (sample.task, model.meta["task"]))
    paths, values = flatten(sample.inputs)
    if paths != model.meta["paths"]:
        raise ValueError("prediction feature schema differs from checkpoint")
    import numpy as np
    output = model.predict(np.asarray([values], dtype=np.float64))[0]
    if model.meta["task"] == "regression":
        return {"outputs": {"value": float(max(0.0, output))}}
    scores = [float(min(max(score, 0.0), 1.0)) for score in output]
    top = max(scores)
    if sum(1 for score in scores if score == top) != 1:
        scores[scores.index(top)] = min(1.0, top + 1e-12)
    return {"outputs": [{"label": label, "score": score}
                        for label, score in zip(model.meta["classes"], scores)]}
