"""Observer：只看事实——数据规模、标签分布、分组、起点指标、历史最好。

三题共用一套观测字段，回归题额外给出 RUL 分布与尾段规模，分类题给出类别分布与不均衡度。
B 的字段名一个没删（`train_bearings`、`best_rmse` 等），因为日志已被引用进文档与判读器。
"""
from pathlib import Path
import json
import numpy as np

from baselines.model import Model
from harness import dataset_cache
from harness.profiles import group_of


def observe(spec, state):
    train = dataset_cache.read(spec.train_jsonl, spec.task, spec.classes)
    groups = {}
    for sample_id in train["ids"]:
        key = group_of(sample_id, spec.profile)
        groups[key] = groups.get(key, 0) + 1

    facts = {
        "profile": spec.profile,
        "task": spec.task,
        "n_train": len(train["ids"]),
        "n_features": len(train["paths"]),
        # B 原来叫 train_bearings，保留同名；A/C 的语义由 profile 字段说明
        "train_bearings": len(groups),
        "train_groups": len(groups),
        "largest_bearing_share": round(max(groups.values()) / max(len(train["ids"]), 1), 4),
        "has_validation": spec.has_validation,
        "input_mode": Model.load(spec.ckpt).meta.get("input_mode"),
    }

    if spec.task == "regression":
        facts.update({
            "train_rul_min": float(train["y"].min()),
            "train_rul_mean": float(train["y"].mean()),
            "train_rul_max": float(train["y"].max()),
        })
    else:
        counts = np.bincount(train["y"], minlength=max(len(spec.classes), 1))
        facts["train_label_counts"] = {name: int(counts[i]) for i, name in enumerate(spec.classes)}
        facts["train_imbalance"] = round(float(counts.max()) / max(float(counts.mean()), 1e-9), 2)

    if spec.has_validation:
        validation = dataset_cache.read(spec.validation_jsonl, spec.task, spec.classes, train["paths"])
        facts["n_val"] = len(validation["ids"])
        facts["val_groups"] = len({group_of(i, spec.profile) for i in validation["ids"]})
        if spec.task == "regression":
            facts["val_rul_mean"] = float(validation["y"].mean())
            facts["val_bearings"] = facts["val_groups"]
        else:
            counts = np.bincount(validation["y"], minlength=max(len(spec.classes), 1))
            facts["val_label_counts"] = {name: int(counts[i]) for i, name in enumerate(spec.classes)}

    facts["start_checkpoint"] = str(spec.ckpt)
    facts["primary_metric"] = spec.primary_metric
    facts["higher_is_better"] = spec.higher_is_better
    # B 沿用 best_rmse；A/C 用 best_score。两者同时给出，避免下游拿错字段。
    facts["best_rmse"] = state.get("best_rmse")
    facts["best_score"] = state.get("best_score", state.get("best_rmse"))
    facts["best_tail_rmse"] = state.get("best_tail_rmse")
    # 当前最优若是「权重平均」点，planner 不会把它当续训起点（见 executor.is_aggregate）
    from harness.executor import is_aggregate
    facts["best_is_aggregate"] = bool(state.get("best_checkpoint")) and is_aggregate(state["best_checkpoint"])
    facts["rounds_used"] = state.get("rounds_used", 0)
    facts["epochs_used"] = state.get("epochs_used", 0)
    facts["epochs_left"] = max(spec.epoch_budget - facts["epochs_used"], 0)

    history = Path(spec.run_dir) / "decisions.jsonl"
    if history.is_file():
        rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
        facts["history_rounds"] = len(rows)
        facts["history_rmse"] = [(row.get("result") or {}).get("rmse") for row in rows]
        facts["history_score"] = [(row.get("result") or {}).get("primary") for row in rows]
        accepted = [row for row in rows if row.get("decision", {}).get("action") == "accept"]
        facts["history_accept_rate"] = round(len(accepted) / len(rows), 3) if rows else None
        # 已试过的配置和结果：不给这条，模型会反复提同一个被回滚的配置，
        # 同配置从同一回滚点重训结果逐位相同，纯烧预算。
        facts["history_trials"] = [
            {"lr": (row.get("proposal") or {}).get("lr"),
             "epochs": (row.get("proposal") or {}).get("epochs"),
             "batch_size": (row.get("proposal") or {}).get("batch_size"),
             "score": (row.get("result") or {}).get("primary"),
             "rmse": (row.get("result") or {}).get("rmse"),
             "action": (row.get("decision") or {}).get("action")}
            for row in rows]
    else:
        facts["history_rounds"] = 0
        facts["history_rmse"] = []
        facts["history_score"] = []
        facts["history_accept_rate"] = None
        facts["history_trials"] = []
    return facts
