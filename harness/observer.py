"""Observer：只看事实——数据规模、标签分布、轴承分组、起点指标、历史最好。"""
from pathlib import Path
import json
import numpy as np

from baselines.data import read_dataset
from baselines.model import Model


def _bearing_of(sample_id):
    """xjtu_Bearing1_1_s000123 -> xjtu_Bearing1_1；无该结构时退回整串。"""
    head, sep, _ = sample_id.rpartition("_s")
    return head if sep else sample_id


def observe(spec, state):
    train = read_dataset(spec.train_jsonl, "regression", [])
    groups = {}
    for sample_id in train["ids"]:
        groups[_bearing_of(sample_id)] = groups.get(_bearing_of(sample_id), 0) + 1

    facts = {
        "n_train": len(train["ids"]),
        "n_features": len(train["paths"]),
        "train_rul_min": float(train["y"].min()),
        "train_rul_mean": float(train["y"].mean()),
        "train_rul_max": float(train["y"].max()),
        "train_bearings": len(groups),
        "largest_bearing_share": round(max(groups.values()) / max(len(train["ids"]), 1), 4),
        "has_validation": spec.has_validation,
        "input_mode": Model.load(spec.ckpt).meta.get("input_mode"),
    }

    if spec.has_validation:
        validation = read_dataset(spec.validation_jsonl, "regression", [], train["paths"])
        facts["n_val"] = len(validation["ids"])
        facts["val_rul_mean"] = float(validation["y"].mean())
        facts["val_bearings"] = len({_bearing_of(i) for i in validation["ids"]})

    facts["start_checkpoint"] = str(spec.ckpt)
    facts["best_rmse"] = state.get("best_rmse")
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
        accepted = [row for row in rows if row.get("decision", {}).get("action") == "accept"]
        facts["history_accept_rate"] = round(len(accepted) / len(rows), 3) if rows else None
        # 已试过的配置和结果：不给这条，模型会反复提同一个被回滚的配置，
        # 同配置从同一回滚点重训结果逐位相同，纯烧预算。
        facts["history_trials"] = [
            {"lr": (row.get("proposal") or {}).get("lr"),
             "epochs": (row.get("proposal") or {}).get("epochs"),
             "batch_size": (row.get("proposal") or {}).get("batch_size"),
             "rmse": round((row.get("result") or {}).get("rmse") or 0, 2),
             "action": (row.get("decision") or {}).get("action")}
            for row in rows]
    else:
        facts["history_rounds"] = 0
        facts["history_rmse"] = []
        facts["history_accept_rate"] = None
        facts["history_trials"] = []
    return facts
