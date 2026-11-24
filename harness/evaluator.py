"""Evaluator：不信任训练日志里的自报指标，独立加载检查点重算一遍。

多算两个量，因为它们对应平台的两个评分项：
  tail_rmse   —— 真实 RUL < tail_minutes 的样本，对应「最差阶段」
  worst_group —— 分轴承最差的那一根，用来防「总体涨分但某根崩了」
"""
from pathlib import Path
import numpy as np

from baselines.data import read_dataset
from baselines.model import Model


def _bearing_of(sample_id):
    head, sep, _ = sample_id.rpartition("_s")
    return head if sep else sample_id


def evaluate(spec, checkpoint_path, train_paths):
    if not spec.has_validation:
        return None
    data = read_dataset(spec.validation_jsonl, "regression", [], train_paths)
    model = Model.load(Path(checkpoint_path))
    predicted = model.predict(data["x"])
    truth = data["y"]
    error = predicted - truth

    metrics = {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "n_val": int(len(truth)),
        "pred_min": float(predicted.min()),
        "pred_max": float(predicted.max()),
    }

    tail = truth < spec.tail_minutes
    if tail.any():
        metrics["tail_rmse"] = float(np.sqrt(np.mean(error[tail] ** 2)))
        metrics["tail_mae"] = float(np.mean(np.abs(error[tail])))
        metrics["tail_n"] = int(tail.sum())
    else:
        metrics["tail_rmse"] = None
        metrics["tail_n"] = 0

    groups = {}
    bearings = np.array([_bearing_of(i) for i in data["ids"]])
    for name in sorted(set(bearings)):
        mask = bearings == name
        groups[name] = {
            "n": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean(error[mask] ** 2))),
            "truth_mean": float(truth[mask].mean()),
            "pred_mean": float(predicted[mask].mean()),
        }
    metrics["groups"] = groups
    worst = max(groups.items(), key=lambda kv: kv[1]["rmse"])
    metrics["worst_group"] = {"name": worst[0], **worst[1]}
    return metrics
