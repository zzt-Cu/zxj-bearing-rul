"""Evaluator：不信任训练日志里的自报指标，独立加载检查点重算一遍。

三题指标口径不同，但都统一产出 `primary` 字段，门控只看这一个值：
  A  macro_f1   （越高越好）  另给 accuracy、六类 F1、混淆矩阵
  B  rmse       （越低越好）  另给 tail_rmse（最差阶段）、worst_group（单根退化）
  C  rmse       （越低越好）  另给 mae、NASA 非对称分数（高估惩罚更重）

B 的原有字段一个没改，`tail_rmse` / `worst_group` / `groups` 名字保持原样。
"""
from pathlib import Path
import numpy as np

from baselines.model import Model
from harness import dataset_cache
from harness.profiles import group_of


def _nasa_score(predicted, truth):
    """NASA C-MAPSS 非对称评分：高估（pred>truth）罚得更重。

    d = pred - truth；d<0 用 exp(-d/13)-1，d>=0 用 exp(d/10)-1。逐样本求和。
    只在 C 上计算：B 的单位是分钟、评分项不同，不能套用。
    """
    difference = np.asarray(predicted, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    score = np.where(difference < 0,
                     np.exp(-difference / 13.0) - 1.0,
                     np.exp(difference / 10.0) - 1.0)
    return float(score.sum())


def _classification_metrics(model, data):
    scores = np.asarray(model.predict(data["x"]), dtype=np.float64)
    predicted = scores.argmax(axis=1)
    truth = data["y"]
    classes = list(model.meta.get("classes") or [])
    n_classes = max(len(classes), 1)

    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for actual, guess in zip(truth, predicted):
        confusion[int(actual), int(guess)] += 1

    per_class = {}
    f1_values = []
    for index, name in enumerate(classes):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - tp)
        fn = float(confusion[index, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[name] = {"precision": round(precision, 6), "recall": round(recall, 6),
                           "f1": round(f1, 6), "support": int(confusion[index, :].sum())}
        f1_values.append(f1)

    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    worst_class = min(per_class.items(), key=lambda kv: kv[1]["f1"]) if per_class else None
    return {
        "primary": macro_f1,
        "macro_f1": macro_f1,
        "accuracy": float((predicted == truth).mean()),
        "n_val": int(len(truth)),
        "per_class": per_class,
        "worst_class": None if worst_class is None else {"name": worst_class[0], **worst_class[1]},
        "confusion": confusion.tolist(),
        "classes": classes,
    }


def _regression_metrics(spec, model, data):
    predicted = model.predict(data["x"])
    truth = data["y"]
    error = predicted - truth

    rmse = float(np.sqrt(np.mean(error ** 2)))
    metrics = {
        "primary": rmse,
        "rmse": rmse,
        "mae": float(np.mean(np.abs(error))),
        "n_val": int(len(truth)),
        "pred_min": float(predicted.min()),
        "pred_max": float(predicted.max()),
        "pred_negative": int((predicted < 0).sum()),
    }

    if spec.profile == "C":
        metrics["nasa_score"] = _nasa_score(predicted, truth)
        metrics["over_estimate_rate"] = float((error > 0).mean())

    # 尾段只在 B 上算：那里的评分项是「最差阶段与恢复速度」。
    if spec.tail_minutes:
        tail = truth < spec.tail_minutes
        if tail.any():
            metrics["tail_rmse"] = float(np.sqrt(np.mean(error[tail] ** 2)))
            metrics["tail_mae"] = float(np.mean(np.abs(error[tail])))
            metrics["tail_n"] = int(tail.sum())
        else:
            metrics["tail_rmse"] = None
            metrics["tail_n"] = 0

    groups = {}
    names = np.array([group_of(i, spec.profile) for i in data["ids"]])
    for name in sorted(set(names)):
        mask = names == name
        groups[str(name)] = {
            "n": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean(error[mask] ** 2))),
            "truth_mean": float(truth[mask].mean()),
            "pred_mean": float(predicted[mask].mean()),
        }
    metrics["groups"] = groups
    worst = max(groups.items(), key=lambda kv: kv[1]["rmse"])
    metrics["worst_group"] = {"name": worst[0], **worst[1]}
    return metrics


def evaluate(spec, checkpoint_path, train_paths):
    if not spec.has_validation:
        return None
    data = dataset_cache.read(spec.validation_jsonl, spec.task, spec.classes, train_paths)
    model = Model.load(Path(checkpoint_path))
    if spec.task == "classification":
        return _classification_metrics(model, data)
    return _regression_metrics(spec, model, data)
