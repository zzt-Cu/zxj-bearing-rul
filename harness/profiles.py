"""三题 profile：判题、指标口径、门控参数。

单镜像要跑 A/B/C 三题，平台的提交位是三个、镜像只有一个。所以镜像必须靠**数据本身**
判断这一轮是哪一题，不能靠环境变量（平台未必给），也不能靠文件名（未必叫得一样）。

判据用 `inputs` 的结构，三题天然不同：

    A  inputs.features 是长度 128 的数组          -> 气体六分类
    B  inputs.features 是 {horizontal, vertical}  -> 轴承 RUL（分钟）
    C  inputs.settings(3) + inputs.sensors(21)    -> C-MAPSS RUL（循环）

B 的取值与 0.8 逐位一致：这次泛化只把写死的地方换成查表，没有改任何数值。
"""
from pathlib import Path
import json


# A 题的类别顺序取自任务书与 configs/A.json，顺序即标签索引，不能重排。
CLASSES_A = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


PROFILES = {
    "A": {
        "task": "classification",
        "primary": "macro_f1",
        "higher_is_better": True,
        "n_features": 128,
        "classes": CLASSES_A,
        # 分类没有「最差阶段/单轴承」这种评分项，分组退化检查只对 B 有意义。
        "group_check": False,
        # Macro-F1 是六类 F1 的平均，所以真正要防的是「均值涨、某一类崩」。
        "per_class_floor": 0.20,
        "min_gain": 0.005,
        "max_rounds": 6,
        "epoch_budget": 400,
        "wall_limit_s": 240.0,
        "tail_minutes": None,
        # 官方 A/C 检查点是 abs8 + target_power=1.0（见 checkpoint 元数据）。
        # 这两个值是「模型出生时的设定」，不是超参，改了预测就会静默错位。
        "input_mode": "abs8",
        "target_power": 1.0,
        "ensemble": [],
        "max_group_degrade": None,
        "min_tail_n": 0,
    },
    "B": {
        "task": "regression",
        "primary": "rmse",
        "higher_is_better": False,
        "n_features": 34,
        "classes": [],
        "group_check": True,
        "per_class_floor": None,
        "min_gain": 0.005,
        "max_rounds": 2,
        "epoch_budget": 330,
        "wall_limit_s": 390.0,
        "tail_minutes": 100.0,
        # 以下五项原先是 Dockerfile 的 ENV，现收进这张表；取值与 0.8 完全一致。
        "input_mode": "abs8log",
        "target_power": 0.45,
        "ensemble": [96, 112, 128, 144, 192, 256],
        "max_group_degrade": 0.25,
        "min_tail_n": 30,
        "tail_trade_gain": 0.20,
        "tail_trade_rmse": 0.06,
    },
    "C": {
        "task": "regression",
        "primary": "rmse",
        "higher_is_better": False,
        "n_features": 24,
        "classes": [],
        # C 的主指标是 RMSE，NASA 分数是「另报告」；不做分组退化检查，避免自造约束。
        "group_check": False,
        "per_class_floor": None,
        # C 的单轮成本最高（11 万行、每轮约 50 秒），轮次定为 4：
        # 6 轮会在 300 秒以上，逼近平台窗口；4 轮足够展示「采纳→回滚」的完整决策链。
        "min_gain": 0.005,
        "max_rounds": 4,
        "epoch_budget": 400,
        "wall_limit_s": 360.0,
        # 高估剩余寿命的惩罚更大，所以额外记 NASA 分数并设一条温和保护条款。
        "tail_minutes": None,
        "nasa_guard": 0.20,
        "input_mode": "abs8",
        "target_power": 1.0,
        "ensemble": [],
        "max_group_degrade": None,
        "min_tail_n": 0,
    },
}


def get(profile):
    if profile not in PROFILES:
        raise ValueError("unknown profile %r (expect A/B/C)" % (profile,))
    return PROFILES[profile]


def detect_from_inputs(inputs):
    """按一条样本的 inputs 结构判题；判不出返回 None。

    A  features 是数组（128 个气体传感器读数）
    B  features 是 {horizontal, vertical} 两通道振动统计量
    C  settings + sensors 分开给（3 项运行设置 + 21 项传感器）
    """
    if not isinstance(inputs, dict):
        return None
    if isinstance(inputs.get("settings"), list) and isinstance(inputs.get("sensors"), list):
        return "C"
    features = inputs.get("features")
    if isinstance(features, list):
        return "A"
    if isinstance(features, dict):
        return "B"
    return None


def detect_from_file(train_jsonl):
    """读训练集首行判题。读不到就返回 None，由调用方决定兜底。"""
    try:
        with open(train_jsonl, encoding="utf-8") as handle:
            line = handle.readline()
        row = json.loads(line)
    except (OSError, ValueError):
        return None
    return detect_from_inputs(row.get("inputs"))


def group_of(sample_id, profile):
    """把样本归到「一根轴承 / 一个气体批次 / 一台发动机」，用于退化检查与诊断。

    B  xjtu_Bearing1_4_s000012        -> xjtu_Bearing1_4
    A  gas_b02_000001                 -> gas_b02
    C  fd001_train_u0071_c00001       -> fd001_train_u0071
    """
    parts = str(sample_id).split("_")
    if profile == "B":
        head, sep, _ = str(sample_id).rpartition("_s")
        return head if sep else str(sample_id)
    if profile == "A":
        return "_".join(parts[:2]) if len(parts) >= 3 else str(sample_id)
    if profile == "C":
        return "_".join(parts[:3]) if len(parts) >= 4 else str(sample_id)
    return str(sample_id)


def default_ckpt(profile, app_dir):
    """每题的行动检查点。

    B 必须留在 /app/checkpoint.npz —— 平台按这个路径取权重，也是 0.8 已验证过的位置。
    A/C 的官方起点放在 baselines_ckpt/ 下，运行时复制成 checkpoint.start.npz，
    发布仍写回 /app/checkpoint.npz，这样平台无论从哪个路径取都拿得到本次的产物。
    """
    if profile == "B":
        return Path(app_dir) / "checkpoint.npz"
    return Path(app_dir) / "baselines_ckpt" / ("%s.npz" % profile)
