"""Executor：跑一轮续训。只调用官方 train_run，不复制训练逻辑。"""
from pathlib import Path
import json
import os
import shutil

import numpy as np

from baselines.data import sha256
from baselines.train import train_run


def ensemble_sizes(spec):
    """镜像级固定策略：同一续训起点跑多条 batch_size 轨迹再平均权重，用于压掉单轨迹过拟合。

    取值来自 profile（B 是 96/112/128/144/192/256），可用 ENSEMBLE_BATCH 覆盖。
    只有一条时退化为单轨迹训练，行为与官方基线完全一致。
    A/C 默认不启用：它们的数据量与任务形态不需要这个手段，硬套只会增加解释成本。
    """
    sizes = [int(size) for size in (spec.ensemble or [])]
    return sizes if len(sizes) >= 2 else None


def is_aggregate(checkpoint):
    """检查点是否为「多轨迹权重平均」产物。

    这种点不能作为下一轮的续训起点：它的低误差来自 6 个函数平均，而不是权重本身
    落在某个好解上。从平均点再走一步，6 条轨迹会立刻塌回同一个单模型解，集成收益
    完全消失——实测 302.75 起训 3 轮（lr=1e-4）即回到 352.7，15 轮（lr=0.015）
    回到 517.9。证据见 docs/P3_集成点不可续训_诊断.md。
    """
    try:
        with np.load(checkpoint, allow_pickle=True) as data:
            meta = json.loads(str(data["metadata"]))
    except (OSError, ValueError, KeyError):
        return False
    return meta.get("aggregate") == "weight_average"


def run_round(spec, proposal, parent_checkpoint, round_index):
    out_dir = Path(spec.run_dir) / ("round-%02d" % round_index)
    if out_dir.exists():
        # 每轮必须是新目录；重跑时先归档而不是静默覆盖
        archived = out_dir.with_name(out_dir.name + ".old")
        if archived.exists():
            shutil.rmtree(archived)
        out_dir.rename(archived)

    # 不能从权重平均点继续训练：改从「官方起点」独立训练，让这一轮成为一次可比的独立实验，
    # 而不是在平均点上做一次必然劣化的续训。
    # 起点取 spec.resolve_start()：备份优先。因为采纳后 publish 会把 spec.ckpt 覆盖成
    # 权重平均点，直接读 spec.ckpt 会让这一轮的起点又变成平均点（P3 复发）。
    parent = str(parent_checkpoint)
    restarted = False
    if is_aggregate(parent_checkpoint):
        parent = str(spec.resolve_start())
        restarted = True

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    # B 的 stage_id 必须保持 `zxj-round-NN`：它进 checkpoint 元数据，改了会连带改
    # 发布指纹 c8bd6608…，而那个指纹已经写进已提交的文档里。A/C 才加题目前缀区分。
    prefix = spec.stage_prefix if spec.profile == "B" else "%s-%s" % (spec.stage_prefix, spec.profile.lower())
    stage_id = "%s-round-%02d" % (prefix, round_index)
    tracks = ensemble_sizes(spec)
    report = train_run(
        spec.profile,
        spec.train_jsonl,
        spec.validation_jsonl,
        out_dir,
        resume=parent,
        epochs=int(proposal["epochs"]),
        learning_rate=float(proposal["lr"]),
        batch_size=int(proposal["batch_size"]),
        ensemble=tracks,
        stage_id=stage_id,
    )
    checkpoint = out_dir / "checkpoint.npz"
    return {
        "out_dir": str(out_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint.read_bytes()),
        "parent_checkpoint_sha256": report.get("parent_checkpoint_sha256"),
        "requested_parent_sha256": sha256(Path(parent_checkpoint).read_bytes()),
        "restarted_from_start": restarted,
        "restart_from": parent,
        "restart_from_sha256": report.get("parent_checkpoint_sha256"),
        "ensemble_tracks": len(tracks or [1]),
        "stage_id": stage_id,
        "initial_steps": report.get("initial_steps"),
        "final_steps": report.get("final_steps"),
        "training_seconds": report.get("training_seconds"),
        "self_reported": (report.get("validation_after") or {}),
        "self_reported_rmse": (report.get("validation_after") or {}).get("rmse"),
        "self_reported_macro_f1": (report.get("validation_after") or {}).get("macro_f1"),
    }
