"""Harness 主循环：无人工干预，跑到预算用尽或轮次用尽为止。

用法（容器内外一致）：
    python -m harness.loop --max-rounds 3 --epoch-budget 100
    python -m harness.loop --dry-run          # 只观察与提案，不训练
"""
from pathlib import Path
import argparse
import json
import sys
import time

try:
    from harness import evaluator, executor, gatekeeper, observer, planner, recorder
    from harness import publish as publish_module
    from harness import spec as spec_module
except ImportError:  # 直接以脚本方式运行时的兜底
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from harness import evaluator, executor, gatekeeper, observer, planner, recorder
    from harness import publish as publish_module
    from harness import spec as spec_module

from baselines.data import read_dataset
from baselines.model import Model


def _emit(kind, payload):
    print("[harness] %s %s" % (kind, json.dumps(payload, ensure_ascii=False, allow_nan=False)), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="含 train.jsonl / validation.jsonl 的目录")
    parser.add_argument("--ckpt", help="起点检查点")
    parser.add_argument("--run-dir", help="运行目录")
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--epoch-budget", type=int)
    parser.add_argument("--min-gain", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    spec = spec_module.Spec.build(
        data_dir=Path(args.data) if args.data else None,
        ckpt=Path(args.ckpt) if args.ckpt else None,
        run_dir=Path(args.run_dir) if args.run_dir else None,
        max_rounds=args.max_rounds,
        epoch_budget=args.epoch_budget,
        min_gain=args.min_gain,
    )
    if not spec.train_jsonl.is_file():
        _emit("fatal", {"error": "train.jsonl not found", "data_dir": str(spec.data_dir)})
        return 2
    if not spec.ckpt.is_file():
        _emit("fatal", {"error": "checkpoint not found", "ckpt": str(spec.ckpt)})
        return 2

    Path(spec.run_dir).mkdir(parents=True, exist_ok=True)

    # 起点备份必须在任何训练之前：后面 publish 会覆盖 spec.ckpt，
    # 没有这份备份，第 2 轮就会把权重平均点当起点（P3 复发）。
    try:
        start_backup = publish_module.ensure_start_backup(spec.ckpt, spec.start_ckpt)
    except Exception as error:  # noqa: BLE001 - 备份失败只降级，不终止
        start_backup = {"created": False, "path": str(spec.ckpt), "sha256": None,
                        "error": "%s: %s" % (type(error).__name__, error)}
    publish_enabled = start_backup.get("sha256") is not None
    _emit("start_checkpoint", dict(start_backup, publish_enabled=publish_enabled))

    state = recorder.load_state(spec)
    state.setdefault("best_groups", {})
    state.setdefault("track_epochs_used", 0)
    state.setdefault("unverified_rounds", [])
    if not Path(state["best_checkpoint"]).is_file():
        state["best_checkpoint"] = str(spec.ckpt)

    train_paths = read_dataset(spec.train_jsonl, "regression", [])["paths"]
    facts = observer.observe(spec, state)
    _emit("spec", spec.as_dict())
    _emit("observe", facts)
    _emit("input_mode", {"checkpoint": Model.load(spec.ckpt).meta.get("input_mode"),
                         "process_env": spec.input_mode})

    if args.dry_run:
        policy = planner._load_policy()
        _emit("propose", planner.propose(facts, policy) or {"proposal": None, "reason": "预算不足"})
        return 0

    started = time.monotonic()
    policy = planner._load_policy()
    while True:
        if state["rounds_used"] >= spec.max_rounds:
            _emit("stop", {"reason": "达到轮次上限", "max_rounds": spec.max_rounds})
            break
        if state["epochs_used"] >= spec.epoch_budget:
            _emit("stop", {"reason": "epoch 预算用尽", "epoch_budget": spec.epoch_budget})
            break
        if time.monotonic() - started > spec.wall_limit_s:
            _emit("stop", {"reason": "墙钟上限", "wall_limit_s": spec.wall_limit_s})
            break

        proposal = planner.propose(facts, policy)
        if not proposal:
            _emit("stop", {"reason": "无可行方案（预算不足以再跑一轮）"})
            break

        round_index = state["rounds_used"] + 1
        _emit("propose", {"round": round_index, **proposal})

        try:
            execution = executor.run_round(spec, proposal, state["best_checkpoint"], round_index)
        except Exception as error:  # 单轮失败不终止整体，留给 gatekeeper 回滚
            record = recorder.new_record(
                round_index,
                {"best_rmse_before": state.get("best_rmse"), "epochs_left": facts.get("epochs_left")},
                proposal, {"error": "%s: %s" % (type(error).__name__, error)}, None,
                {"action": "reject", "reason": "本轮执行异常，回滚到上一检查点"})
            recorder.append_decision(spec, record)
            state["rounds_used"] += 1
            state["epochs_used"] += int(proposal["epochs"])
            state["track_epochs_used"] = (state.get("track_epochs_used", 0)
                                          + int(proposal["epochs"]) * len(executor._ensemble() or [1]))
            state["best_rmse"] = state.get("best_rmse")
            state["accepted_rounds"] = state.get("accepted_rounds", [])
            recorder.save_state(spec, state)
            facts = observer.observe(spec, state)
            _emit("error", {"round": round_index, "detail": record["execution"]["error"]})
            continue

        metrics = evaluator.evaluate(spec, execution["checkpoint"], train_paths)
        decision = gatekeeper.decide(spec, state, metrics)
        if facts.get("_llm_error"):
            # 大模型未能给出可用提案时，把原因写进记录，便于审计"为什么这轮走策略表"
            proposal = dict(proposal, llm_bypass=facts["_llm_error"])
        observation = {"best_rmse_before": state.get("best_rmse"),
                       "best_tail_before": state.get("best_tail_rmse"),
                       "epochs_left": facts.get("epochs_left")}

        state["rounds_used"] += 1
        state["epochs_used"] += int(proposal["epochs"])
        # 账面 epoch 是「每条轨迹」的；集成会把它乘以轨迹数，这里单独记真实工作量
        tracks = int(execution.get("ensemble_tracks") or 1)
        state["track_epochs_used"] = (state.get("track_epochs_used", 0)
                                      + int(proposal["epochs"]) * tracks)

        if decision["action"] in ("accept", "advance_unverified"):
            if metrics is not None:
                state["best_rmse"] = metrics["rmse"]
                state["best_tail_rmse"] = metrics.get("tail_rmse")
                state["best_groups"] = {name: info["rmse"]
                                        for name, info in (metrics.get("groups") or {}).items()}
                state["accepted_rounds"] = state.get("accepted_rounds", []) + [round_index]
            else:
                state["unverified_rounds"] = state.get("unverified_rounds", []) + [round_index]
            state["best_checkpoint"] = execution["checkpoint"]
            # 写回平台默认路径：平台无论从 ckpt、best 副本还是 summary 取模型，都是同一份权重
            if publish_enabled:
                outcome = publish_module.publish(execution["checkpoint"], spec.publish_targets)
                execution["publish"] = outcome
                state["published_checkpoint"] = outcome["published"]
                if outcome["errors"]:
                    state["publish_errors"] = state.get("publish_errors", []) + outcome["errors"]
                    _emit("publish_error", {"round": round_index, "errors": outcome["errors"]})
            else:
                execution["publish"] = {"skipped": "起点备份失败，为免覆盖官方起点，本轮不发布"}
        else:
            execution["publish"] = {"skipped": "本轮未上线，不发布"}

        record = recorder.new_record(round_index, observation, proposal, execution, metrics, decision)
        recorder.append_decision(spec, record)
        _emit("result", {"round": round_index,
                         "rmse": None if metrics is None else round(metrics["rmse"], 2),
                         "tail_rmse": None if metrics is None else metrics.get("tail_rmse"),
                         "worst_group": None if metrics is None else metrics["worst_group"]["name"],
                         "decision": decision["action"]})
        recorder.save_state(spec, state)
        policy = planner.remember(proposal)
        facts = observer.observe(spec, state)

    published = state.get("published_checkpoint") or {}
    submitted_sha = published.get(str(spec.ckpt))
    if submitted_sha is None and Path(spec.ckpt).is_file():
        submitted_sha = publish_module.sha256_file(spec.ckpt)
    summary = {
        "rounds": state["rounds_used"],
        "accepted_rounds": state.get("accepted_rounds", []),
        "unverified_rounds": state.get("unverified_rounds", []),
        "epochs_used": state["epochs_used"],
        "track_epochs_used": state.get("track_epochs_used", 0),
        "best_rmse": state.get("best_rmse"),
        "best_tail_rmse": state.get("best_tail_rmse"),
        "best_checkpoint": state["best_checkpoint"],
        # 平台到底读哪个路径没有公开契约，所以这三个字段一起写清楚
        "submitted_checkpoint": str(spec.ckpt),
        "submitted_checkpoint_sha256": submitted_sha,
        "start_checkpoint": str(spec.start_ckpt),
        "start_checkpoint_sha256": start_backup.get("sha256"),
        "input_mode": spec.input_mode,
        "target_power": spec.target_power,
        "ensemble": executor._ensemble() or [],
        "publish_errors": state.get("publish_errors", []),
        "decisions": str(recorder.decisions_path(spec)),
        "elapsed_s": round(time.monotonic() - started, 2),
    }
    (Path(spec.run_dir) / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    _emit("summary", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
