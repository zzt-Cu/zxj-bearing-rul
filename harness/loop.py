"""Harness 主循环：无人工干预，跑到预算用尽或轮次用尽为止。

用法（容器内外一致）：
    python -m harness.loop --max-rounds 3 --epoch-budget 100
    python -m harness.loop --dry-run          # 只观察与提案，不训练
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback

try:
    from harness import evaluator, executor, gatekeeper, observer, planner, recorder
    from harness import publish as publish_module
    from harness import spec as spec_module
except ImportError:  # 直接以脚本方式运行时的兜底
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from harness import evaluator, executor, gatekeeper, observer, planner, recorder
    from harness import publish as publish_module
    from harness import spec as spec_module

from baselines.model import Model
from harness import dataset_cache


def _emit(kind, payload):
    print("[harness] %s %s" % (kind, json.dumps(payload, ensure_ascii=False, allow_nan=False)), flush=True)


def _pick_writable_dir(preferred, fallbacks):
    """返回第一个能真正写入的目录；全都不行时返回 (None, 失败原因列表)。

    只 mkdir 不够：目录可能存在但不可写（非 root 遇到 root 拥有的 755 目录），
    所以真写一个探针文件再删掉。
    """
    errors = []
    for cand in (preferred,) + tuple(fallbacks):
        try:
            p = Path(cand)
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return p, errors
        except OSError as error:  # noqa: PERF203 - 逐个候选试，异常就是分支条件
            errors.append("%s: %s: %s" % (cand, type(error).__name__, error))
    return None, errors


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

    # 第一行就吐心跳：平台若按日志判存活，这一行能证明进程起来了、参数是什么；
    # 若后面卡住，日志停在哪个事件也能一眼定位。
    _emit("boot", {"argv": sys.argv[1:], "pid": os.getpid(), "cwd": os.getcwd(),
                   "cpu_threads": os.environ.get("BASELINE_CPU_THREADS", ""),
                   "use_llm": os.environ.get("ZXJ_USE_LLM", "0"),
                   "max_rounds": os.environ.get("MAX_ROUNDS", ""),
                   "wall_limit_s": os.environ.get("WALL_LIMIT_S", "")})

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

    # 运行目录必须可写。平台若以非 root 或只读根文件系统启动容器，/app/runs 会
    # PermissionError / EROFS。0.6 及以前直接抛异常退出，traceback 只进 stderr，
    # 平台只收 stdout 时就成了「运行失败且无日志」。0.7 起按候选回退，并把结果打进日志。
    run_dir, run_dir_errors = _pick_writable_dir(spec.run_dir,
                                                 ("/tmp/zxj_runs", "/dev/shm/zxj_runs",
                                                  str(Path.home() / "zxj_runs")))
    if run_dir is None:
        _emit("fatal", {"error": "no writable run dir", "preferred": str(spec.run_dir),
                        "tried": run_dir_errors})
        return 2
    if Path(run_dir) != Path(spec.run_dir):
        _emit("warn", {"run_dir_fallback": str(run_dir), "preferred": str(spec.run_dir),
                       "tried": run_dir_errors})
        spec.run_dir = Path(run_dir)
    _emit("writable", {
        "run_dir": str(spec.run_dir),
        "ckpt": str(spec.ckpt),
        "ckpt_writable": os.access(str(spec.ckpt), os.W_OK),
        "ckpt_dir_writable": os.access(str(Path(spec.ckpt).parent), os.W_OK),
        "data_dir_readable": os.access(str(spec.data_dir), os.R_OK),
        "uid": os.getuid() if hasattr(os, "getuid") else None,
    })

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
    state.setdefault("best_score", state.get("best_rmse"))
    state.setdefault("best_classes", {})
    state.setdefault("best_nasa_score", None)
    if not Path(state["best_checkpoint"]).is_file():
        state["best_checkpoint"] = str(spec.ckpt)

    train_paths = dataset_cache.read(spec.train_jsonl, spec.task, spec.classes)["paths"]
    facts = observer.observe(spec, state)
    _emit("spec", spec.as_dict())
    _emit("observe", facts)
    _emit("input_mode", {"checkpoint": Model.load(spec.ckpt).meta.get("input_mode"),
                         "process_env": spec.input_mode})

    if args.dry_run:
        policy = planner._load_policy(spec.profile)
        _emit("propose", planner.propose(facts, policy) or {"proposal": None, "reason": "预算不足"})
        return 0

    started = time.monotonic()
    policy = planner._load_policy(spec.profile)
    _emit("budget", {"wall_limit_s": spec.wall_limit_s, "max_rounds": spec.max_rounds,
                     "epoch_budget": spec.epoch_budget,
                     "note": "墙钟预算用于在平台强杀容器前自行收尾：已发布的最好权重不受影响"})
    last_round_cost = 0.0
    stop_reason = {"reason": "未记录（异常退出）"}
    while True:
        elapsed = time.monotonic() - started
        if state["rounds_used"] >= spec.max_rounds:
            stop_reason = {"reason": "达到轮次上限", "max_rounds": spec.max_rounds,
                           "elapsed_s": round(elapsed, 1)}
            _emit("stop", stop_reason)
            break
        if state["epochs_used"] >= spec.epoch_budget:
            stop_reason = {"reason": "epoch 预算用尽", "epoch_budget": spec.epoch_budget,
                           "elapsed_s": round(elapsed, 1)}
            _emit("stop", stop_reason)
            break
        if elapsed > spec.wall_limit_s:
            stop_reason = {"reason": "墙钟上限", "wall_limit_s": spec.wall_limit_s,
                           "elapsed_s": round(elapsed, 1)}
            _emit("stop", stop_reason)
            break
        # 0.8：平台会在固定窗口后强杀容器，杀掉时拿不到 summary、也没法回滚。
        # 所以每轮开始前用「上一轮实测耗时 × 1.2」估下一轮，装不进预算就自己收尾。
        # 收尾不影响成绩：第 1 轮 accept 后已经 publish 过，最好的权重早写回平台路径。
        if last_round_cost:
            estimate = last_round_cost * 1.2
            if elapsed + estimate > spec.wall_limit_s:
                stop_reason = {"reason": "剩余预算装不下下一轮", "wall_limit_s": spec.wall_limit_s,
                               "elapsed_s": round(elapsed, 1),
                               "next_round_est_s": round(estimate, 1)}
                _emit("stop", stop_reason)
                break

        proposal = planner.propose(facts, policy)
        if not proposal:
            _emit("stop", {"reason": "无可行方案（预算不足以再跑一轮）"})
            break

        round_index = state["rounds_used"] + 1
        _emit("propose", {"round": round_index, **proposal})
        round_t0 = time.monotonic()

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
                                          + int(proposal["epochs"]) * len(spec.ensemble or [1]))
            state["best_rmse"] = state.get("best_rmse")
            state["accepted_rounds"] = state.get("accepted_rounds", [])
            recorder.save_state(spec, state)
            facts = observer.observe(spec, state)
            _emit("error", {"round": round_index, "detail": record["execution"]["error"]})
            last_round_cost = time.monotonic() - round_t0
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
                if spec.profile == "B":
                    state["best_rmse"] = metrics["rmse"]
                    state["best_tail_rmse"] = metrics.get("tail_rmse")
                    state["best_groups"] = {name: info["rmse"]
                                            for name, info in (metrics.get("groups") or {}).items()}
                    state["best_score"] = metrics["rmse"]
                elif spec.profile == "A":
                    state["best_score"] = metrics["macro_f1"]
                    state["best_classes"] = {name: info["f1"]
                                             for name, info in (metrics.get("per_class") or {}).items()}
                else:
                    state["best_score"] = metrics["rmse"]
                    state["best_nasa_score"] = metrics.get("nasa_score")
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
                         "metric": spec.primary_metric,
                         "primary": None if metrics is None else round(metrics["primary"], 4),
                         "rmse": None if metrics is None or metrics.get("rmse") is None
                                 else round(metrics["rmse"], 2),
                         "tail_rmse": None if metrics is None else metrics.get("tail_rmse"),
                         "worst_group": None if metrics is None
                                        else (metrics.get("worst_group") or {}).get("name"),
                         "decision": decision["action"],
                         "elapsed_s": round(time.monotonic() - started, 1)})
        recorder.save_state(spec, state)
        policy = planner.remember(proposal, spec.profile)
        facts = observer.observe(spec, state)
        last_round_cost = time.monotonic() - round_t0
        # 每轮落一份 progress：万一平台在收尾前强杀，也能看出跑到第几轮、当时最好是多少
        try:
            (Path(spec.run_dir) / "progress.json").write_text(json.dumps({
                "status": "running", "rounds_done": state["rounds_used"],
                "best_rmse": state.get("best_rmse"),
                "published_sha256": (state.get("published_checkpoint") or {}).get(str(spec.ckpt)),
                "elapsed_s": round(time.monotonic() - started, 1)}, ensure_ascii=False) + "\n",
                encoding="utf-8")
        except OSError:
            pass

    published = state.get("published_checkpoint") or {}
    # 提交路径取 publish_targets 的第一个：B 就是 ckpt 本身，A/C 是 /app/checkpoint.npz。
    # 0.9 首版这里写的是 spec.ckpt，A/C 会报出「官方基线」的指纹，看起来像没训练过。
    submitted_path = Path(spec.publish_targets[0])
    submitted_sha = published.get(str(submitted_path))
    if submitted_sha is None and submitted_path.is_file():
        submitted_sha = publish_module.sha256_file(submitted_path)
    summary = {
        "profile": spec.profile,
        "profile_source": spec.profile_source,
        "task": spec.task,
        "primary_metric": spec.primary_metric,
        "rounds": state["rounds_used"],
        "planned_rounds": spec.max_rounds,
        "stop": stop_reason,
        "accepted_rounds": state.get("accepted_rounds", []),
        "unverified_rounds": state.get("unverified_rounds", []),
        "epochs_used": state["epochs_used"],
        "track_epochs_used": state.get("track_epochs_used", 0),
        # B 用 best_rmse（原名不动）；A/C 用 best_score 承载主指标
        "best_rmse": state.get("best_rmse"),
        "best_score": state.get("best_score"),
        "best_tail_rmse": state.get("best_tail_rmse"),
        "best_nasa_score": state.get("best_nasa_score"),
        "best_checkpoint": state["best_checkpoint"],
        # 平台到底读哪个路径没有公开契约，所以这三个字段一起写清楚
        "submitted_checkpoint": str(submitted_path),
        "submitted_checkpoint_sha256": submitted_sha,
        "start_checkpoint": str(spec.start_ckpt),
        "start_checkpoint_sha256": start_backup.get("sha256"),
        "input_mode": spec.input_mode,
        "target_power": spec.target_power,
        "ensemble": list(spec.ensemble or []),
        "publish_errors": state.get("publish_errors", []),
        "decisions": str(recorder.decisions_path(spec)),
        "elapsed_s": round(time.monotonic() - started, 2),
    }
    (Path(spec.run_dir) / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    _emit("summary", summary)
    return 0


if __name__ == "__main__":
    # 平台多数只回收 stdout。0.6 及以前，未捕获异常只进 stderr，平台侧就是
    # 「运行失败且无日志」，连查都没法查。这里把整段 traceback 也打到 stdout。
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - 最后一道兜底，任何异常都要留下证据
        tb = traceback.format_exc()
        _emit("fatal", {"error": "uncaught", "type": tb.strip().splitlines()[-1][:200],
                        "traceback": tb.strip().splitlines()[-14:]})
        print(tb, file=sys.stdout, flush=True)
        sys.exit(1)
