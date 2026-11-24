"""Recorder：决策日志（只追加）+ 可恢复状态（原子写）。

decisions.jsonl 一行一条完整决策链，满足平台「至少 3 次读历史—做实验—依据结果决策」。
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import tempfile


def _now():
    return datetime.now(timezone.utc).isoformat()


def decisions_path(spec):
    return Path(spec.run_dir) / "decisions.jsonl"


def state_path(spec):
    return Path(spec.run_dir) / "state.json"


def append_decision(spec, record):
    path = decisions_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, allow_nan=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_state(spec):
    path = state_path(spec)
    if not path.is_file():
        return {"rounds_used": 0, "epochs_used": 0, "best_rmse": None, "best_tail_rmse": None,
                "best_checkpoint": str(spec.ckpt), "accepted_rounds": []}
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def save_state(spec, state):
    """临时文件 → flush → fsync → os.replace（再 fsync 目录），掉电也不留半截状态。"""
    path = state_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent),
                                     prefix=".state-", delete=False) as stream:
        json.dump(state, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)
    if os.name == "posix":
        try:
            handle = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(handle)
        except OSError:
            pass
        finally:
            os.close(handle)


def read_decisions(spec):
    """读决策日志，容忍进程被杀留下的残尾行（只丢弃解析失败的最后一行）。"""
    path = decisions_path(spec)
    if not path.is_file():
        return []
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            if index == len(lines) - 1:
                break  # 最后一行写了一半，忽略；前面的记录仍然可用
            raise
    return rows


def new_record(round_index, observation, proposal, execution, metrics, decision):
    return {
        "round": round_index,
        "ts": _now(),
        "observation": observation,
        "proposal": proposal,
        "result": metrics,
        "execution": execution,
        "decision": decision,
    }
