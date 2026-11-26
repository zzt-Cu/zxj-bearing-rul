"""镜像入口：先判题、把 ENV 定下来，再启动 harness.loop。

为什么需要这一层：`baselines/model.py` 在**模块导入时**把 `ZXJ_INPUT_MODE` 与
`ZXJ_TARGET_POWER` 读成模块级常量，之后再用 os.environ 改是无效的（Python 已经
执行过那行）。而三题的输入整形模式不同——A/C 的官方检查点是 abs8，B 是 abs8log——
所以必须在 import baselines 之前把 ENV 定好。

判据与 spec.build 完全同源（两边都调 profiles.detect_from_file），不会出现
「入口认为是 A、spec 认为是 B」的错位；万一错位，model.py 会直接抛异常而不是静默算错。
"""
import os
import sys
from pathlib import Path

from harness import profiles, spec as spec_module


def _argv_value(flag, argv):
    if flag in argv:
        index = argv.index(flag)
        if index + 1 < len(argv):
            return argv[index + 1]
    return None


def resolve(argv):
    """返回 (profile, 来源, 数据目录)。来源只用于写日志，不影响行为。"""
    override = _argv_value("--data", argv)
    if override:
        data_dir = Path(override)
    elif os.environ.get("DATA_DIR"):
        data_dir = Path(os.environ["DATA_DIR"])
    else:
        data_dir, _ = spec_module._default_data_dir(spec_module.probe_data_dirs())

    forced = (os.environ.get("ZXJ_PROFILE") or "").strip().upper()
    if forced in profiles.PROFILES:
        return forced, "env", data_dir

    detected = profiles.detect_from_file(Path(data_dir) / "train.jsonl")
    if detected:
        return detected, "probe", data_dir
    return "B", "default", data_dir


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    profile, source, data_dir = resolve(argv)
    prof = profiles.get(profile)

    # 用 setdefault：平台或调试时显式给的环境变量优先，这里只补缺省。
    os.environ.setdefault("ZXJ_PROFILE_RESOLVED", profile)
    os.environ.setdefault("ZXJ_INPUT_MODE", prof.get("input_mode") or "abs8")
    os.environ.setdefault("ZXJ_TARGET_POWER", str(prof.get("target_power") or 1.0))

    # 这一行必须在 ENV 定好之后：它会把 baselines 拉进来，模块级常量此时才读取。
    from harness.loop import main as loop_main

    print("[harness] route %s" % __import__("json").dumps(
        {"profile": profile, "source": source, "data_dir": str(data_dir),
         "input_mode": os.environ.get("ZXJ_INPUT_MODE"),
         "target_power": os.environ.get("ZXJ_TARGET_POWER")}, ensure_ascii=False), flush=True)
    return loop_main(argv)


if __name__ == "__main__":
    sys.exit(main())
