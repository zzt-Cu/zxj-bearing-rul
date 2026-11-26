"""运行契约：路径、预算、门槛全部从环境变量或命令行来，不写死任何绝对路径。

单镜像跑 A/B/C 三题，题目由**数据本身**判定（见 harness/profiles.py）。
每个 profile 自带一份默认预算；环境变量仍然优先，便于调试与平台覆盖。
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import os

from harness import profiles

APP = Path(__file__).resolve().parent.parent


def _env_path(name, default):
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else Path(default)


def _env_int(name, default):
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_float(name, default):
    raw = os.environ.get(name)
    return float(raw) if raw else default


# 平台把数据挂到哪没有公开契约。按常见路径依次探测，命中即用；
# 探测结果会写进 spec 事件，所以即便全部没命中，日志本身也能说明平台挂了什么。
DATA_CANDIDATES = (
    "/data",
    "/data/public",
    "/input",
    "/input/public",
    "/mnt/data",
    "/workspace/data",
    "/app/data",
)


def probe_data_dirs():
    """返回每个候选路径上 train/validation 的存在情况，用于写进日志。"""
    probes = []
    for raw in DATA_CANDIDATES:
        path = Path(raw)
        probes.append({"path": raw,
                       "dir": path.is_dir(),
                       "train": (path / "train.jsonl").is_file(),
                       "validation": (path / "validation.jsonl").is_file()})
    bundled = APP / "public"
    probes.append({"path": str(bundled),
                   "dir": bundled.is_dir(),
                   "train": (bundled / "train.jsonl").is_file(),
                   "validation": (bundled / "validation.jsonl").is_file(),
                   "bundled": True})
    return probes


def _default_data_dir(probes=None):
    """优先显式挂载；否则按候选顺序探测；最后回退到打包时的 public/。

    返回 (路径, 来源)，来源 ∈ {env, probe, bundled, fallback}，
    进日志后可以直接回答「这次到底读了谁的数据」。
    """
    probes = probes if probes is not None else probe_data_dirs()
    for item in probes:
        if item["train"]:
            return Path(item["path"]), ("bundled" if item.get("bundled") else "probe")
    return APP / "public", "fallback"


def _start_name(profile):
    """起点备份的文件名。B 保持 checkpoint.start.npz（0.8 已验证），A/C 分开存。"""
    return "checkpoint.start.npz" if profile == "B" else "%s.start.npz" % profile


@dataclass
class Spec:
    data_dir: Path
    ckpt: Path
    run_dir: Path
    start_ckpt: Path = None
    profile: str = "B"
    profile_source: str = "default"
    task: str = "regression"
    primary_metric: str = "rmse"
    higher_is_better: bool = False
    classes: list = field(default_factory=list)
    max_rounds: int = 6
    epoch_budget: int = 200
    min_gain: float = 0.005
    tail_minutes: float = 100.0
    tail_trade_gain: float = 0.20
    tail_trade_rmse: float = 0.06
    min_tail_n: int = 30
    max_group_degrade: float = 0.25
    group_check: bool = True
    per_class_floor: float = None
    wall_limit_s: float = 1500.0
    input_mode: str = "abs8"
    target_power: float = 0.5
    ensemble: list = field(default_factory=list)
    stage_prefix: str = "zxj"
    data_source: str = ""
    data_probes: list = field(default_factory=list)

    @property
    def train_jsonl(self):
        return self.data_dir / "train.jsonl"

    @property
    def validation_jsonl(self):
        return self.data_dir / "validation.jsonl"

    @property
    def has_validation(self):
        return self.validation_jsonl.is_file()

    @property
    def publish_targets(self):
        """采纳后要写回的路径：平台默认检查点 + 一个显式的 best 副本。

        平台从哪个路径取模型没有公开契约，所以两个都写；写不进去只记账不中断。
        A/C 的官方起点在 baselines_ckpt/ 下，但**发布仍写回 /app/checkpoint.npz**：
        平台按这个路径取权重，写回子目录等于没交。三题各跑各的容器实例，互不覆盖。
        """
        if self.profile == "B":
            return (Path(self.ckpt), Path(self.ckpt).with_name("best_checkpoint.npz"))
        # PUBLISH_CKPT 只在本地调试时用：三题共用一个 /app/checkpoint.npz，
        # 本地连跑三题会互相覆盖 B 的官方起点，所以测试时指到临时目录。
        target = Path(os.environ.get("PUBLISH_CKPT") or (APP / "checkpoint.npz"))
        return (target, target.with_name("best_checkpoint.npz"))

    def resolve_start(self):
        """续训允许使用的「官方起点」：备份存在就用备份，否则用 ckpt 本身。

        publish 会把 ckpt 覆盖成权重平均点，所以重启来源必须指向备份，
        否则第 2 轮就从平均点起训（P3：平均点不可续训）。
        """
        backup = Path(self.start_ckpt) if self.start_ckpt else None
        if backup is not None and backup.is_file():
            return backup
        return Path(self.ckpt)

    def as_dict(self):
        data = asdict(self)
        for key in ("data_dir", "ckpt", "run_dir", "start_ckpt"):
            data[key] = str(data[key])
        return data

    @classmethod
    def build(cls, **overrides):
        probes = probe_data_dirs()
        if os.environ.get("DATA_DIR"):
            data_dir, data_source = _env_path("DATA_DIR", APP / "public"), "env"
        else:
            data_dir, data_source = _default_data_dir(probes)
        if overrides.get("data_dir") is not None:
            data_dir, data_source = Path(overrides["data_dir"]), "override"

        # 判题顺序：显式指定 > 入口已定的结论 > 读数据首行 > 兜底 B（保持 0.8 行为）。
        # 平台给哪一题就挂哪一题的数据，所以「读数据」比「读环境变量」可靠。
        forced = (os.environ.get("ZXJ_PROFILE") or "").strip().upper()
        resolved = (os.environ.get("ZXJ_PROFILE_RESOLVED") or "").strip().upper()
        if forced in profiles.PROFILES:
            profile, profile_source = forced, "env"
        elif resolved in profiles.PROFILES:
            profile, profile_source = resolved, "probe"
        else:
            detected = profiles.detect_from_file(Path(data_dir) / "train.jsonl")
            profile, profile_source = (detected, "probe") if detected else ("B", "default")
        prof = profiles.get(profile)

        ckpt = (Path(os.environ["CKPT"]).expanduser() if os.environ.get("CKPT")
                else profiles.default_ckpt(profile, APP))
        spec = cls(
            data_dir=data_dir,
            ckpt=ckpt,
            start_ckpt=_env_path("START_CKPT", ckpt.with_name(_start_name(profile))),
            run_dir=_env_path("RUN_DIR", APP / "runs"),
            profile=profile,
            profile_source=profile_source,
            task=prof["task"],
            primary_metric=prof["primary"],
            higher_is_better=bool(prof["higher_is_better"]),
            classes=list(prof.get("classes") or []),
            max_rounds=_env_int("MAX_ROUNDS", prof["max_rounds"]),
            epoch_budget=_env_int("EPOCH_BUDGET", prof["epoch_budget"]),
            min_gain=_env_float("MIN_GAIN", prof["min_gain"]),
            tail_minutes=float(prof["tail_minutes"] if prof.get("tail_minutes") is not None
                               else _env_float("TAIL_MINUTES", 0.0)),
            tail_trade_gain=_env_float("TAIL_TRADE_GAIN", prof.get("tail_trade_gain", 0.20)),
            tail_trade_rmse=_env_float("TAIL_TRADE_RMSE", prof.get("tail_trade_rmse", 0.06)),
            min_tail_n=_env_int("MIN_TAIL_N", int(prof.get("min_tail_n") or 0)),
            max_group_degrade=_env_float("MAX_GROUP_DEGRADE", float(prof.get("max_group_degrade") or 0.25)),
            group_check=bool(prof.get("group_check")),
            per_class_floor=prof.get("per_class_floor"),
            wall_limit_s=_env_float("WALL_LIMIT_S", prof["wall_limit_s"]),
            input_mode=os.environ.get("ZXJ_INPUT_MODE", prof.get("input_mode") or ""),
            target_power=float(os.environ.get("ZXJ_TARGET_POWER", prof.get("target_power") or 0.0)),
            ensemble=list(prof.get("ensemble") or []),
            stage_prefix=os.environ.get("STAGE_PREFIX", "zxj"),
            data_source=data_source,
            data_probes=probes,
        )
        # 平台的 ENSEMBLE_BATCH 若显式给出，覆盖 profile 默认（调试用）
        raw_ensemble = os.environ.get("ENSEMBLE_BATCH", "").strip()
        if raw_ensemble:
            sizes = [int(part) for part in raw_ensemble.split(",") if part.strip()]
            spec.ensemble = sizes if len(sizes) >= 2 else []
        for key, value in overrides.items():
            if value is not None:
                setattr(spec, key, value)
        if overrides.get("data_dir") is not None:
            spec.data_source = "override"
        # --ckpt 覆盖之后，起点备份必须跟着落到同一个目录（显式 START_CKPT 除外），
        # 否则 --ckpt 指到外部时备份会落在镜像里、和别人重跑互相干扰。
        if not os.environ.get("START_CKPT"):
            spec.start_ckpt = Path(spec.ckpt).with_name(_start_name(spec.profile))
        return spec
