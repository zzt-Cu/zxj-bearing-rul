"""运行契约：路径、预算、门槛全部从环境变量或命令行来，不写死任何绝对路径。"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import os

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


@dataclass
class Spec:
    data_dir: Path
    ckpt: Path
    run_dir: Path
    start_ckpt: Path = None
    max_rounds: int = 6
    epoch_budget: int = 200
    min_gain: float = 0.005
    tail_minutes: float = 100.0
    tail_trade_gain: float = 0.20
    tail_trade_rmse: float = 0.06
    min_tail_n: int = 30
    max_group_degrade: float = 0.25
    wall_limit_s: float = 1500.0
    input_mode: str = "abs8"
    target_power: float = 0.5
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
        """
        return (Path(self.ckpt), Path(self.ckpt).with_name("best_checkpoint.npz"))

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
        ckpt = _env_path("CKPT", APP / "checkpoint.npz")
        probes = probe_data_dirs()
        if os.environ.get("DATA_DIR"):
            data_dir, data_source = _env_path("DATA_DIR", APP / "public"), "env"
        else:
            data_dir, data_source = _default_data_dir(probes)
        spec = cls(
            data_dir=data_dir,
            ckpt=ckpt,
            # 起点备份默认与 ckpt 同目录：--ckpt 指到别处时不能把备份落在镜像里
            start_ckpt=_env_path("START_CKPT", ckpt.with_name("checkpoint.start.npz")),
            run_dir=_env_path("RUN_DIR", APP / "runs"),
            max_rounds=_env_int("MAX_ROUNDS", 6),
            epoch_budget=_env_int("EPOCH_BUDGET", 200),
            min_gain=_env_float("MIN_GAIN", 0.005),
            tail_minutes=_env_float("TAIL_MINUTES", 100.0),
            tail_trade_gain=_env_float("TAIL_TRADE_GAIN", 0.20),
            tail_trade_rmse=_env_float("TAIL_TRADE_RMSE", 0.06),
            min_tail_n=_env_int("MIN_TAIL_N", 30),
            max_group_degrade=_env_float("MAX_GROUP_DEGRADE", 0.25),
            wall_limit_s=_env_float("WALL_LIMIT_S", 1500.0),
            input_mode=os.environ.get("ZXJ_INPUT_MODE", "abs8"),
            target_power=float(os.environ.get("ZXJ_TARGET_POWER", "0.5")),
            stage_prefix=os.environ.get("STAGE_PREFIX", "zxj"),
            data_source=data_source,
            data_probes=probes,
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(spec, key, value)
        if overrides.get("data_dir") is not None:
            spec.data_source = "override"
        # --ckpt 覆盖之后，起点备份必须跟着落到同一个目录（显式 START_CKPT 除外），
        # 否则 --ckpt 指到外部时备份会落在镜像里、和别人重跑互相干扰。
        if not os.environ.get("START_CKPT"):
            spec.start_ckpt = Path(spec.ckpt).with_name("checkpoint.start.npz")
        return spec
