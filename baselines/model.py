"""Small NumPy MLP + Adam. Checkpoints preserve every state needed to resume."""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import zipfile

_THREADS = int(os.environ.get("BASELINE_CPU_THREADS", "1"))
if not 1 <= _THREADS <= 8:
    raise ValueError("BASELINE_CPU_THREADS must be between 1 and 8")
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = str(_THREADS)
import numpy as np

PARAMETERS = ("w1", "b1", "w2", "b2")
FORMAT = "contest.cpu-mlp.v1"
MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024

# 逐样本输入整形模式。none 为参考行为；abs8 在标准化后把每个通道的 8 个
# 绝对量特征按通道内均值/标准差归一化，抑制跨轴承绝对幅度差异。
# abs8log 在 abs8 之上再追加 14 列对数绝对量（两通道各 7 列）：能量类特征跨数量级，
# 取对数后缓慢退化段（如长寿命轴承前 90% 寿命）的微小趋势才不会被线性尺度淹没。
# 两种变换都逐样本完成，不引入任何集合统计量，训练与推理共用同一实现。
INPUT_MODE = os.environ.get("ZXJ_INPUT_MODE", "abs8").strip().lower()
INPUT_MODES = ("none", "abs8", "abs8log")
if INPUT_MODE not in INPUT_MODES:
    raise ValueError("ZXJ_INPUT_MODE must be one of " + ", ".join(INPUT_MODES))

# 回归目标变换 y' = y**p。RUL 分布右偏，直接用 y 训练时高 RUL 段主导梯度，
# 末端（真实 RUL 小）被系统性高估。p<1 压缩高位、放大相对差异，使末端误差下降。
# p=1.0 为参考行为。该变换单调可逆，不改动特征、结构、种子与任务，predict 侧用同一实现还原。
TARGET_POWER = float(os.environ.get("ZXJ_TARGET_POWER", "0.5"))
if not 0.1 <= TARGET_POWER <= 1.0:
    raise ValueError("ZXJ_TARGET_POWER must be within [0.1, 1.0]")

_ABS_SUFFIXES = ("band1_energy_0_2k", "band2_energy_2k_6k", "band3_energy_6k_12k",
                 "mean_abs", "peak", "peak_to_peak", "rms", "spectral_rms")


def absolute_groups(paths):
    """按通道把绝对量特征的列下标分组；缺失则跳过，A/C 特征不受影响。"""
    groups = {}
    for index, path in enumerate(paths):
        for channel in ("horizontal", "vertical"):
            if '"%s"' % channel not in path:
                continue
            for suffix in _ABS_SUFFIXES:
                if path.endswith('["key","%s"]]' % suffix):
                    groups.setdefault(channel, []).append(index)
                    break
    return [sorted(columns) for columns in groups.values() if len(columns) >= 2]


# abs8log 追加列：两通道各取 7 个量纲为"幅值/能量"的特征取自然对数。
# 顺序固定为 horizontal 的 7 个、再 vertical 的 7 个，保证训练与推理列序一致。
_LOG_SUFFIXES = ("rms", "peak", "mean_abs", "spectral_rms",
                 "band1_energy_0_2k", "band2_energy_2k_6k", "band3_energy_6k_12k")
_LOG_EPS = 1e-12


def log_columns(paths):
    """返回被取对数的原始列下标，顺序即追加列顺序。"""
    located = {}
    for index, path in enumerate(paths):
        for channel in ("horizontal", "vertical"):
            if '"%s"' % channel not in path:
                continue
            for suffix in _LOG_SUFFIXES:
                if path.endswith('["key","%s"]]' % suffix):
                    located[(channel, suffix)] = index
    return [located[(channel, suffix)] for channel in ("horizontal", "vertical")
            for suffix in _LOG_SUFFIXES if (channel, suffix) in located]


def log_features(paths, x):
    return np.log(np.maximum(x[:, log_columns(paths)], _LOG_EPS))


class Model:
    def __init__(self, task, paths, classes, x, y, *, hidden=32, seed=42, learning_rate=.003, batch_size=128):
        if task not in ("classification", "regression") or not 1 <= hidden <= 128:
            raise ValueError("invalid model configuration")
        if x.ndim != 2 or x.shape[1] != len(paths):
            raise ValueError("feature schema mismatch")
        output = len(classes) if task == "classification" else 1
        if task == "classification" and (len(classes) < 2 or len(set(classes)) != len(classes)):
            raise ValueError("classification needs distinct classes")
        source = log_columns(paths) if INPUT_MODE == "abs8log" else []
        width = x.shape[1] + len(source)
        count = width * hidden + hidden + hidden * output + output
        if count >= 100000:
            raise ValueError("reference baseline must have fewer than 100,000 parameters")
        rng = np.random.default_rng(seed)
        self.params = {"w1": rng.normal(0, math.sqrt(1 / width), (width, hidden)),
                       "b1": np.zeros(hidden), "w2": rng.normal(0, math.sqrt(1 / hidden), (hidden, output)),
                       "b2": np.zeros(output)}
        self.first = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.second = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.mean = np.zeros(width)
        self.scale = np.ones(width)
        self.mean[:x.shape[1]] = x.mean(axis=0)
        scale = x.std(axis=0)
        self.scale[:x.shape[1]] = np.where(scale < 1e-12, 1.0, scale)
        if source:
            logs = log_features(paths, x)
            self.mean[x.shape[1]:] = logs.mean(axis=0)
            tail = logs.std(axis=0)
            self.scale[x.shape[1]:] = np.where(tail < 1e-12, 1.0, tail)
        shaped = y ** TARGET_POWER if task == "regression" else y
        self.meta = {"format": FORMAT, "device": "cpu", "task": task, "paths": paths,
                     "classes": list(classes), "hidden": hidden, "seed": seed, "epochs": 0, "steps": 0,
                     "learning_rate": learning_rate, "batch_size": batch_size, "parameter_count": count,
                     "target_mean": float(shaped.mean()) if task == "regression" else 0.0,
                     "target_scale": max(float(shaped.std()), 1.0) if task == "regression" else 1.0}
        self.meta["input_mode"] = INPUT_MODE
        self.meta["input_width"] = width
        if source:
            self.meta["derived"] = "logabs14"
        self.meta["target_power"] = TARGET_POWER if task == "regression" else 1.0
        self.validate()

    def validate(self):
        m = self.meta
        if m.get("format") != FORMAT or m.get("device") != "cpu" or m.get("task") not in ("classification", "regression"):
            raise ValueError("unsupported checkpoint")
        for name in ("seed", "epochs", "steps", "hidden", "batch_size", "parameter_count"):
            if type(m.get(name)) is not int or m[name] < 0:
                raise ValueError("invalid checkpoint integer")
        if not 1 <= m["hidden"] <= 128 or not 1 <= m["batch_size"] <= 65536 or not 0 < m["parameter_count"] < 100000:
            raise ValueError("invalid checkpoint architecture")
        if not 0 <= m["seed"] < 2**32:
            raise ValueError("seed must fit an unsigned 32-bit integer")
        if not isinstance(m.get("paths"), list) or not 1 <= len(m["paths"]) <= 4096 or any(not isinstance(p, str) for p in m["paths"]) or len(set(m["paths"])) != len(m["paths"]):
            raise ValueError("invalid checkpoint feature schema")
        classes = m.get("classes")
        if not isinstance(classes, list) or any(not isinstance(c, str) or not c for c in classes) or len(set(classes)) != len(classes):
            raise ValueError("invalid checkpoint classes")
        if (m["task"] == "classification" and len(classes) < 2) or (m["task"] == "regression" and classes):
            raise ValueError("invalid checkpoint classes")
        for name in ("learning_rate", "target_mean", "target_scale"):
            if isinstance(m.get(name), bool) or not isinstance(m.get(name), (float, int)) or not math.isfinite(m[name]):
                raise ValueError("invalid checkpoint numeric state")
        if not 0 < m["learning_rate"] <= 1 or m["target_scale"] <= 0:
            raise ValueError("invalid optimizer/scaler state")
        expected_derived = len(log_columns(m["paths"])) if m.get("derived") == "logabs14" else 0
        width = m.get("input_width", len(m["paths"]))
        if type(width) is not int or width != len(m["paths"]) + expected_derived:
            raise ValueError("invalid checkpoint input width")
        hidden = m["hidden"]
        output = len(classes) if m["task"] == "classification" else 1
        shapes = {"w1": (width, hidden), "b1": (hidden,), "w2": (hidden, output), "b2": (output,)}
        for group in (self.params, self.first, self.second):
            if set(group) != set(PARAMETERS):
                raise ValueError("checkpoint tensors missing")
            for name, shape in shapes.items():
                if group[name].shape != shape or group[name].dtype != np.dtype("float64") or not np.all(np.isfinite(group[name])):
                    raise ValueError("checkpoint tensor shape or values invalid")
        if sum(value.size for value in self.params.values()) != m["parameter_count"]:
            raise ValueError("checkpoint parameter count mismatch")
        if self.mean.dtype != np.dtype("float64") or self.scale.dtype != np.dtype("float64") or self.mean.shape != (width,) or self.scale.shape != (width,) or not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise ValueError("checkpoint normalization invalid")
        if any(np.any(value < 0) for value in self.second.values()):
            raise ValueError("invalid Adam second moment")
        if self.meta.get("input_mode") not in INPUT_MODES:
            raise ValueError("invalid input shaping mode")
        power = self.meta.get("target_power", 1.0)
        if isinstance(power, bool) or not isinstance(power, (float, int)) or not 0.1 <= power <= 1.0:
            raise ValueError("invalid target shaping power")

    def normalized(self, x):
        if x.ndim != 2 or x.shape[1] != len(self.meta["paths"]) or not np.all(np.isfinite(x)):
            raise ValueError("invalid feature matrix")
        width = x.shape[1]
        z = np.clip((x - self.mean[:width]) / self.scale[:width], -10, 10)
        if self.meta.get("input_mode") in ("abs8", "abs8log"):
            for columns in absolute_groups(self.meta["paths"]):
                block = z[:, columns]
                z[:, columns] = (block - block.mean(axis=1, keepdims=True)) / (
                    block.std(axis=1, keepdims=True) + 1e-6)
        if self.meta.get("derived") == "logabs14":
            logs = (log_features(self.meta["paths"], x) - self.mean[width:]) / self.scale[width:]
            z = np.hstack([z, np.clip(logs, -10, 10)])
        return z

    def expand_inputs(self, x):
        """把参考起点的输入宽度扩到当前模式所需宽度。

        新增 log 列用与常规初始化同分布的小随机数起步（不置零）：置零会让这 14 列
        在 tanh 饱和区长期拿不到有效梯度，实测比随机起步差一个量级。
        随机流按构造顺序对齐（先 w1、再 w2、再扩展），保证与从头构造同一模式的
        初始化一致，从而训练可复现。归一化统计量取本次训练集，不用参考起点的集合量。
        """
        source = log_columns(self.meta["paths"])
        old = self.params["w1"].shape[0]
        width = x.shape[1] + len(source)
        if not source or old != x.shape[1] or width == old:
            return self
        hidden = self.meta["hidden"]
        rng = np.random.default_rng(self.meta["seed"])
        rng.normal(0, math.sqrt(1 / width), (width, hidden))      # 对应构造时的 w1
        rng.normal(0, math.sqrt(1 / hidden), (hidden, 1))         # 对应构造时的 w2
        grown = rng.normal(0, math.sqrt(1 / width), (width, hidden))
        grown[:old] = self.params["w1"]
        first = np.zeros((width, hidden))
        first[:old] = self.first["w1"]
        second = np.zeros((width, hidden))
        second[:old] = self.second["w1"]
        self.params["w1"], self.first["w1"], self.second["w1"] = grown, first, second
        mean = np.zeros(width)
        scale = np.ones(width)
        mean[:old], scale[:old] = self.mean, self.scale
        logs = log_features(self.meta["paths"], x)
        mean[old:] = logs.mean(axis=0)
        tail = logs.std(axis=0)
        scale[old:] = np.where(tail < 1e-12, 1.0, tail)
        self.mean, self.scale = mean, scale
        self.meta["input_mode"] = INPUT_MODE
        self.meta["input_width"] = width
        self.meta["derived"] = "logabs14"
        self.meta["parameter_count"] = int(sum(v.size for v in self.params.values()))
        self.validate()
        return self

    def forward(self, x):
        hidden = np.tanh(x @ self.params["w1"] + self.params["b1"])
        return hidden, hidden @ self.params["w2"] + self.params["b2"]

    def fit(self, x, y, epochs):
        if type(epochs) is not int or not 1 <= epochs <= 10000:
            raise ValueError("epochs must be between 1 and 10000")
        x = self.normalized(x)
        if y.shape != (len(x),) or not len(x):
            raise ValueError("target shape mismatch")
        if self.meta["task"] == "regression":
            power = TARGET_POWER
            if abs(float(self.meta.get("target_power", 1.0)) - power) > 1e-9:
                # 起点（参考权重或上一阶段）的目标空间与本次不同：按本次数据一次性重锚定，
                # 之后 target_power 与实际统计量一致，predict 侧据此还原，不会静默错位。
                shaped = y ** power
                self.meta["target_mean"] = float(shaped.mean())
                self.meta["target_scale"] = max(float(shaped.std()), 1.0)
                self.meta["target_power"] = power
            target = (y ** power - self.meta["target_mean"]) / self.meta["target_scale"]
        else:
            if np.any(y < 0) or np.any(y >= len(self.meta["classes"])):
                raise ValueError("unknown class")
            target = y
        epoch_losses = []
        for _ in range(epochs):
            # Epoch-indexed shuffling makes split/resumed execution reproduce a continuous run.
            rng = np.random.default_rng(np.random.SeedSequence([self.meta["seed"], self.meta["epochs"]]))
            order = rng.permutation(len(x))
            loss_sum = 0.0
            for start in range(0, len(order), self.meta["batch_size"]):
                indices = order[start:start + self.meta["batch_size"]]
                batch, expected = x[indices], target[indices]
                hidden, logits = self.forward(batch)
                if self.meta["task"] == "classification":
                    logits -= logits.max(axis=1, keepdims=True)
                    probabilities = np.exp(logits)
                    probabilities /= probabilities.sum(axis=1, keepdims=True)
                    loss_sum -= float(np.log(np.maximum(probabilities[np.arange(len(indices)), expected], 1e-300)).sum())
                    delta = probabilities
                    delta[np.arange(len(indices)), expected] -= 1
                    delta /= len(indices)
                else:
                    error = logits[:, 0] - expected
                    loss_sum += float(np.sum(error ** 2))
                    delta = (2 * error / len(indices))[:, None]
                gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
                hidden_delta = (delta @ self.params["w2"].T) * (1 - hidden ** 2)
                gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))
                norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
                if not math.isfinite(norm):
                    raise ValueError("nonfinite gradient")
                self.meta["steps"] += 1
                for name, gradient in gradients.items():
                    gradient *= min(1.0, 5.0 / max(norm, 1e-12))
                    self.first[name] = .9 * self.first[name] + .1 * gradient
                    self.second[name] = .999 * self.second[name] + .001 * gradient ** 2
                    first = self.first[name] / (1 - .9 ** self.meta["steps"])
                    second = self.second[name] / (1 - .999 ** self.meta["steps"])
                    self.params[name] -= self.meta["learning_rate"] * first / (np.sqrt(second) + 1e-8)
                if not all(np.all(np.isfinite(p)) for p in self.params.values()):
                    raise ValueError("nonfinite model state")
            self.meta["epochs"] += 1
            epoch_losses.append(loss_sum / len(x))
        self.validate()
        return epoch_losses

    def predict(self, x):
        _, logits = self.forward(self.normalized(x))
        if self.meta["task"] == "classification":
            logits -= logits.max(axis=1, keepdims=True)
            result = np.exp(logits)
            return result / result.sum(axis=1, keepdims=True)
        return np.maximum(0, logits[:, 0] * self.meta["target_scale"] + self.meta["target_mean"]) ** (
            1.0 / float(self.meta.get("target_power", 1.0)))

    def save(self, path):
        self.validate()
        arrays = {**self.params, **{"adam_first_" + k: v for k, v in self.first.items()},
                  **{"adam_second_" + k: v for k, v in self.second.items()}, "mean": self.mean, "scale": self.scale,
                  "metadata": np.array(json.dumps(self.meta, sort_keys=True, separators=(",", ":"), allow_nan=False))}
        # Fixed ZIP metadata avoids changing the identity solely because of wall-clock time.
        with Path(path).open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
                info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint must be a regular file no larger than 16 MiB")
        expected = {*PARAMETERS, *("adam_first_" + k for k in PARAMETERS), *("adam_second_" + k for k in PARAMETERS), "mean", "scale", "metadata"}
        with zipfile.ZipFile(path) as archive:
            names = [entry.filename for entry in archive.infolist()]
            if len(names) != len(expected) or set(names) != {name + ".npy" for name in expected} or sum(e.file_size for e in archive.infolist()) > MAX_CHECKPOINT_BYTES:
                raise ValueError("invalid checkpoint archive")
        with np.load(path, allow_pickle=False) as arrays:
            model = cls.__new__(cls)
            metadata = arrays["metadata"]
            if metadata.ndim != 0 or metadata.dtype.kind != "U":
                raise ValueError("invalid checkpoint metadata")
            model.meta = json.loads(str(metadata))
            if "input_mode" not in model.meta:
                # 参考起点不含该字段，语义是原始 34 列。若当前模式要追加派生列，
                # 先按参考语义加载（宽度不变、校验通过），再由 train_run 拿到训练集后
                # 调用 expand_inputs 扩展，避免在不知道数据统计量的情况下臆造归一化参数。
                model.meta["input_mode"] = "abs8" if INPUT_MODE == "abs8log" else INPUT_MODE
            elif model.meta["input_mode"] != INPUT_MODE:
                raise ValueError("checkpoint input shaping mode %r does not match %r; "
                                 "predictions would silently differ" % (model.meta["input_mode"], INPUT_MODE))
            # 参考起点不含该字段：其统计量属于原始目标空间，fit 首轮会按当前幂次重锚定。
            model.meta.setdefault("target_power", 1.0)
            model.params = {k: arrays[k].copy() for k in PARAMETERS}
            model.first = {k: arrays["adam_first_" + k].copy() for k in PARAMETERS}
            model.second = {k: arrays["adam_second_" + k].copy() for k in PARAMETERS}
            model.mean, model.scale = arrays["mean"].copy(), arrays["scale"].copy()
        model.validate()
        return model


def checkpoint_digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
