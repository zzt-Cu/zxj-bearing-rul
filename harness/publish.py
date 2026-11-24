"""检查点发布：把采纳的权重原子写回「平台默认路径」，并保留原始起点。

这里的顺序不能调换，两件事必须成对出现：

1. **起点备份**：原始 `checkpoint.npz` 只在启动时读一次，备份成 `checkpoint.start.npz`，
   且已存在就不再覆盖（重跑不会把"上一轮的结果"当起点）。
2. **原子发布**：采纳后用 `os.replace` 覆盖 `checkpoint.npz` 与 `best_checkpoint.npz`，
   复制前后各算一次 sha256，不一致就报错，绝不静默使用半截文件。

只做 2 不做 1 会重新踩 P3 的坑：第 1 轮采纳的是 6 条轨迹的权重平均点，
一旦把它写回默认路径，第 2 轮 `executor` 会把「平均点」当成续训起点，
而平均点不可续训（302.75 → 352.7）。备份的存在让"发布"与"重启来源"互不干扰。
"""
from pathlib import Path
import hashlib
import os
import shutil
import tempfile

CHUNK = 1 << 20


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _fsync_dir(path):
    """目录项落盘；Windows 上 os.open 目录会失败，直接跳过。"""
    if os.name != "posix":
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_copy(source, destination):
    """同目录临时文件 → fsync → 校验 sha → os.replace。返回源文件 sha256。"""
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError("source checkpoint missing: %s" % source)

    expected = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    handle, temporary = tempfile.mkstemp(prefix=".ckpt-", dir=str(destination.parent))
    try:
        with os.fdopen(handle, "wb") as target, source.open("rb") as original:
            shutil.copyfileobj(original, target, CHUNK)
            target.flush()
            os.fsync(target.fileno())
        copied = sha256_file(temporary)
        if copied != expected:
            raise OSError("checkpoint hash mismatch after copy: %s != %s" % (copied, expected))
        os.replace(temporary, destination)
        _fsync_dir(destination.parent)
        return expected
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def ensure_start_backup(ckpt, backup):
    """原始起点只备份一次。已存在则直接复用，返回其 sha256。"""
    ckpt = Path(ckpt)
    backup = Path(backup)
    if backup.is_file():
        return {"created": False, "path": str(backup), "sha256": sha256_file(backup)}
    digest = atomic_copy(ckpt, backup)
    return {"created": True, "path": str(backup), "sha256": digest}


def publish(source, targets):
    """依次发布到多个目标路径。

    单个目标失败（例如被只读挂载）不中断闭环，只在返回值里记 errors——
    因为"某一处写不进去"远好过"整轮训练崩掉"。
    """
    published = {}
    errors = []
    digest = None
    for target in targets:
        try:
            digest = atomic_copy(source, target)
            published[str(target)] = digest
        except Exception as error:  # noqa: BLE001 - 发布失败要记账，不能中断主循环
            errors.append("%s: %s: %s" % (target, type(error).__name__, error))
    return {"sha256": digest, "published": published, "errors": errors}
