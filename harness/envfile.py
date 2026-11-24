"""从本地 .env.pytrio 读取凭证，省掉每次开 shell 都 export。

为什么单独做一个文件：Key 属于凭证，不该写进代码、也不该打进评测镜像。
这里只在"文件存在且该变量尚未设置"时才填充，所以：

- 本地开发：把 Key 贴进 pkg_b/.env.pytrio 即可，不用改任何代码；
- 评测镜像内：.dockerignore 已排除该文件，这里是空操作，仍走真实环境变量；
- 显式 export 优先于文件内容，便于临时覆盖。
"""
from pathlib import Path
import os

_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = (_ROOT / ".env.pytrio", _ROOT / ".env")


def _parse(path):
    filled = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            filled.append(key)
    return filled


def load(verbose=False):
    """返回本次从文件里补上的变量名列表（不含值，避免密钥进日志）。"""
    filled = []
    for path in CANDIDATES:
        if path.is_file():
            filled.extend(_parse(path))
    if verbose and filled:
        print("[env] 已从本地凭证文件载入: " + ", ".join(filled))
    return filled
