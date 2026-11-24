#!/usr/bin/env bash
# 构建 linux/amd64 镜像并导出为未压缩 tar。
# 用法：bash build_image.sh [tag] [输出名]
set -euo pipefail

TAG="${1:-zxj-harness:0.1}"
OUT="${2:-zxj-harness-0.1.tar}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if ! docker info >/dev/null 2>&1; then
  echo "[x] Docker 守护进程不可用。Windows 请先启动 Docker Desktop 的 Linux engine。" >&2
  exit 1
fi

echo "[1/4] 构建（linux/amd64）"
docker buildx build --platform linux/amd64 -t "$TAG" --load .

echo "[2/4] 容器内自检（合成数据，无需比赛数据）"
docker run --rm --entrypoint python "$TAG" selfcheck.py

echo "[3/4] 导出未压缩 tar"
rm -f "$OUT"
docker save -o "$OUT" "$TAG"

# Docker 29 默认 containerd 存储，docker save 产出 OCI 布局。平台校验器可能只认经典
# Docker v2 布局，因此额外导出一份经典格式，两者都验证过可被 docker load 还原。
CLASSIC="${OUT%.tar}-classic.tar"
rm -f "$CLASSIC"
echo "[3.5/4] 额外导出经典 Docker v2 布局"
python to_classic_tar.py "$OUT" "$CLASSIC" "$TAG"

echo "[4/4] 结果"
ls -lh "$OUT" "$CLASSIC"
python - "$OUT" "$CLASSIC" <<'PY' || sha256sum "$OUT" "$CLASSIC" 2>/dev/null || true
import hashlib, sys, pathlib
for arg in sys.argv[1:]:
    p = pathlib.Path(arg)
    print("%s\n  sha256: %s\n  size  : %d bytes" % (p.name, hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_size))
PY

echo "完成。两个文件都是未压缩 .tar："
echo "  ${OUT}      OCI 布局（docker save 原生产物）"
echo "  ${CLASSIC}  经典 Docker v2 布局（体积更大，兼容性更保守）"
echo "平台报格式不符时优先换第二份。"
