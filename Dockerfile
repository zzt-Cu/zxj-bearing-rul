# 智训引擎 B 题 Harness 镜像（默认，Alpine 瘦身版）
# 平台要求：Linux/amd64、docker save 导出的未压缩 tar ≤ 50 MB、不含源码压缩包。
# 基础镜像用 musl 版 python:3.12-alpine：slim 版导出 64.7 MB 会超限，本版 40.5 MB。
# 代码、数据、环境变量、入口与 Dockerfile.slim 完全一致，只有基础镜像不同；
# numpy 1.26.4 有 musllinux 轮子，不需要编译工具链（见下方 --only-binary）。
FROM --platform=linux/amd64 python:3.12-alpine

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BASELINE_CPU_THREADS=2 \
    ZXJ_INPUT_MODE=abs8log \
    ZXJ_TARGET_POWER=0.45 \
    ENSEMBLE_BATCH=96,112,128,144,192,256 \
    TAIL_MINUTES=100 \
    TAIL_TRADE_GAIN=0.20 \
    TAIL_TRADE_RMSE=0.06 \
    CKPT=/app/checkpoint.npz \
    RUN_DIR=/app/runs \
    MAX_ROUNDS=3 \
    EPOCH_BUDGET=330 \
    STAGE_PREFIX=zxj \
    MIN_TAIL_N=30 \
    MAX_GROUP_DEGRADE=0.25

WORKDIR /app

ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
COPY baselines/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --disable-pip-version-check \
        --index-url "$PIP_INDEX" --retries 8 --timeout 120 \
        --only-binary=:all: \
        -r /tmp/requirements.txt \
    && python -c "import numpy, sys; print('numpy', numpy.__version__, 'python', sys.version.split()[0], 'libc', sys.platform); assert numpy.__version__ == '1.26.4'" \
    && rm -rf /root/.cache /tmp/*

COPY baselines/ /app/baselines/
COPY harness/ /app/harness/
COPY selfcheck.py /app/selfcheck.py
COPY pytrio_probe.py /app/pytrio_probe.py

COPY checkpoint.npz /app/checkpoint.npz
COPY public/ /app/public/

RUN python -c "import baselines.train, baselines.predict, harness.loop; print('imports ok')"

ENTRYPOINT ["python"]
CMD ["-m", "harness.loop"]
