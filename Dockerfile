# 智训引擎 B 题 Harness 镜像（默认，Alpine 瘦身版）
# 平台要求：Linux/amd64、docker save 导出的未压缩 tar ≤ 50 MB、不含源码压缩包。
# 基础镜像用 musl 版 python:3.12-alpine：slim 版导出 64.7 MB 会超限，本版 40.5 MB。
# 代码、数据、环境变量、入口与 Dockerfile.slim 完全一致，只有基础镜像不同；
# numpy 1.26.4 有 musllinux 轮子，不需要编译工具链（见下方 --only-binary）。
FROM --platform=linux/amd64 python:3.12-alpine

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BASELINE_CPU_THREADS=2 \
    RUN_DIR=/app/runs \
    STAGE_PREFIX=zxj

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
COPY baselines_ckpt/ /app/baselines_ckpt/

RUN python -c "import baselines.train, baselines.predict, harness.loop; print('imports ok')"

# 平台可能以非 root 用户启动容器。0.6 及以前 /app 是 root:root 755，非 root 建不了
# /app/runs，直接 PermissionError 退出、traceback 只进 stderr，平台侧表现为
# 「运行失败 + 无日志」。这里把 /app 放开到任意用户可写；运行期仍有 /tmp 回退兜底。
RUN mkdir -p /app/runs && chmod -R a+rwX /app

ENTRYPOINT ["python"]
CMD ["-m", "harness.preflight"]
