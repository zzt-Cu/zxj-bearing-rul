# 智训引擎 · contest.harness.workspace.v2 提交镜像
#
# 与 0.9 的根本区别：平台协议换了。0.9 的入口是 harness.preflight（自定义 JSONL 帧），
# 平台现在要求固定入口 /app/harness.py describe|train|predict，SDK 1.0.5，
# 协议 contest.harness.workspace.v2，训练与预测都通过 contest_harness 通信。
#
# 三条硬约束来自 HARNESS_SDK.md：
#   1. /app 只读——镜像内不得依赖在 /app 下写文件；运行期只写 /workspace 与 /tmp。
#   2. 权重从平台父产物取（/previous），不把任何检查点烤进镜像。
#   3. 密钥不入镜像——本文件与构建上下文都不含 PyTRiO 凭据。
FROM --platform=linux/amd64 python:3.12-alpine

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BASELINE_CPU_THREADS=2 \
    CONTEST_WORK=/workspace

WORKDIR /app

ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir --disable-pip-version-check \
        --index-url "$PIP_INDEX" --retries 8 --timeout 120 \
        --only-binary=:all: numpy==1.26.4 \
    && python -c "import numpy, sys; print('numpy', numpy.__version__, 'python', sys.version.split()[0]); assert numpy.__version__ == '1.26.4'" \
    && rm -rf /root/.cache /tmp/*

# SDK 装进 site-packages（不是简单放在 /app 靠 sys.path 碰运气）。
COPY contest_harness /usr/local/lib/python3.12/site-packages/contest_harness

COPY harness.py /app/harness.py
COPY contest_submission.py /app/contest_submission.py
COPY baselines /app/baselines

RUN python -c "import contest_harness, baselines.model; print('sdk', contest_harness.SDK_VERSION, 'imports ok')" \
    && python /app/harness.py --version

# describe 是最便宜的契约检查：它导入选手模块、检查 train/predict 与任务声明。
RUN python /app/harness.py describe

# 不设 ENTRYPOINT：平台按「python3 /app/harness.py <describe|train|predict>」调用，
# 任何前置 entrypoint 都会把这行命令拼坏。CMD 只给裸 docker run 一个可用的默认值。
ENTRYPOINT []
CMD ["python3", "/app/harness.py", "describe"]
