# 智训引擎 · 三题提交源码

协议 `contest.harness.workspace.v2`，SDK `contest_harness` 1.0.5。一份镜像跑三个提交位：
A 气体六分类、B 轴承剩余寿命（分钟）、C 涡扇剩余寿命（循环）。题型由宿主给的 `rnd.dataset_key` 判定。

## 目录

```
harness.py               入口模板，把 argv 交给 contest_harness.cli（原样取自 SDK 的 template/）
contest_submission.py    train(rnd) 与 predict(sample, model_dir)，本轮改造的全部逻辑都在这里
baselines/model.py       MLP + Adam + 可续训的非 pickle 检查点（读平台下发的 baselines，未换结构）
baselines/data.py        inputs 展开（flatten）与数据集校验
Dockerfile               镜像定义；不设 ENTRYPOINT，CMD 只给裸 docker run 一个默认值
tools/host.py            本地 workspace v2 宿主模拟器，用于不消耗平台次数地跑满十轮
evidence/summary_*.json  三题各 10 轮的逐轮记录（假设、配置、实测指标、是否被采纳）
evidence/*/checkpoint.npz 每题最终发布的检查点
SHA256SUMS               本包内文件的摘要
```

## 与镜像的关系

`harness.py`、`contest_submission.py`、`baselines/*.py` 与镜像内 `/app` 下的同名文件逐字一致。
镜像标签 `zxj-sdk:1.0.0`，`/app` 只读，运行期只写 `/workspace` 与 `/tmp`。

## SDK

`contest_harness` 1.0.5 由平台下发，**不随本包分发**。构建镜像前先安装：

```bash
tar -xzf contest-harness-sdk-1.0.5.tar.gz
cd contest-harness-sdk-1.0.5 && python3 -m pip install .
```

Dockerfile 里对应的是把 `contest_harness` 拷进
`/usr/local/lib/python3.12/site-packages/contest_harness`，即装进 site-packages，
而不是运行时插 `sys.path`。

## 本地复现

```bash
docker build --platform linux/amd64 -t zxj-sdk:1.0.0 .

# 入口自检（应打印一行 JSON）
docker run --rm --network none zxj-sdk:1.0.0 python3 /app/harness.py describe

# 按协议跑满十轮（模拟宿主，不消耗平台次数）
python tools/host.py --dataset A --rounds 10 --workdir runs_a
python tools/host.py --dataset B --rounds 10 --workdir runs_b
python tools/host.py --dataset C --rounds 10 --workdir runs_c
```

模拟器以 `--read-only --tmpfs /tmp` 启动容器，`/app` 不可写；每轮起一个新容器，候选由
`python3 /app/harness.py predict` 在公开验证集上重新测量，不复用训练进程里的模型对象。

数据需挂到 `_abc/pkg_{a,b,c}/public/`（`train.jsonl` 与 `validation.jsonl`），这是平台公开投影的本地副本，
**不在本包内分发**。

## 三题结果（本地十轮，公开验证集）

| 题 | 指标 | 官方基线 | 十轮最终 | 变化 | 采纳轮次 |
|---|---|---|---|---|---|
| A | Macro-F1 | 0.714030 | 0.728812 | +2.1% | 1 |
| B | RMSE（分钟） | 681.159833 | 303.634878 | -55.4% | 1 |
| C | RMSE（循环） | 52.868573 | 51.135918 | -3.3% | 1、4 |

一致性检验：把官方起点当父产物、走 `predict` 入口预测，三题得到 0.714030 / 681.159833 / 52.868573，
与平台公布的基线逐位一致。

## 设计要点

1. 题型只认 `rnd.dataset_key`；输入整形与目标幂次必须在 `import baselines` 之前写进环境变量——
   `model.py` 在导入期就把它们读成模块常量，导入后再改不生效。
2. 每轮至少一次真实模型调用（`rnd.model()`），返回 JSON 超参；解析失败或越界时退回内置排程。
3. 换挡：被回滚过的配置不原样重放，连续被拒按阶梯收缩，第三档之后插入一次高学习率探针（应对漂移）。
4. 门控按题分档：A 0.5%、B 0.2%、C 0.1%。验证集规模差一个量级，同一百分比在 C 上会误杀真实改善。
5. 首轮预算按各题历史最佳设定（A 12 ep / lr 0.005、B 105 ep / lr 0.03 六轨迹集成、C 30 ep / lr 0.003），
   其后各轮只做小步微调——这三个数据集在一次充分训练后即饱和。
