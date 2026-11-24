# CPU 工业模型参考基线

这套代码提供可继续训练的小型 NumPy MLP。训练只读取显式传入的公开训练集和验证集，不扫描数据目录，不读取隐藏测试数据。平台必须继续在文件挂载和数据释放层隔离隐藏标签；文件名或命令行参数本身不是安全边界。

每个下载包的 `TASK.md` 说明对应任务、输入、标签生成和指标；`TRAINING_GUIDE.md` 说明如何从所附权重继续训练。源文档位于 [A](docs/A.md)、[B](docs/B.md)、[C](docs/C.md) 与 [续训指南](docs/TRAINING_GUIDE.md)。

| 配置 | 任务 | 网络 |
| --- | --- | --- |
| A | 六类气体识别 | 输入特征 → 32 个 tanh 隐藏单元 → 6 类 softmax |
| B | 轴承剩余寿命回归（分钟） | 输入特征 → 24 个 tanh 隐藏单元 → 1 个线性输出 |
| C | 发动机剩余寿命回归（循环） | 输入特征 → 32 个 tanh 隐藏单元 → 1 个线性输出 |

所有参考模型均限制在 100,000 个参数以内，只依赖 CPU NumPy，不调用 GPU。默认固定种子 42；命令行入口在导入 NumPy 前请求单个 BLAS 线程，可用 `BASELINE_CPU_THREADS=1..8` 指定请求的 CPU 线程数。若嵌入已加载 NumPy 的进程，实际线程限制仍由宿主负责；报告记录请求值，不伪装成资源计量。特征标准化和 RUL 目标缩放只从初始训练集拟合，续训时保留。回归预测还原为原单位并截断到非负值。

B 使用 XJTU-SY 官方完整运行至失效记录，目标为 `(最后快照序号 − 当前快照序号) × 1 分钟`，最后快照为 0 分钟。它是从官方记录确定性计算的 RUL，不是官方提供的逐样本健康分级标签。训练、验证和隐藏测试按整根轴承分开；模型输入不得包含真实 RUL 或寿命比例。评价采用分钟 RMSE 和 MAE，越低越好。B 默认配置不再使用三类健康标签，也不要求健康分级审核文件；历史分类基线仍保留原版本，不能作为新 RUL 模型的续训起点。

## 环境与数据

参考锁定环境为 Python 3.10–3.12、NumPy 1.26.4。所有构建和验收按项目约定在指定远程机器执行。

```sh
python3 -m venv .venv
.venv/bin/pip install -r baselines/requirements.txt
```

JSONL 每行包含 `id`、`inputs`、`answer`；`inputs` 可为嵌套数值对象／数组，必须具有一致字段路径和数组长度。分类答案为配置中的字符串类别，RUL 为非负数值。训练和验证 ID 不得重叠。`meta` 仅保留在输入数据文件，模型不会读取它。

## 生成参考基线

```sh
.venv/bin/python -m baselines.train --profile A \
  --train /public/A/train.jsonl --validation /public/A/validation.jsonl \
  --output /artifacts/baseline-A --stage-id baseline

.venv/bin/python -m baselines.train --profile B \
  --train /public/B/train.jsonl --validation /public/B/validation.jsonl \
  --output /artifacts/baseline-B --stage-id baseline

.venv/bin/python -m baselines.train --profile C \
  --train /public/C/train.jsonl --validation /public/C/validation.jsonl \
  --output /artifacts/baseline-C --stage-id baseline
```

C 的 FD001–FD004 若合成一个官方数据流，可以拼接其公开 train/validation JSONL，前提是样本 ID 带子集前缀且特征 schema 相同；不可把官方测试标签加入其中。也可以分别训练子集基线，但平台必须明确采用哪种配置，不能把不同设置的误差混用。

每个新输出目录包含：

- `checkpoint.npz`：权重、Adam 一阶／二阶矩、累计更新步数、epoch、随机种子、类别、特征 schema、标准化及目标缩放参数。使用固定 ZIP 元数据，不包含 pickle。
- `training.json`：输入文件摘要、配置摘要、父检查点摘要、步数变化、验证前后指标、训练耗时与运行环境。
- `history.jsonl`：逐 epoch 的 loss 与公开验证指标，带阶段和种子。

文件不会覆盖前一轮目录。所有报告标记为开发参考，不能直接作为正式比赛成绩。基线固定训练轮数，不根据隐藏集选模型；训练耗时来自实际进程，不是 GPU 时长或平台训练额度。

## 选手续训

第一轮必须使用平台给出的基线检查点；后续轮使用上一轮选定的检查点。两种参数均恢复全部优化器和标准化状态，没有重新初始化模型的分支。

```sh
.venv/bin/python -m baselines.train --profile A \
  --train ./train.jsonl --validation ./validation.jsonl \
  --from-baseline ./baseline/checkpoint.npz \
  --epochs 5 --learning-rate 0.001 --output ./runs/round-01 --stage-id round-01

.venv/bin/python -m baselines.train --profile A \
  --train ./train.jsonl --validation ./validation.jsonl \
  --resume ./runs/round-01/checkpoint.npz \
  --epochs 5 --output ./runs/round-02 --stage-id round-02
```

选手可修改训练代码、参数及策略；参考加载器会拒绝错误任务、类别或特征 schema。改变网络结构时，需要显式实现权重迁移并保留来源证据，不能假称沿用了不兼容的检查点。平台仍需独立核验父检查点和实际执行过程，不能只相信选手自报日志。

## 预测与验证

```sh
printf '%s\n' '{"task":"regression","inputs":{"sensors":[0.1,0.2],"settings":[1,2,3]}}' | \
  .venv/bin/python -m baselines.predict --checkpoint /artifacts/model/checkpoint.npz
```

上面的输入仅展示协议；实际字段和维度必须与检查点一致。分类输出 `outputs:[{label,score}]`，回归输出 `outputs:{value}`，与平台预测协议一致。模型加载与预测应在选手隔离环境中执行，不能因为格式是 NPZ 就放到业务 API 进程里处理。

```sh
.venv/bin/python -m unittest discover -s baselines/tests -v
```

测试覆盖 CPU 数组、确定性检查点、续训与连续训练等价、Adam 状态、禁止读取隐藏文件、标签与特征约束、数据隔离及父检查点绑定。配置中的训练参数是参考设置，不是已批准赛规，也不承诺每轮验证成绩必然提高。
# 续训架构约束

参考训练器要求 profile 的 hidden、seed、任务、类别及输入特征与检查点一致。只修改 `hidden` 不会调整权重形状；不一致会在训练前报错，不会静默忽略或重置权重。架构迁移必须由训练代码显式迁移权重与优化器状态并独立验证，当前参考训练器不提供自动迁移。
