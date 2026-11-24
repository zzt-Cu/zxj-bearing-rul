# 使用预训练基线继续优化

先阅读 [本数据集的任务与标签说明](TASK.md)。包中的 `checkpoint.npz` 是已经训练的 CPU 模型；`training.json` 记录原训练输入摘要、源代码摘要、训练轮数和验证结果，`history.jsonl` 是逐 epoch 记录。补充文档或重新打包不会重写这些历史训练证据。

## 文件与数据边界

- `baselines/`：可供修改的模型、训练、预测代码和配置。
- `public/train.jsonl`：训练样本及目标，梯度更新只使用该集合。
- `public/validation.jsonl`：公开验证样本及目标，用于衡量改动；不要合并进训练集。
- `checkpoint.npz`：模型权重、Adam 一阶与二阶矩、累计步数、随机种子、输入标准化和目标缩放状态。

一行 JSONL 的 `inputs` 才是预测输入，`answer` 是监督目标；`id`、`meta` 用于追溯。不要将目标、完整寿命信息或隐藏标签加入模型输入。隐藏数据由平台独立保管，本包不包含隐藏样本、名单或答案。

## 第一轮与后续轮

安装依赖：`python -m pip install -r baselines/requirements.txt`。将下面 `B` 换成本包的 A、B 或 C：

```sh
python -m baselines.train --profile B \
  --train public/train.jsonl --validation public/validation.jsonl \
  --from-baseline checkpoint.npz --epochs 5 --output round-1 --stage-id round-1

python -m baselines.train --profile B \
  --train public/train.jsonl --validation public/validation.jsonl \
  --resume round-1/checkpoint.npz --epochs 5 --output round-2 --stage-id round-2
```

使用实际上一轮选定的检查点；若平台没有采纳某轮候选，应从仍被选中的模型继续。输出目录必须是新目录。`--from-baseline` 和 `--resume` 都恢复权重及优化器，不会重新初始化。

可修改代码与学习率、批大小、训练轮数等参数。改变网络结构、输入维度或任务时，需显式实现兼容的权重迁移，不能将一个新模型声明为原模型续训。参考加载器会拒绝不匹配的特征、任务、类别或网络宽度。

输入均值、标准差及回归目标缩放最初只在训练集拟合；续训沿用检查点状态。验证集和隐藏集不能参与拟合这些统计量。回归预测还原为任务原单位，并截断为非负值。

## 每轮结果

检查新目录的 `training.json`、`history.jsonl` 与 `checkpoint.npz`：父检查点摘要应对应实际起点，累计步数应增加；如实比较训练前后的验证指标。A 的 Macro-F1 越高越好，B/C 的 RMSE 越低越好。平台会独立运行预测，训练日志中的自报指标不能替代平台评测。

本地 CPU 基线不调用 PyTRiO。平台的 PyTRiO 模型与 Harness 用于阅读、修改和运行这些代码；工业模型始终是这些训练代码产生的检查点。
