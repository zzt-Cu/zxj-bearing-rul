# A：气体传感器漂移下的六类气体识别

## 任务与输入

数据来自 [UCI Gas Sensor Array Drift（#224）](https://archive.ics.uci.edu/dataset/224/gas)。根据一次气体测量的 128 个数值特征，预测气体类别。平台按原文件的特征编号 1–128 排序，不生成额外的寿命或阶段标签。JSONL 的 `inputs.features` 为长度 128 的数组；`answer` 为下表中的英文类别。

| 原始类别编号 | answer | 中文名称 |
| --- | --- | --- |
| 1 | Ethanol | 乙醇 |
| 2 | Ethylene | 乙烯 |
| 3 | Ammonia | 氨 |
| 4 | Acetaldehyde | 乙醛 |
| 5 | Acetone | 丙酮 |
| 6 | Toluene | 甲苯 |

## 标签如何生成

原始 `.dat` 每行的第一个整数是官方类别编号，后面是 128 项 `编号:数值`。平台仅将该整数按上表映射为字符串，并保留全部有限特征。转换器检查每个特征编号恰好出现一次；不会根据浓度、时间、模型预测或经验阈值重标类别。转换代码为项目的 `scripts/prepare_gas_sensor.py`。

训练和验证保持批次与时间顺序隔离，不能随机混合所有批次再切分。本包只提供公开的 `public/train.jsonl` 和 `public/validation.jsonl`，实际样本数和摘要见 `training.json`；隐藏测试内容与名单不随包分发。`id`、`meta.source_batch`、`meta.source_index` 仅作追溯，不是参考模型输入。

## 评价与续训

主指标为 Macro-F1，越高越好；辅助查看 Accuracy 和混淆矩阵。分类预测输出每类的标签与分数，类别名称必须与上表一致。不要将输入中的某个连续数值当成标签。

参考模型为 CPU NumPy MLP，128 个输入、32 个 tanh 隐藏单元、6 类 softmax。使用随包检查点续训，保留 Adam 状态及训练集拟合的标准化参数。具体命令和隔离要求见 [续训指南](TRAINING_GUIDE.md)。
