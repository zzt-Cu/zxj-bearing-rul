# 代码来源与许可说明

## 一、来源划分

| 范围 | 来源 | 说明 |
|---|---|---|
| baselines/ 中的 train.py、predict.py、data.py、configs/、tests/ | 赛题官方下发的基线代码包 | 未改动逻辑；仅 model.py 与 train.py 按下述改动 |
| baselines/model.py | 官方代码 + 本队改动 | 改动点见下节第 1~3 条 |
| baselines/train.py | 官方代码 + 本队改动 | 改动点见下节第 4 条（集成入口） |
| harness/ 全部模块 | 本队原创 | 观察、提案、执行、评估、门控、记录、主循环 |
| selfcheck.py、pytrio_probe.py、mock_pytrio.py | 本队原创 | 自检、接入探测、本地模拟 |
| Dockerfile、build_image.sh、to_classic_tar.py | 本队原创 | 构建与格式转换 |

## 二、对官方代码的改动（逐条可核）

1. baselines/model.py：新增 ZXJ_INPUT_MODE 环境变量（默认 abs8log）。
   abs8：对 16 个绝对量特征（每通道 8 个）在整体标准化之后追加一次逐样本标准化，
   其余 18 个无量纲特征不做二次归一。
   abs8log：在 abs8 基础上追加 14 列自然对数绝对量（两通道 × 7 个量：rms、peak、
   mean_abs、spectral_rms、band1/2/3_energy），下限保护 log(max(x, 1e-12))，
   按训练集统计量标准化后拼到输入尾部，输入宽度 34 -> 48。外部契约不变。
2. baselines/model.py：新增 ZXJ_TARGET_POWER 环境变量（默认 0.45）。
   训练目标改为 (y**p - mean)/scale，预测时以 (z*scale + mean)**(1/p) 还原到分钟；
   target_power 与统计量写入检查点 meta，载入时若与进程配置不一致直接报错。
   同时新增 expand_inputs()：从 34 维起点检查点扩展输入宽度时，新增列用与常规
   初始化同分布的小随机数起步（不置零），并保持随机流可复现。
3. baselines/model.py：validate() 增加输入宽度与派生列一致性校验，防止用错模式
   的检查点静默产出错误预测。
4. baselines/train.py：新增 --ensemble 参数与 ENSEMBLE_BATCH 环境变量。
   同一续训起点按不同 batch_size 各训一条轨迹，结束后对 w1/w2/b1/b2 与 Adam
   一阶、二阶矩取算术平均，并把 target_power/target_mean/target_scale 同步为
   轨迹值（否则合并结果会落在错误的目标空间）。不传该参数时行为与官方一致。

以上改动均不改变模型结构：hidden=24、seed=42、34 维特征路径、task=regression
四项保持赛题原值，官方 --resume 校验全部通过。参数总数由 865 变为 1,201，
来源是输入宽度 34 -> 48，不是加宽隐层。

## 三、第三方依赖

numpy 1.26.4（BSD 3-Clause License），由镜像内的 pip 安装，本包不包含其二进制。

## 四、使用范围

本包为参加「智训引擎」B 题企业赛的提交材料，仅供赛题评审与复现使用。
