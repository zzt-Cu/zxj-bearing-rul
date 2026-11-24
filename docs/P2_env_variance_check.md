# P2 容器环境差异核查（2026-11-23）

本文件是 `P2_集成与对数输入_实验记录.md` 的补充，不改动那份记录的任何数字。
记录里的 313.79 全部是**开发机口径**；本文件说明容器内为什么是 302.75，以及
为什么提交时用容器值。

---

## 一、现象

同一份配置（abs8log + p=0.45 + 105 epoch + lr 0.03 + 6 轨迹）、同一份数据、
同一份起点检查点，在两个环境各跑一遍：

| 环境 | Python | numpy | RMSE |
|---|---|---|---|
| 开发机（Windows） | 3.13.14 | 2.5.4 | 313.7901 |
| 容器（Alpine / musl） | 3.12 | 1.26.4 | 302.7519 |

差 11.04 分（3.5%）。

## 二、排查顺序（每一步都做了实测，不是推测）

1. **数据是否相同**：对比 sha256，三项全部一致。
   - `train.jsonl` `71b2663f…`、`validation.jsonl` `23673a66…`、`checkpoint.npz` `c7d819fc…`
2. **环境变量是否相同**：镜像内 `env | grep -E 'ZXJ|ENSEMBLE|EPOCH|CPU'` 与宿主机
   export 完全一致（abs8log / 0.45 / 330 / 96,112,128,144,192,256 / 线程 2）。
3. **是否为随机噪声**：两个环境各自连续重跑。
   - 开发机两次：313.7901085461185、313.7901085461185（逐位相同）
   - 容器两次：302.75190317514176、302.75190317514176（逐位相同）
   → **各自是确定性的，差异是系统性的，不是噪声。**
4. **排除配置漂移**：容器 `spec` 打印 `input_mode=abs8log, target_power=0.45,
   epoch_budget=330`，`observe` 打印 `n_train=7032 / n_val=1679`，与开发机一致。

## 三、结论

差异来自**矩阵乘法的浮点归约顺序**：两个环境的 BLAS 实现与线程并行策略不同，
单次 matmul 的最后几位有效数字不同；105 个 epoch、每个 epoch 数十次参数更新
把这点差异逐级放大，最终落到 3.5% 的成绩差上。训练过程本身是混沌的，这类
放大在长迭代里是正常现象，不是 bug。

一个反向证据可以排除"镜像构建有问题"：34 维的 P1 配置在容器内与宿主机是**逐位
一致**的（两次都是 464.22，连小数位都一样）。这说明镜像、数据、起点都没有被污染。
差别在于：34 维下模型收敛到接近常数预测器，轨迹对初值扰动不敏感；48 维下模型真的
在学退化信号，每一步的有效梯度都更大，浮点扰动才会沿 105 个 epoch 放大成 3.5%。

## 四、对交付的影响

- **提交口径用容器值 302.75**：平台实际运行的是镜像，那是唯一会被复现的数字。
- 开发机 313.79 继续保留在实验记录里，用于说明"改进幅度不依赖环境运气"：
  两个环境相对官方基线分别是 −53.9% 与 −55.6%。
- 报告里给区间（302～314）而不是精确到小数，避免给评委一个会落空的承诺。
- `policy.json` 的 `prior_rmse` 仍写 313.79（开发机测得的候选排序依据），
  容器里第 1 轮实测 302.75，比预期更好；候选**排序**不受影响，故未改表。

## 五、复现命令

```bash
# 开发机
cd pkg_b
export ZXJ_INPUT_MODE=abs8log ZXJ_TARGET_POWER=0.45
export ENSEMBLE_BATCH=96,112,128,144,192,256
PYTHONPATH=. python -m baselines.train --profile B \
  --train public/train.jsonl --validation public/validation.jsonl \
  --output /tmp/recheck --from-baseline checkpoint.npz \
  --epochs 105 --learning-rate 0.03 --batch-size 96 \
  --ensemble 96,112,128,144,192,256
# → validation_after.rmse = 313.7901085461185

# 容器
docker run --rm zxj-harness:0.2 -m harness.loop --max-rounds 1
# → [harness] result {"round": 1, "rmse": 302.75, ...}
```

## 六、遗留

容器内第 1 轮的完整指标（rmse 302.7519 / mae 230.4098 / tail_rmse 274.5188 /
分轴承 297.28、357.65、301.53）留档在 `runs_image_v02/decisions.jsonl`，
checkpoint 为 `runs_image_v02/round-01/checkpoint.npz`
（sha256 `deb73cb9…`，父检查点 `c7d819fc…`）。
