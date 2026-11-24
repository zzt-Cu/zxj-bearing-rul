"""智训引擎 B 题：受控自进化 Harness。

六个角色，单文件一责：
  observer  读历史与数据事实
  planner   提出本轮实验方案（策略表 / PyTRiO 大模型）
  executor  执行一轮续训
  evaluator 在公开验证集上独立复评
  gatekeeper 上线或回滚
  recorder  落决策日志与可恢复状态

不变式：
  * 不动 checkpoint 的 hidden / seed / paths / task
  * 每轮输出新目录，不覆盖历史
  * 验证集只用于衡量，不参与拟合
  * 平台无验证标签时降级为固定日程，不伪造成绩
"""
__version__ = "0.1.0"
