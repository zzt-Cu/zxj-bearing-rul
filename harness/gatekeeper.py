"""Gatekeeper：上线还是回滚。规则固定、可复述，不依赖任何人工判断。

判定顺序（有验证集时）：
  1. 首个可评估轮次 → 采纳
  2. 分轴承退化超限 → 拒绝（总体涨分但某根崩了，这种不能上线）
  3. 末端权衡条款 → 采纳（总体小幅退让换末端大幅改善）
  4. 未达 0.5% 改善门槛 → 拒绝
  5. 末端显著恶化 → 拒绝
  6. 其余 → 采纳
"""


def _group_breach(spec, state, metrics):
    """找出一根退化最多的轴承；超过 max_group_degrade 就返回拒绝理由。"""
    best_groups = state.get("best_groups") or {}
    worst = None
    for name, info in (metrics.get("groups") or {}).items():
        before = best_groups.get(name)
        if not before:
            continue  # 新出现的轴承没有基准，不参与比较
        ratio = info["rmse"] / before - 1.0
        if worst is None or ratio > worst[1]:
            worst = (name, ratio, before, info)
    if worst is None or worst[1] <= spec.max_group_degrade:
        return None
    name, ratio, before, info = worst
    return ("%s 的 RMSE 从 %.1f 恶化到 %.1f（+%.0f%%，超过 %.0f%% 上限），"
            "总体均值不能掩盖单根退化，回滚" % (name, before, info["rmse"],
                                              ratio * 100, spec.max_group_degrade * 100))


def decide(spec, state, metrics):
    """返回 {'action': 'accept'|'reject'|'advance_unverified', 'reason': str}

    无验证集时（平台未给标签）降级：固定日程推进，不声称任何提升。
    注意 action 是 advance_unverified 而不是 accept —— 日志里不能把"没核实的推进"
    写成"接受"，否则审计时会以为它比上一轮更好。
    """
    if metrics is None:
        return {"action": "advance_unverified",
                "reason": "无验证集，按固定日程推进并落盘；本轮提升未经本地核实"}

    rmse = metrics["rmse"]
    best = state.get("best_rmse")
    if best is None:
        return {"action": "accept", "reason": "首个可评估轮次，直接作为当前最好（RMSE %.2f）" % rmse}

    breach = _group_breach(spec, state, metrics)
    if breach:
        return {"action": "reject", "reason": breach}

    best_tail = state.get("best_tail_rmse")
    tail = metrics.get("tail_rmse")
    tail_n = int(metrics.get("tail_n") or 0)
    if tail_n < spec.min_tail_n:
        # 尾段样本太少，20% 的改善可能只是一两个样本的偶然；权衡条款不适用
        tail = None

    # 末端权衡条款：评分里「最差阶段与恢复速度」单独占分，因此当总体 RMSE 只是小幅
    # 退让、而末端大幅改善时，这一轮仍应上线。阈值由 spec 给出，不在这里写死数字。
    if (best_tail and tail
            and rmse <= best * (1 + spec.tail_trade_rmse)
            and tail <= best_tail * (1 - spec.tail_trade_gain)):
        return {"action": "accept",
                "reason": "总体 RMSE %.2f 退让 %.1f%%（未超 %.0f%% 上限），但末端 RMSE 从 %.1f 改善到 %.1f"
                          "（%.0f%%，尾段 %d 条），按末端权衡条款上线" % (
                              rmse, (rmse - best) / best * 100, spec.tail_trade_rmse * 100,
                              best_tail, tail, (best_tail - tail) / best_tail * 100, tail_n)}

    if rmse > best * (1 - spec.min_gain):
        return {"action": "reject",
                "reason": "RMSE %.2f 相对最好 %.2f 未达 %.1f%% 改善门槛，回滚" % (
                    rmse, best, spec.min_gain * 100)}

    if best_tail and tail and tail > best_tail * 1.15 and rmse > best * 0.98:
        return {"action": "reject",
                "reason": "总体仅涨 %.1f%%，但末端 RMSE 从 %.1f 恶化到 %.1f（>15%%），"
                          "会拉低「最差阶段」评分，回滚" % ((best - rmse) / best * 100, best_tail, tail)}

    return {"action": "accept",
            "reason": "RMSE %.2f 优于最好 %.2f，改善 %.1f%%" % (rmse, best, (best - rmse) / best * 100)}
