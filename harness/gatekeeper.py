"""Gatekeeper：上线还是回滚。规则固定、可复述，不依赖任何人工判断。

三题共用「主指标必须真的变好」这一条底线，差异在附加保护条款：

  B（RMSE↓）  原规则一字未改：分组退化 → 末端权衡 → 改善门槛 → 末端恶化
  A（Macro-F1↑）  首轮采纳；某一类 F1 崩掉 → 拒绝；未达 0.5% 改善 → 拒绝
  C（RMSE↓）  首轮采纳；未达改善门槛 → 拒绝；RMSE 几乎没动而 NASA 分数明显恶化 → 拒绝

B 的分支走的是 0.8 的原始判断顺序与文案，因为那套决策链已经写进已提交的文档，
改一个字都会让「302.75 采纳 / 305.99 回滚」的复述对不上。
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


def _class_breach(spec, state, metrics):
    """分类版的「单组崩掉」：某一类 F1 相对上一版跌得太多，即使均值涨了也回滚。

    Macro-F1 是六类平均，均值涨 1% 完全可能是「两类大涨、一类塌到 0」换来的。
    """
    floor = spec.per_class_floor
    best_classes = state.get("best_classes") or {}
    if not floor or not best_classes:
        return None
    worst = None
    for name, info in (metrics.get("per_class") or {}).items():
        before = best_classes.get(name)
        if before is None or before <= 0:
            continue
        drop = before - info["f1"]
        if worst is None or drop > worst[1]:
            worst = (name, drop, before, info["f1"])
    if worst is None or worst[1] <= floor:
        return None
    name, drop, before, after = worst
    return ("%s 的 F1 从 %.3f 掉到 %.3f（-%.3f，超过 %.2f 上限），"
            "宏平均不能掩盖单类崩塌，回滚" % (name, before, after, drop, floor))


def _decide_b(spec, state, metrics):
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


def _decide_a(spec, state, metrics):
    score = metrics["macro_f1"]
    best = state.get("best_score")
    if best is None:
        return {"action": "accept",
                "reason": "首个可评估轮次，Macro-F1 %.4f 作为当前最好" % score}

    breach = _class_breach(spec, state, metrics)
    if breach:
        return {"action": "reject", "reason": breach}

    if score < best * (1 + spec.min_gain):
        return {"action": "reject",
                "reason": "Macro-F1 %.4f 相对最好 %.4f 未达 %.1f%% 改善门槛，回滚" % (
                    score, best, spec.min_gain * 100)}

    return {"action": "accept",
            "reason": "Macro-F1 %.4f 优于最好 %.4f，改善 %.2f%%" % (
                score, best, (score - best) / best * 100)}


def _decide_c(spec, state, metrics):
    score = metrics["rmse"]
    best = state.get("best_score")
    if best is None:
        return {"action": "accept",
                "reason": "首个可评估轮次，RMSE %.2f 作为当前最好（NASA 分数 %.0f）" % (
                    score, metrics.get("nasa_score") or 0.0)}

    if score > best * (1 - spec.min_gain):
        return {"action": "reject",
                "reason": "RMSE %.2f 相对最好 %.2f 未达 %.1f%% 改善门槛，回滚" % (
                    score, best, spec.min_gain * 100)}

    # NASA 保护：RMSE 几乎没动（改善 <2%）但非对称分数明显恶化时，高估惩罚会把收益吃掉。
    guard = 0.20
    best_nasa = state.get("best_nasa_score")
    nasa = metrics.get("nasa_score")
    if best_nasa and nasa and score <= best * 0.98 and nasa > best_nasa * (1 + guard):
        return {"action": "reject",
                "reason": "RMSE 仅改善 %.2f%%，但 NASA 分数从 %.0f 恶化到 %.0f（+%.0f%%，超过 %.0f%% 上限），"
                          "高估惩罚更重，回滚" % ((best - score) / best * 100, best_nasa, nasa,
                                                (nasa - best_nasa) / best_nasa * 100, guard * 100)}

    return {"action": "accept",
            "reason": "RMSE %.2f 优于最好 %.2f，改善 %.2f%%" % (
                score, best, (best - score) / best * 100)}


def decide(spec, state, metrics):
    """返回 {'action': 'accept'|'reject'|'advance_unverified', 'reason': str}

    无验证集时（平台未给标签）降级：固定日程推进，不声称任何提升。
    注意 action 是 advance_unverified 而不是 accept —— 日志里不能把"没核实的推进"
    写成"接受"，否则审计时会以为它比上一轮更好。
    """
    if metrics is None:
        return {"action": "advance_unverified",
                "reason": "无验证集，按固定日程推进并落盘；本轮提升未经本地核实"}

    if spec.profile == "B":
        return _decide_b(spec, state, metrics)
    if spec.profile == "A":
        return _decide_a(spec, state, metrics)
    return _decide_c(spec, state, metrics)
