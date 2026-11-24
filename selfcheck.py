"""容器自检：不依赖任何比赛数据，30 秒内证明整条链路是通的。

检查项
  1 numpy / python 版本
  2 检查点可加载，输入整形模式自洽
  3 用合成数据打通 harness.loop（真实子进程，不是函数调用）
  4 决策日志与状态文件落地
  5 官方预测契约 baselines.predict 能输出 {"outputs": {"value": ...}}
  7 真实数据可读（--real 时）
  8 PyTRiO 配置
  9 检查点发布：采纳后写回默认路径，且第 2 轮从"备份的官方起点"重启（P0/P3 回归）
 10 决策日志容错：截断的尾行不会让整轮恢复失败

用法：python selfcheck.py            # 合成数据自检
      python selfcheck.py --real     # 用 DATA_DIR 下的真实数据自检
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))

import numpy as np

from baselines.data import read_dataset
from baselines.model import Model
from baselines.predict import predict
from harness import publish, recorder

FAILS = []


def check(name, fn):
    try:
        detail = fn()
        print("[PASS] %-38s %s" % (name, detail if detail is not None else ""))
        return True
    except Exception as error:
        print("[FAIL] %-38s %s: %s" % (name, type(error).__name__, error))
        FAILS.append(name)
        return False


def nested_from_paths(paths, values):
    """把扁平特征按官方字段路径还原成嵌套 inputs。"""
    root = {}
    for raw, value in zip(paths, values):
        steps = json.loads(raw)
        cursor = root
        for index, (kind, key) in enumerate(steps):
            last = index == len(steps) - 1
            if kind == "key":
                if last:
                    cursor[key] = float(value)
                else:
                    cursor = cursor.setdefault(key, {})
            else:
                raise ValueError("unsupported path element: %r" % (kind,))
    return root


def make_dataset(path, paths, rows, seed, namespace):
    rng = np.random.default_rng(seed)
    with Path(path).open("w", encoding="utf-8") as stream:
        for index in range(rows):
            base = rng.normal(0, 1, len(paths))
            minutes = float(abs(rng.normal(700, 500)))
            inputs = nested_from_paths(paths, base)
            stream.write(json.dumps({"id": "syn_%s_bearing%d_s%06d" % (namespace, index % 4 + 1, index),
                                     "inputs": inputs, "answer": minutes},
                                    ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="用真实数据，而非合成数据")
    parser.add_argument("--data", help="真实数据目录（含 train.jsonl / validation.jsonl）")
    parser.add_argument("--ckpt", default=str(APP / "checkpoint.npz"))
    args = parser.parse_args(argv)

    print("=" * 62)
    print("智训引擎 B 题 Harness 容器自检")
    print("=" * 62)

    def versions():
        return "python %s / numpy %s / %s" % (sys.version.split()[0], np.__version__, sys.platform)
    check("1. 运行环境", versions)

    def load_ckpt():
        model = Model.load(args.ckpt)
        return "%d 特征 / hidden %d / %d 参数 / input_mode=%s" % (
            len(model.meta["paths"]), model.meta["hidden"],
            model.meta["parameter_count"], model.meta.get("input_mode"))
    ok = check("2. 检查点加载与输入整形", load_ckpt)
    if not ok:
        return 1
    model = Model.load(args.ckpt)
    paths = model.meta["paths"]

    def pytrio_ready():
        key = os.environ.get("PYTRiO_API_KEY", "")
        base = os.environ.get("PYTRiO_BASE_URL", "")
        model_name = os.environ.get("PYTRiO_MODEL", "")
        if not key:
            return "跳过（未配置 PYTRiO_API_KEY，planner 走策略表，功能不受影响）"
        if not base.startswith("http"):
            raise RuntimeError("已配置 Key 但 PYTRiO_BASE_URL 不合法：%r" % base)
        if not model_name:
            raise RuntimeError("已配置 Key 但 PYTRiO_MODEL 为空")
        return "已配置 base=%s model=%s；连通性请另跑 pytrio_probe.py" % (base, model_name)
    check("8. PyTRiO 配置", pytrio_ready)

    with tempfile.TemporaryDirectory(prefix="selfcheck-") as temporary:
        root = Path(temporary)
        # 自检会真实跑一轮并触发"检查点发布"，所以绝不能拿镜像里的 checkpoint.npz 当目标：
        # 复制到临时目录再跑，镜像里的官方起点一个字节都不能动。
        ckpt = root / "checkpoint.npz"
        shutil.copyfile(Path(args.ckpt), ckpt)
        if args.real:
            data_dir = Path(args.data) if args.data else APP / "public"
            print("使用真实数据：%s" % data_dir)
        else:
            data_dir = root / "data"
            data_dir.mkdir()
            make_dataset(data_dir / "train.jsonl", paths, 400, 0, "train")
            make_dataset(data_dir / "validation.jsonl", paths, 100, 1, "val")
            print("使用合成数据：400 训练 / 100 验证（指标无意义，只验通路）")

        run_dir = root / "runs"

        def loop_runs():
            command = [sys.executable, "-m", "harness.loop",
                       "--data", str(data_dir), "--ckpt", str(ckpt), "--run-dir", str(run_dir),
                       "--max-rounds", "2", "--epoch-budget", "200"]
            done = subprocess.run(command, cwd=str(APP), capture_output=True, text=True, timeout=900)
            if done.returncode != 0:
                raise RuntimeError("exit %d\nstdout:\n%s\nstderr:\n%s"
                                   % (done.returncode, done.stdout[-2000:], done.stderr[-2000:]))
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            return "rounds=%d best_rmse=%s elapsed=%ss" % (
                summary["rounds"], summary["best_rmse"], summary["elapsed_s"])
        check("3. harness.loop 端到端", loop_runs)

        def decisions_written():
            rows = [json.loads(line) for line in
                    (run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            if not rows:
                raise RuntimeError("决策日志为空")
            first = rows[0]
            for key in ("round", "observation", "proposal", "result", "decision"):
                if key not in first:
                    raise RuntimeError("决策日志缺字段 %s" % key)
            accepted = sum(1 for r in rows if r["decision"]["action"] == "accept")
            return "%d 条记录，%d 条被采纳，字段齐全" % (len(rows), accepted)
        check("4. 决策日志结构", decisions_written)

        def state_written():
            state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
            if "best_checkpoint" not in state:
                raise RuntimeError("state 缺 best_checkpoint")
            if not Path(state["best_checkpoint"]).is_file():
                raise RuntimeError("best_checkpoint 指向的文件不存在")
            return "best_checkpoint=%s" % Path(state["best_checkpoint"]).name
        check("5. 状态文件可恢复", state_written)

        def predict_contract():
            produced = None
            for line in (data_dir / "validation.jsonl").read_text(encoding="utf-8").splitlines()[:3]:
                request = json.loads(line)
                loaded = Model.load(run_dir / "round-01" / "checkpoint.npz")
                produced = predict(loaded, request["inputs"])
                if set(produced) != {"outputs"} or "value" not in produced["outputs"]:
                    raise RuntimeError("预测契约不符：%r" % produced)
                if produced["outputs"]["value"] < 0:
                    raise RuntimeError("预测值为负：%r" % produced)
            return "outputs.value = %.1f 分钟" % produced["outputs"]["value"]
        check("6. 官方预测契约", predict_contract)

        def real_data_sane():
            if not args.real:
                return "跳过（合成数据模式）"
            data = read_dataset(data_dir / "validation.jsonl", "regression", [], paths)
            return "真实验证集 %d 行 / %d 个轴承" % (len(data["ids"]),
                                              len({i.rsplit('_s', 1)[0] for i in data["ids"]}))
        check("7. 真实数据可读", real_data_sane)

        def publish_and_restart():
            """P0 回归：采纳后必须把权重写回默认路径；第 2 轮必须从备份的官方起点重启。"""
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            rows = json.loads(json.dumps(recorder.read_decisions(SimpleNamespace(run_dir=run_dir))))
            accepted = [r for r in rows if r["decision"]["action"] in ("accept", "advance_unverified")]
            if not accepted:
                raise RuntimeError("两轮里没有采纳/推进的轮次，无法验证发布")
            outcome = accepted[0]["execution"].get("publish") or {}
            if outcome.get("errors"):
                raise RuntimeError("发布报错：%s" % outcome["errors"])
            if not (outcome.get("published") or {}):
                raise RuntimeError("采纳后没有发布任何目标路径")
            # 两轮都采纳时，后一轮会覆盖前一轮的产物，所以只校验最后一次发布
            last = accepted[-1]["execution"]
            final = last.get("publish") or {}
            if final.get("errors"):
                raise RuntimeError("末次发布报错：%s" % final["errors"])
            for path, digest in (final.get("published") or {}).items():
                if publish.sha256_file(path) != digest:
                    raise RuntimeError("发布后校验失败：%s" % path)
            start_sha = summary.get("start_checkpoint_sha256")
            submitted = summary.get("submitted_checkpoint_sha256")
            if not start_sha:
                raise RuntimeError("summary 缺 start_checkpoint_sha256")
            if start_sha == submitted:
                raise RuntimeError("发布未生效：提交路径的哈希仍等于官方起点")
            if submitted != final.get("sha256"):
                raise RuntimeError("summary 与末次发布的哈希不一致")
            if len(rows) >= 2:
                second = rows[1]["execution"]
                if not second.get("restarted_from_start"):
                    raise RuntimeError("第 2 轮没有识别出权重平均点，误从平均点续训")
                recorded = str(second.get("restart_from_sha256") or "").replace("sha256:", "")
                if recorded != start_sha:
                    raise RuntimeError("第 2 轮的起点不是备份的官方起点：%s != %s"
                                       % (recorded[:12], start_sha[:12]))
            return "发布 %d 个目标 / 起点 %s… / 第 2 轮 restart_from=backup" % (
                len(final.get("published") or {}), start_sha[:12])
        check("9. 检查点发布与重启语义", publish_and_restart)

        def decisions_tolerant():
            """尾行被进程杀掉留下半截时，读取只丢最后一行，不整轮失败。"""
            path = run_dir / "decisions.jsonl"
            original = path.read_text(encoding="utf-8")
            good = len(recorder.read_decisions(SimpleNamespace(run_dir=run_dir)))
            try:
                path.write_text(original + '{"round": 99, "decision": {"action": "ac', encoding="utf-8")
                survived = recorder.read_decisions(SimpleNamespace(run_dir=run_dir))
                if len(survived) != good:
                    raise RuntimeError("残行导致记录数变化：%d -> %d" % (good, len(survived)))
            finally:
                path.write_text(original, encoding="utf-8")
            return "%d 条完整记录，追加半截尾行后仍读出 %d 条" % (good, good)
        check("10. 决策日志容错", decisions_tolerant)

    print("=" * 62)
    if FAILS:
        print("自检结果：%d 项失败 -> %s" % (len(FAILS), ", ".join(FAILS)))
        return 1
    print("自检结果：全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
