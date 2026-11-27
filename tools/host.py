"""本地 workspace v2 宿主模拟器。

它把平台那一侧的最小可用实现补齐，让镜像在本机跑完 describe / train / predict，
并按 SDK 1.0.5 的期望逐帧应答。它不是平台，也不产生正式成绩：
- 模型调用返回固定决策（模拟 PyTRiO 回复），不联网；
- 候选测量在公开验证集上做，隐藏测试集不可得；
- 分块上传、状态保存、轮次确认按协议实现，用来暴露协议层错误。

用法：
    python host.py --dataset A --rounds 10 --image zxj-sdk:1.0.0
"""
import argparse
import base64
import hashlib
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = "contest.harness.workspace.v2"
PROMPT_VERSION = "industrial-workspace-flexible-v3"
TASKS = {"A": "classification", "B": "regression", "C": "regression"}
METRICS = {"A": "macro_f1", "B": "rmse", "C": "rmse"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def manifest(directory, names=None):
    root = Path(directory)
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if names is not None and relative not in names:
            continue
        entries.append({"path": relative, "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path)})
    return entries


def receipt_of(artifact_id, directory, descriptor):
    names = set(descriptor["checkpoint_paths"]) | set(descriptor["code_paths"])
    return {"artifact_id": artifact_id, "artifact": descriptor,
            "files": manifest(directory, names)}


def load_rows(dataset):
    path = ROOT / "_abc" / ("pkg_%s" % dataset.lower()) / "public" / "validation.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(task, expected, produced):
    if task == "classification":
        hit = sum(1 for e, p in zip(expected, produced) if e == p)
        f1 = []
        for label in sorted(set(expected) | set(produced)):
            tp = sum(1 for e, p in zip(expected, produced) if e == label and p == label)
            fp = sum(1 for e, p in zip(expected, produced) if e != label and p == label)
            fn = sum(1 for e, p in zip(expected, produced) if e == label and p != label)
            if tp + fp + fn:
                f1.append(2 * tp / (2 * tp + fp + fn))
        return {"macro_f1": sum(f1) / len(f1) if f1 else 0.0, "accuracy": hit / len(expected)}
    errors = [float(p) - float(e) for e, p in zip(expected, produced)]
    return {"rmse": (sum(e * e for e in errors) / len(errors)) ** 0.5,
            "mae": sum(abs(e) for e in errors) / len(errors)}


class Host:
    def __init__(self, args):
        self.args = args
        self.key = args.dataset
        self.task = TASKS[self.key]
        self.metric = METRICS[self.key]
        self.root = Path(args.workdir).resolve()
        self.artifacts = self.root / "artifacts"
        self.states = self.root / "states"
        self.uploads = self.root / "uploads"
        self.runs = self.root / "runs"
        for path in (self.artifacts, self.states, self.uploads, self.runs):
            path.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.parent_receipt = None
        self.parent_dir = None
        self.parent_checkpoint_sha = None
        self.current_state = None

    # ---------------------------------------------------------------- 容器

    def _popen(self, mounts, env, tail, log_path, stdin=True):
        cmd = ["docker", "run", "-i" if stdin else "-d", "--rm", "--network", "none",
               "--cpus", str(self.args.cpus), "--memory", self.args.memory,
               "--read-only", "--tmpfs", "/tmp:rw,exec"]
        if self.args.user:
            cmd += ["--user", self.args.user]
        for mount in mounts:
            cmd += ["-v", mount]
        for item in env:
            cmd += ["-e", item]
        cmd += [self.args.image] + tail
        log = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=log,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        proc._log = log
        proc._log_path = log_path
        return proc

    def train_process(self, work, state, log_path):
        mounts = ["%s:/data:ro" % (ROOT / "_abc" / ("pkg_%s" % self.key.lower()) / "public"),
                  "%s:/previous:ro" % self.parent_dir,
                  "%s:/workspace" % work]
        if state is not None:
            mounts.append("%s:/state:ro" % state)
        return self._popen(mounts, [], ["python3", "/app/harness.py", "train"], log_path)

    def predict(self, artifact_dir, rows):
        mounts = ["%s:/model:ro" % artifact_dir]
        context = {"protocol": "contest.predict-context.v1", "dataset_key": self.key,
                   "task": self.task, "artifact": {}, "artifact_id": "local"}
        env = ["CONTEST_ARTIFACT_PATH=/model/checkpoint.npz",
               "CONTEST_DATASET_KEY=%s" % self.key,
               "CONTEST_TASK=%s" % self.task,
               "CONTEST_PREDICTION_CONTEXT=%s" % json.dumps(context, separators=(",", ":"))]
        log_path = self.root / "predict.log"
        proc = self._popen(mounts, env, ["python3", "/app/harness.py", "predict"], log_path)
        payload = "".join(json.dumps({"task": self.task, "inputs": row["inputs"]},
                                     ensure_ascii=False) + "\n" for row in rows)
        out, _ = proc.communicate(payload, timeout=self.args.predict_timeout)
        proc._log.close()
        if proc.returncode != 0:
            raise RuntimeError("predict failed: %s" % Path(log_path).read_text(encoding="utf-8")[-1200:])
        produced = []
        for line in out.splitlines():
            if not line.strip():
                continue
            body = json.loads(line)["outputs"]
            if self.task == "classification":
                produced.append(max(body, key=lambda item: item["score"])["label"])
            else:
                produced.append(body["value"])
        if len(produced) != len(rows):
            raise RuntimeError("predict returned %d of %d rows" % (len(produced), len(rows)))
        return produced

    # ---------------------------------------------------------------- 帧

    @staticmethod
    def send(proc, frame):
        proc.stdin.write(json.dumps(frame, ensure_ascii=False) + "\n")
        proc.stdin.flush()

    def serve_upload(self, proc, frame):
        kind = frame["type"]
        if kind == "upload_begin_request":
            upload_id = "u-" + uuid.uuid4().hex[:12]
            target = self.uploads / upload_id
            target.mkdir()
            self.pending[upload_id] = {"dir": target, "kind": frame.get("kind", "candidate"),
                                       "files": frame["files"]}
            self.send(proc, {"type": "upload_begin_response", "id": frame["id"], "upload_id": upload_id})
            return
        if kind == "upload_chunk_request":
            entry = self.pending[frame["upload_id"]]
            path = entry["dir"] / frame["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as handle:
                handle.write(base64.b64decode(frame["data_base64"]))
            self.send(proc, {"type": "upload_chunk_response", "id": frame["id"],
                             "offset": path.stat().st_size})
            return
        if kind == "upload_finish_request":
            upload_id = frame["upload_id"]
            entry = self.pending.pop(upload_id)
            verified = {item["path"]: item for item in manifest(entry["dir"])}
            for expected in entry["files"]:
                found = verified.get(expected["path"])
                if found is None or found["sha256"] != expected["sha256"]:
                    raise RuntimeError("upload digest mismatch on %s" % expected["path"])
            self.finished[upload_id] = entry
            self.send(proc, {"type": "upload_finish_response", "id": frame["id"], "upload_id": upload_id})
            return
        raise RuntimeError("unknown upload frame %r" % kind)

    def decision(self, index):
        plan = {
            "A": {"epochs": 4, "learning_rate": 0.005, "batch_size": 32,
                  "reason": "小样本分类，小批量多轮次更稳"},
            "B": {"epochs": 14 if index <= 3 else 10, "learning_rate": 0.003, "batch_size": 128,
                  "reason": "回归误差仍在下降，保持中等步长"},
            "C": {"epochs": 12 if index <= 3 else 9, "learning_rate": 0.003, "batch_size": 256,
                  "reason": "大样本回归，步长不宜放大"},
        }[self.key]
        return "根据父模型指标与历史记录，本轮建议：\n" + json.dumps(plan, ensure_ascii=False)

    def loop(self, proc, index, rows):
        self.pending, self.finished = {}, {}
        candidate = None
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("round %d: host stream closed" % index)
            frame = json.loads(line)
            kind = frame.get("type", "")
            if kind.startswith("upload_"):
                self.serve_upload(proc, frame)
                continue
            if kind == "model_request":
                self.send(proc, {"type": "model_response", "id": frame["id"],
                                 "call_id": "call-%d-%d" % (index, len(self.records) + 1),
                                 "response": {"choices": [{"message": {"role": "assistant",
                                              "content": self.decision(index)}}]}})
                continue
            if kind == "history_request":
                self.send(proc, {"type": "history_response", "id": frame["id"],
                                 "records": [{"id": item["artifact_id"],
                                              "round_index": item["round_index"],
                                              "score": item["score"]} for item in self.records]})
                continue
            if kind == "candidate_request":
                entry = self.finished.pop(frame["upload_id"])
                artifact_id = "art-%s-r%02d" % (self.key.lower(), index)
                destination = self.artifacts / artifact_id
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.move(str(entry["dir"]), str(destination))
                descriptor = frame["artifact"]
                produced = self.predict(destination, rows)
                metrics = score(self.task, [row["answer"] for row in rows], produced)
                candidate = {"artifact_id": artifact_id, "round_index": index,
                             "metrics": metrics, "score": metrics[self.metric],
                             "receipt": receipt_of(artifact_id, destination, descriptor),
                             "dir": destination}
                self.send(proc, {"type": "candidate_response", "id": frame["id"],
                                 "artifact_id": artifact_id, "validation": metrics,
                                 "artifact_sha256": "sha256:local", "artifact": descriptor})
                continue
            if kind == "state_request":
                entry = self.finished.pop(frame["upload_id"])
                destination = self.states / ("r%02d" % index)
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.move(str(entry["dir"]), str(destination))
                files = sorted(destination.rglob("*"))
                digest = sha256_file(files[0]) if files else "sha256:empty"
                self.next_state = destination
                self.send(proc, {"type": "state_response", "id": frame["id"],
                                 "state_sha256": digest})
                continue
            if kind == "round_complete":
                record = frame["record"]
                if candidate is None:
                    raise RuntimeError("round %d finished without a candidate" % index)
                selected = record["selected_artifact_id"]
                if selected == candidate["artifact_id"]:
                    chosen_dir, chosen_receipt = candidate["dir"], candidate["receipt"]
                    chosen_score = candidate["score"]
                else:
                    if selected != self.parent_receipt["artifact_id"]:
                        raise RuntimeError("selected artifact %r is unknown" % selected)
                    chosen_dir, chosen_receipt = self.parent_dir, self.parent_receipt
                    chosen_score = next((item["score"] for item in self.records
                                         if item["artifact_id"] == selected), None)
                self.parent_dir, self.parent_receipt = chosen_dir, chosen_receipt
                self.parent_checkpoint_sha = next(item["sha256"] for item in chosen_receipt["files"]
                                                  if item["path"] == "checkpoint.npz")
                candidate.update(selected=selected, selected_score=chosen_score,
                                 reason=record["reason"], hypothesis=record["hypothesis"],
                                 call_ids=record["model_call_ids"],
                                 epochs=record["configuration"].get("epochs"),
                                 lr=record["configuration"].get("learning_rate"))
                self.send(proc, {"type": "round_ack", "accepted": True})
                return candidate

    # ---------------------------------------------------------------- 主流程

    def seed(self):
        source = ROOT / "_abc" / ("pkg_%s" % self.key.lower())
        artifact_id = "official-%s" % self.key.lower()
        destination = self.artifacts / artifact_id
        if destination.exists():
            shutil.rmtree(destination)
        (destination / "baselines").mkdir(parents=True)
        shutil.copyfile(source / "checkpoint.npz", destination / "checkpoint.npz")
        shutil.copyfile(source / "baselines" / "model.py", destination / "baselines" / "model.py")
        descriptor = {"checkpoint_paths": ["checkpoint.npz"],
                      "code_paths": ["baselines/model.py"],
                      "predict_argv": ["python3", "/app/harness.py", "predict"]}
        self.parent_dir = destination
        self.parent_receipt = receipt_of(artifact_id, destination, descriptor)
        self.parent_checkpoint_sha = next(item["sha256"] for item in self.parent_receipt["files"]
                                          if item["path"] == "checkpoint.npz")

    def run(self):
        self.seed()
        rows = load_rows(self.key)
        if self.args.eval_rows:
            rows = rows[: self.args.eval_rows]
        for index in range(1, self.args.rounds + 1):
            work = self.runs / ("r%02d" % index)
            if work.exists():
                shutil.rmtree(work)
            work.mkdir(parents=True)
            log_path = work / "stderr.log"
            proc = self.train_process(work, self.current_state, log_path)
            frame = {
                "type": "start", "protocol": PROTOCOL,
                "prompt": "%s 题第 %d 轮。只使用 /data 的公开投影，不要读取隐藏标签。" % (self.key, index),
                "prompt_version": PROMPT_VERSION,
                "dataset_key": self.key, "round_index": index,
                "task_contract": {"task": self.task, "metric": self.metric,
                                  "public_projection": {"train_path": "/data/train.jsonl",
                                                        "validation_path": "/data/validation.jsonl"}},
                "model_name": "local-decision-stub",
                "max_output_tokens": self.args.max_tokens,
                "wall_seconds": self.args.wall_seconds,
                "parent_checkpoint_sha256": self.parent_checkpoint_sha,
                "parent_artifact": self.parent_receipt,
                "artifacts": [item["receipt"] for item in self.records],
                "resource_limits": {"artifact_upload_protocol": "chunked-v1",
                                    "artifact_bytes": 512 * 1024 * 1024,
                                    "state_bytes": 512 * 1024 * 1024,
                                    "candidate_upload_bytes_per_round": 1024 * 1024 * 1024,
                                    "state_upload_bytes_per_round": 512 * 1024 * 1024,
                                    "upload_chunk_bytes": 1024 * 1024},
            }
            if self.current_state is not None:
                frame["state_path"] = "/state"
                frame["state_sha256"] = "sha256:local"
            self.send(proc, frame)
            started = time.time()
            try:
                result = self.loop(proc, index, rows)
            finally:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
                proc.wait(timeout=60)
                proc._log.close()
            if proc.returncode != 0:
                raise RuntimeError("round %d exit=%s\n%s" % (
                    index, proc.returncode, log_path.read_text(encoding="utf-8")[-2500:]))
            result["seconds"] = round(time.time() - started, 1)
            self.current_state = getattr(self, "next_state", None)
            self.records.append(result)
            print("round %02d  %s=%.6f  selected=%s  %.1fs" % (
                index, self.metric, result["score"], result["selected"][:20], result["seconds"]),
                flush=True)
        return self.records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["A", "B", "C"])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--image", default="zxj-sdk:1.0.0")
    parser.add_argument("--workdir", default=str(ROOT / "_sdk_local" / "runs"))
    parser.add_argument("--cpus", default="2")
    parser.add_argument("--memory", default="4g")
    parser.add_argument("--user", default="")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--wall-seconds", type=float, default=600.0)
    parser.add_argument("--eval-rows", type=int, default=0)
    parser.add_argument("--predict-timeout", type=int, default=1800)
    args = parser.parse_args()
    host = Host(args)
    records = host.run()
    out = Path(args.workdir) / ("summary_%s.json" % args.dataset)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
