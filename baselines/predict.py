"""JSONL prediction adapter. Reads inputs, never labels; no model execution hooks."""
import argparse
import json
import sys

try:
    from .model import Model
    from .data import flatten, json_value
except ImportError:
    from model import Model
    from data import flatten, json_value
import numpy as np


def predict(model, inputs):
    paths, values = flatten(inputs)
    if paths != model.meta["paths"]:
        raise ValueError("prediction feature schema differs from checkpoint")
    output = model.predict(np.asarray([values], dtype=np.float64))[0]
    if model.meta["task"] == "regression":
        return {"outputs": {"value": float(output)}}
    return {"outputs": [{"label": label, "score": float(score)} for label, score in zip(model.meta["classes"], output)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    model = Model.load(args.checkpoint)
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json_value(line)
        if not isinstance(request, dict) or set(request) - {"task", "inputs"} or "inputs" not in request:
            raise ValueError("prediction request requires only task and inputs")
        if request.get("task", model.meta["task"]) != model.meta["task"]:
            raise ValueError("prediction task mismatch")
        print(json.dumps(predict(model, request["inputs"]), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
