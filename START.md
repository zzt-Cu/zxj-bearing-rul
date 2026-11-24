# cpu-b-rul-20261110-v1

CPU development baseline; not an approved competition scoring baseline.

Read [task and label definitions](TASK.md) and [continuation guide](TRAINING_GUIDE.md).

Install with `python -m pip install -r baselines/requirements.txt`.

Continue the supplied weights and Adam state:

```sh
python -m baselines.train --profile B --train public/train.jsonl --validation public/validation.jsonl --from-baseline checkpoint.npz --output round-1
```

B predicts remaining minutes to the last observed lifecycle snapshot. Targets are calculated from the published run-to-failure records, not official per-snapshot health classes. Evaluate RMSE and MAE in minutes; lower is better.

You may modify the provided training code and settings. Keep validation separate from training. This archive contains only public train/validation data; hidden evaluation stays with the platform.
