"""按文件指纹缓存 JSONL 解析结果。

C 的训练集 11.3 万行、验证集 2.3 万行，每读一遍约 10 秒与 3 秒。而一轮里
observer 要读训练集、evaluator 要读验证集，六轮就白花 80 秒——这些时间不产生
任何新信息，纯粹是重复解析。

缓存键带 `(mtime_ns, size)`：数据文件一变就自动失效，不会读到旧内容。
只缓存解析结果，不缓存任何指标——指标每轮都要按新的检查点重算，这是评估独立性的前提。
"""
import os

from baselines.data import read_dataset

_CACHE = {}


def read(path, task, classes=None, expected_paths=None):
    try:
        stat = os.stat(path)
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return read_dataset(path, task, classes, expected_paths)

    key = (str(path), task, tuple(classes or ()), stamp,
           None if expected_paths is None else tuple(expected_paths))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    data = read_dataset(path, task, classes, expected_paths)
    _CACHE[key] = data
    return data


def clear():
    _CACHE.clear()
