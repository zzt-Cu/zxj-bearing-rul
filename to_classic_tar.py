"""把 containerd/OCI 布局的 docker save 归档转成经典 Docker v2 布局。

背景：Docker 29 默认启用 containerd 镜像存储，`docker save` 产出 OCI 布局
（layer 是 blobs/sha256/<digest> 且为 gzip 压缩）。部分平台的校验器只认
经典布局（每个 layer 一个目录、内含未压缩 layer.tar）。本脚本产出后者的等价物，
两者都应能被 `docker load` 接受。

用法：python to_classic_tar.py 源.tar 目标.tar [仓库标签]
"""
import gzip
import json
import sys
import tarfile
import time

GZIP_MAGIC = b"\x1f\x8b"


def main(src, dst, tag=None):
    with tarfile.open(src) as archive:
        names = set(archive.getnames())
        if "manifest.json" not in names:
            raise SystemExit("源归档缺少 manifest.json，不是 docker save 产物")
        manifest = json.load(archive.extractfile("manifest.json"))
        entry = manifest[0]
        config_path = entry["Config"]
        layers = entry["Layers"]
        repo_tags = [tag] if tag else entry.get("RepoTags") or ["zxj-harness:latest"]
        config_bytes = archive.extractfile(config_path).read()
        config_name = config_path.rsplit("/", 1)[-1] + ".json"
        if not config_name.endswith(".json"):
            config_name += ".json"

        with tarfile.open(dst, "w") as out:
            def add_bytes(name, data):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = int(time.time())
                import io
                out.addfile(info, io.BytesIO(data))

            layer_names = []
            for index, path in enumerate(layers):
                raw = archive.extractfile(path).read()
                if raw[:2] == GZIP_MAGIC:
                    raw = gzip.decompress(raw)
                name = "layer%02d/layer.tar" % index
                add_bytes(name, raw)
                layer_names.append(name)
                print("  layer %02d <- %s (%d 字节)" % (index, path.rsplit("/", 1)[-1], len(raw)))

            add_bytes(config_name, config_bytes)
            classic = [{"Config": config_name, "RepoTags": repo_tags, "Layers": layer_names}]
            add_bytes("manifest.json", json.dumps(classic, indent=2).encode("utf-8"))
            add_bytes("repositories", json.dumps({repo_tags[0].split(":")[0]: {repo_tags[0].split(":")[1]: layer_names[-1].rsplit("/", 1)[0]}}, indent=2).encode("utf-8"))
    print("完成：%s（%d 层，标签 %s）" % (dst, len(layers), repo_tags))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
