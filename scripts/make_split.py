from __future__ import annotations

import argparse
import json
import re
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import image_paths, load_yaml


def group_key(path: Path) -> str:
    stem = re.sub(r"_jpg\.rf\.[a-z0-9]+$", "", path.stem, flags=re.IGNORECASE)
    return stem


def dhash(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read {path}")
    resized = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    return resized[:, 1:] > resized[:, :-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/split.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    paths = image_paths(config["input_dir"])
    groups: dict[str, list[str]] = {}
    for path in paths:
        groups.setdefault(group_key(path), []).append(str(path))
    names = sorted(groups)
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    hashes = {name: dhash(Path(groups[name][0])) for name in names}
    threshold = int(config["near_duplicate_hamming"])
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if np.count_nonzero(hashes[left] != hashes[right]) <= threshold:
                union(left, right)
    merged: dict[str, list[str]] = {}
    for name in names:
        merged.setdefault(find(name), []).extend(groups[name])
    group_names = list(merged)
    random.Random(int(config["seed"])).shuffle(group_names)
    cut = round(len(group_names) * float(config["dev_fraction"]))
    dev = [item for name in group_names[:cut] for item in merged[name]]
    eval_paths = [item for name in group_names[cut:] for item in merged[name]]
    result = {"seed": config["seed"], "dev": dev, "eval": eval_paths, "groups": merged}
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "split.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {len(dev)} dev and {len(eval_paths)} eval images.")


if __name__ == "__main__":
    main()
