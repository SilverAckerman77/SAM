from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classical_seg import segment
from src.metrics import binary_metrics
from src.utils import load_yaml, read_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    split = json.loads(Path(config["split_file"]).read_text(encoding="utf-8"))
    frozen = json.loads((Path(config["output_dir"]).parent / "experiments" / "frozen_config.json").read_text(encoding="utf-8"))
    method = frozen["method"]
    rows = []
    for split_name in ("dev", "eval"):
        for filename in split[split_name]:
            path = Path(filename)
            truth = cv2.imread(str(Path(config["task1_dir"]) / "masks" / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
            if truth is None:
                raise FileNotFoundError(f"Missing Task 1 mask for {path}")
            rows.append({"split": split_name, "filename": filename, "method": method, **binary_metrics(truth, segment(read_image(path), method))})
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for split_name in ("dev", "eval"):
        values = [row for row in rows if row["split"] == split_name]
        summary[split_name] = {
            "count": len(values),
            "mean_iou": float(np.mean([row["iou"] for row in values])),
            "median_iou": float(np.median([row["iou"] for row in values])),
            "std_iou": float(np.std([row["iou"] for row in values])),
            "mean_dice": float(np.mean([row["dice"] for row in values])),
            "mean_precision": float(np.mean([row["precision"] for row in values])),
            "mean_recall": float(np.mean([row["recall"] for row in values])),
            "failure_count_iou_below_0_5": sum(float(row["iou"]) < 0.5 for row in values),
        }
    (output / "summary.json").write_text(json.dumps({"method": method, **summary}, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} evaluation rows.")


if __name__ == "__main__":
    main()
