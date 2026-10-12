from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classical_seg import segment
from src.metrics import binary_metrics
from src.utils import load_yaml, read_image


def evaluate(paths: list[str], task1: Path, method: str) -> dict:
    values = []
    for item in paths:
        path = Path(item)
        image = read_image(path)
        truth = cv2.imread(str(task1 / "masks" / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if truth is None:
            raise FileNotFoundError(f"Missing Task 1 mask for {path}")
        values.append(binary_metrics(truth, segment(image, method)))
    return {
        "iou": float(np.mean([x["iou"] for x in values])),
        "dice": float(np.mean([x["dice"] for x in values])),
        "precision": float(np.mean([x["precision"] for x in values])),
        "recall": float(np.mean([x["recall"] for x in values])),
        "failure_count": sum(x["iou"] < 0.5 for x in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    import json
    split = json.loads(Path(config["split_file"]).read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    hypotheses = {
        "otsu": "Grayscale intensity separates bright eggs from darker backgrounds.",
        "hsv_value": "HSV brightness is more robust than grayscale under colour variation.",
        "lab_lightness": "LAB lightness reduces sensitivity to chroma differences.",
        "adaptive": "Local thresholding handles uneven illumination.",
        "hsv_saturation": "Eggs should be less saturated than many backgrounds.",
        "hsv_value_inv": "Inverting brightness may capture darker egg boundaries.",
        "lab_a_inv": "The inverse LAB green-red channel may separate neutral eggs.",
        "lab_b": "LAB blue-yellow contrast may separate eggs from backgrounds.",
        "ycrcb_cr_inv": "The inverse red-difference channel may suppress skin/background.",
        "ycrcb_cb": "Blue-difference thresholding may separate pale eggs.",
        "adaptive_inv": "Inverse local thresholding targets darker egg interiors.",
        "bright": "A strict brightness threshold tests whether highlights alone are sufficient.",
        "bright_low_sat": "Bright, low-saturation pixels should target pale eggs while rejecting colour.",
    }
    rows = []
    for method in config["trials"]:
        result = evaluate(split["dev"], Path(config["task1_dir"]), method)
        rows.append({
            "trial_id": method,
            "method": method,
            "split": "dev",
            "hypothesis": hypotheses[method],
            **result,
            "decision": "pending",
        })
    best = max(rows, key=lambda row: row["iou"])
    for row in rows:
        row["decision"] = "kept_final" if row is best else "discarded"
        row["decision_reason"] = "highest dev IoU" if row is best else f"lower dev IoU than {best['method']}"
    with (output / "experiments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "frozen_config.json").write_text(
        json.dumps({"method": best["method"], "dev_metrics": best}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} experiment results to {output / 'experiments.csv'}")


if __name__ == "__main__":
    main()
