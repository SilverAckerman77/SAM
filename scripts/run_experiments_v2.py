from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classical_seg import segment
from src.metrics import binary_metrics
from src.utils import read_image


TRIALS = [
    ("otsu", "otsu", {}, "Grayscale intensity separates bright eggs from darker backgrounds.", None),
    ("hsv_value", "hsv_value", {}, "HSV brightness is robust to colour variation.", None),
    ("lab_lightness", "lab_lightness", {}, "LAB lightness reduces chroma sensitivity.", None),
    ("adaptive", "adaptive", {}, "Local thresholding handles uneven illumination.", None),
    ("hsv_saturation", "hsv_saturation", {}, "Eggs should be less saturated than many backgrounds.", None),
    ("hsv_value_inv", "hsv_value_inv", {}, "Inverted brightness may capture darker boundaries.", None),
    ("lab_a_inv", "lab_a_inv", {}, "Inverse LAB green-red contrast may separate neutral eggs.", None),
    ("lab_b", "lab_b", {}, "LAB blue-yellow contrast may separate pale eggs.", None),
    ("ycrcb_cr_inv", "ycrcb_cr_inv", {}, "Inverse red-difference may suppress skin/background.", None),
    ("ycrcb_cb", "ycrcb_cb", {}, "Blue-difference thresholding may separate pale eggs.", None),
    ("adaptive_inv", "adaptive_inv", {}, "Inverse local thresholding targets darker interiors.", None),
    ("bright", "bright", {"value_threshold": 150}, "Strict brightness tests whether highlights are sufficient.", None),
    ("bright_low_sat", "bright_low_sat", {"value_threshold": 90, "saturation_threshold": 120}, "Bright low-saturation pixels target pale eggs.", None),
    ("lab_lightness_open3", "lab_lightness_open3", {"morphology_kernel": 3}, "Smaller morphology may preserve small eggs.", "lab_lightness"),
    ("lab_lightness_open11", "lab_lightness_open11", {"morphology_kernel": 11}, "Larger morphology may remove background fragments.", "lab_lightness"),
    ("hsv_value_open3", "hsv_value_open3", {"morphology_kernel": 3}, "Smaller HSV morphology may preserve boundaries.", "hsv_value"),
    ("bright_low_sat_strict", "bright_low_sat_strict", {"value_threshold": 120, "saturation_threshold": 90}, "Stricter colour filtering may reduce false positives.", "bright_low_sat"),
    ("bright_low_sat_loose", "bright_low_sat_loose", {"value_threshold": 70, "saturation_threshold": 150}, "Looser colour filtering may improve recall.", "bright_low_sat"),
]


def resolve(path: str, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    split = json.loads((root / "outputs/split/split.json").read_text(encoding="utf-8"))
    task1 = root / "outputs/task1_gsam"
    output = root / "outputs/experiments_v2"
    output.mkdir(parents=True, exist_ok=True)

    dev_paths = [resolve(item, root) for item in split["dev"]]
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for trial_id, method, params, hypothesis, parent in TRIALS:
        per_image: list[dict[str, float]] = []
        for path in dev_paths:
            image = read_image(path)
            truth = cv2.imread(str(task1 / "masks" / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
            if truth is None:
                raise FileNotFoundError(f"Missing Task 1 mask for {path}")
            metrics = binary_metrics(truth, segment(image, method))
            per_image.append(metrics)
            if metrics["iou"] < 0.5:
                failures.append({"trial_id": trial_id, "filename": str(path), "iou": metrics["iou"], "reason": "iou_below_0.5"})
        records.append({
            "trial_id": trial_id,
            "method": method,
            "split": "dev",
            "hypothesis": hypothesis,
            "parameters_json": json.dumps(params, sort_keys=True),
            "parent_trial": parent or "",
            "change_rationale": "Baseline trial." if parent is None else f"Changed parameters from {parent}.",
            "iou": float(np.mean([item["iou"] for item in per_image])),
            "dice": float(np.mean([item["dice"] for item in per_image])),
            "precision": float(np.mean([item["precision"] for item in per_image])),
            "recall": float(np.mean([item["recall"] for item in per_image])),
            "failure_count": sum(item["iou"] < 0.5 for item in per_image),
            "decision": "pending",
            "decision_reason": "",
        })

    best = max(records, key=lambda item: float(item["iou"]))
    for record in records:
        record["decision"] = "kept_final" if record is best else "discarded"
        record["decision_reason"] = "highest development IoU" if record is best else f"lower development IoU than {best['trial_id']}"
    with (output / "experiments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output / "failures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trial_id", "filename", "iou", "reason"])
        writer.writeheader()
        writer.writerows(failures)
    (output / "frozen_config.json").write_text(
        json.dumps({"method": best["method"], "trial_id": best["trial_id"], "dev_metrics": best}, indent=2),
        encoding="utf-8",
    )

    dev_names = {Path(item).name for item in split["dev"]}
    eval_names = {Path(item).name for item in split["eval"]}
    dev_hashes = {file_hash(resolve(item, root)) for item in split["dev"]}
    eval_hashes = {file_hash(resolve(item, root)) for item in split["eval"]}
    leakage = {
        "seed": split.get("seed"),
        "dev_count": len(split["dev"]),
        "eval_count": len(split["eval"]),
        "filename_overlap": sorted(dev_names & eval_names),
        "content_hash_overlap": sorted(dev_hashes & eval_hashes),
        "passed": not (dev_names & eval_names or dev_hashes & eval_hashes),
    }
    (root / "outputs/split/leakage_check.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} trials; selected {best['trial_id']}; leakage passed={leakage['passed']}")


if __name__ == "__main__":
    main()
