from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sam_segmenter import SamSegmenter
from src.utils import image_paths, load_yaml, read_image, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task1.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/task1")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    config = load_yaml(args.config)
    set_seed(int(config["seed"]))
    output = Path(args.output)
    masks_dir, instances_dir, overlays_dir = output / "masks", output / "instance_masks", output / "overlays"
    for directory in (masks_dir, instances_dir, overlays_dir):
        directory.mkdir(parents=True, exist_ok=True)
    segmenter = SamSegmenter(config, device=args.device)
    rows = []
    existing = {}
    manifest_path = output / "manifest.csv"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            existing = {row["filename"]: row for row in csv.DictReader(handle)}
    start = time.perf_counter()
    for path in tqdm(image_paths(args.input), desc="SAM"):
        image = read_image(path)
        stem = path.stem
        mask_path = masks_dir / f"{stem}.png"
        if mask_path.exists():
            if str(path) in existing:
                rows.append(existing[str(path)])
            continue
        instances = segmenter.segment(image)
        combined = np.zeros(image.shape[:2], dtype=np.uint8)
        for index, item in enumerate(instances):
            instance = item["mask"] * 255
            combined = np.maximum(combined, instance)
            cv2.imwrite(str(instances_dir / f"{stem}_egg{index}.png"), instance)
        overlay = image.copy()
        overlay[combined > 0] = (0.55 * overlay[combined > 0] + np.array([0, 0, 255]) * 0.45).astype(np.uint8)
        cv2.imwrite(str(mask_path), combined)
        cv2.imwrite(str(overlays_dir / f"{stem}.jpg"), overlay)
        area = int(np.count_nonzero(combined))
        rows.append({
            "filename": str(path),
            "num_eggs": len(instances),
            "prompt_strategy": config["prompt_strategy"],
            "sam_score": max((item["score"] for item in instances), default=0.0),
            "mask_area_px": area,
            "area_fraction": area / (image.shape[0] * image.shape[1]),
            "flagged": not bool(instances),
            "flag_reason": "no_filtered_mask" if not instances else "",
        })
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "filename", "num_eggs", "prompt_strategy", "sam_score",
            "mask_area_px", "area_fraction", "flagged", "flag_reason",
        ])
        writer.writeheader()
        writer.writerows(rows)
    (output / "run_info.json").write_text(json.dumps({
        "model": config["model_type"], "checkpoint": config["checkpoint"],
        "device_requested": args.device, "seed": config["seed"],
        "runtime_seconds": time.perf_counter() - start, "config": config,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
