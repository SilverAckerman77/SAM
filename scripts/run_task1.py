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
from src.grounded_sam import GroundedSam
from src.utils import image_paths, load_yaml, read_image, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task1_gsam.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/task1_gsam")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N images.")
    args = parser.parse_args()
    config = load_yaml(args.config)
    set_seed(int(config["seed"]))
    output = Path(args.output)
    masks_dir, instances_dir, overlays_dir = (
        output / "masks", output / "instance_masks", output / "overlays"
    )
    for directory in (masks_dir, instances_dir, overlays_dir):
        directory.mkdir(parents=True, exist_ok=True)
    segmenter = GroundedSam(config, device=args.device)
    paths = image_paths(args.input)
    if args.limit:
        paths = paths[:args.limit]
    manifest_rows: list[dict[str, object]] = []
    instance_rows: list[dict[str, object]] = []
    manifest_path = output / "manifest.csv"
    existing: dict[str, dict[str, str]] = {}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            existing = {row["filename"]: row for row in csv.DictReader(handle)}
    start = time.perf_counter()
    for path in tqdm(paths, desc="Grounded-SAM"):
        image = read_image(path)
        stem = path.stem
        mask_path = masks_dir / f"{stem}.png"
        if mask_path.exists() and str(path) in existing:
            manifest_rows.append(existing[str(path)])
            continue
        instances = segmenter.segment(image)
        combined = np.zeros(image.shape[:2], dtype=np.uint8)
        for item in instances:
            instance_mask = item["mask"] * 255
            combined = np.maximum(combined, instance_mask)
            instance_path = instances_dir / f"{stem}_egg{item['instance_id']}.png"
            cv2.imwrite(str(instance_path), instance_mask)
            instance_rows.append({
                "image": str(path),
                "instance_id": item["instance_id"],
                "box": json.dumps(item["box"]),
                "dino_score": item["dino_score"],
                "sam_score": item["sam_score"],
                "area_fraction": item["area_fraction"],
            })
        overlay = image.copy()
        overlay[combined > 0] = (
            0.55 * overlay[combined > 0] + np.array([0, 0, 255]) * 0.45
        ).astype(np.uint8)
        cv2.imwrite(str(mask_path), combined)
        cv2.imwrite(str(overlays_dir / f"{stem}.png"), overlay)
        area_fraction = float(np.count_nonzero(combined) / combined.size)
        manifest_rows.append({
            "filename": str(path),
            "num_eggs": len(instances),
            "area_fraction": area_fraction,
            "reference_ok": bool(instances),
            "notes": "" if instances else "no GroundingDINO detections",
        })
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "num_eggs", "area_fraction", "reference_ok", "notes"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (output / "instances.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "instance_id", "box", "dino_score", "sam_score", "area_fraction"],
        )
        writer.writeheader()
        writer.writerows(instance_rows)
    (output / "run_info.json").write_text(json.dumps({
        "detector_model": config["detector_model"],
        "sam_model_type": config["sam_model_type"],
        "sam_checkpoint": config["sam_checkpoint"],
        "prompt": config["text_prompt"],
        "device": segmenter.device,
        "seed": config["seed"],
        "runtime_seconds": time.perf_counter() - start,
        "filter_settings": {
            key: config[key]
            for key in ("box_threshold", "text_threshold", "nms_iou",
                        "min_area_fraction", "max_area_fraction")
        },
        "postprocess": ["largest_connected_component", "fill_holes"],
        "images_processed": len(paths),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
