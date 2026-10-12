from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import image_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--task1", required=True)
    args = parser.parse_args()
    task1 = Path(args.task1)
    with (task1 / "manifest.csv").open(encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    errors = []
    for image_path in image_paths(args.input):
        mask_path = task1 / "masks" / f"{image_path.stem}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if mask is None:
            errors.append(f"missing mask: {image_path}")
        elif mask.shape != image.shape[:2]:
            errors.append(f"shape mismatch: {image_path}")
        elif not set(np.unique(mask)).issubset({0, 255}):
            errors.append(f"non-binary mask: {mask_path}")
    if len(manifest) != len(image_paths(args.input)):
        errors.append(f"manifest rows {len(manifest)} != image count {len(image_paths(args.input))}")
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"Task 1 validation passed for {len(manifest)} images.")


if __name__ == "__main__":
    main()
