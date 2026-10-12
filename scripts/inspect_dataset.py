from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import image_paths, write_json


def make_contact_sheet(paths: list[Path], destination: Path, columns: int = 5) -> None:
    thumbnails = []
    for path in paths[:25]:
        image = Image.open(path).convert("RGB")
        image.thumbnail((180, 180))
        canvas = Image.new("RGB", (200, 220), "white")
        canvas.paste(image, ((200 - image.width) // 2, 4))
        ImageDraw.Draw(canvas).text((4, 188), path.name[:28], fill="black")
        thumbnails.append(canvas)
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 200, max(rows, 1) * 220), "#dddddd")
    for index, thumbnail in enumerate(thumbnails):
        sheet.paste(thumbnail, ((index % columns) * 200, (index // columns) * 220))
    sheet.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/exploration")
    args = parser.parse_args()
    paths = image_paths(args.input)
    rows = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        rows.append({
            "filename": str(path),
            "width": image.shape[1],
            "height": image.shape[0],
            "channels": image.shape[2] if image.ndim == 3 else 1,
            "suffix": path.suffix.lower(),
            "roboflow_augmented": ".rf." in path.name,
        })
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "dataset_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["filename"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "summary.json", {"image_count": len(rows)})
    make_contact_sheet(paths, output / "contact_sheet.jpg")
    print(f"Inspected {len(rows)} images; inventory: {output / 'dataset_inventory.csv'}")


if __name__ == "__main__":
    main()
