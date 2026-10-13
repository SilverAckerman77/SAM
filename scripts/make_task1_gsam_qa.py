from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import image_paths


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape, dtype=np.uint8)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(shape, dtype=np.uint8)
    return (mask > 0).astype(np.uint8)


def panel(image: np.ndarray, label: str, size: int = 300) -> Image.Image:
    if image.ndim == 2:
        image = np.where(image > 0, 255, 0).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image)
    pil.thumbnail((size - 12, size - 42))
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(pil, ((size - pil.width) // 2, 4))
    ImageDraw.Draw(canvas).text((6, size - 30), label[:48], fill="black")
    return canvas


def make_sheet(rows: list[dict[str, str]], destination: Path, title: str) -> None:
    columns = 3
    panel_size = 300
    row_height = 330
    sheet = Image.new("RGB", (columns * panel_size, max(1, len(rows)) * row_height + 36), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black")
    for row_index, row in enumerate(rows):
        y = 36 + row_index * row_height
        source = cv2.imread(row["source"], cv2.IMREAD_COLOR)
        shape = source.shape[:2]
        old = load_mask(Path(row["old_mask"]), shape)
        new = load_mask(Path(row["new_mask"]), shape)
        panels = [
            panel(source, f"original: {row['name']}"),
            panel(old, f"old mask; area={row['old_area']}"),
            panel(new, f"new mask; eggs={row['num_eggs']} area={row['new_area']}"),
        ]
        for column, image in enumerate(panels):
            sheet.paste(image, (column * panel_size, y))
    sheet.save(destination)


def iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--old", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-sheet", type=int, default=8)
    args = parser.parse_args()

    source_by_name = {path.name: path for path in image_paths(args.input)}
    new_root = Path(args.new)
    old_root = Path(args.old)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (new_root / "manifest.csv").open(encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))

    rows: list[dict[str, str]] = []
    for item in manifest:
        name = Path(item["filename"]).name
        source = source_by_name.get(name)
        if source is None:
            continue
        shape = cv2.imread(str(source), cv2.IMREAD_COLOR).shape[:2]
        old_path = old_root / "masks" / f"{source.stem}.png"
        new_path = new_root / "masks" / f"{source.stem}.png"
        old_mask = load_mask(old_path, shape)
        new_mask = load_mask(new_path, shape)
        rows.append(
            {
                "name": name,
                "source": str(source),
                "old_mask": str(old_path),
                "new_mask": str(new_path),
                "num_eggs": item["num_eggs"],
                "new_area": item["area_fraction"],
                "old_area": f"{old_mask.mean():.6f}",
                "mask_iou": f"{iou(old_mask, new_mask):.6f}",
                "reference_ok": item["reference_ok"],
            }
        )

    rows.sort(key=lambda row: (-float(row["new_area"]), -float(row["old_area"]), row["name"]))
    with (output / "old_new_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_sheet = max(1, args.per_sheet)
    for start in range(0, len(rows), per_sheet):
        batch = rows[start : start + per_sheet]
        make_sheet(batch, output / f"review_{start // per_sheet + 1:02d}.jpg",
                   f"Grounded-SAM review {start + 1}-{start + len(batch)} (worst first)")

    top_changes = sorted(rows, key=lambda row: float(row["mask_iou"]))[:per_sheet]
    make_sheet(top_changes, output / "comparison_low_iou.jpg", "Old/new comparison: lowest mask IoU")
    false_rows = [row for row in rows if row["reference_ok"].lower() == "false"]
    (output / "qa_summary.txt").write_text(
        f"source_images={len(source_by_name)}\n"
        f"manifest_rows={len(manifest)}\n"
        f"matched_rows={len(rows)}\n"
        f"reference_ok_false={len(false_rows)}\n"
        f"review_sheets={(len(rows) + per_sheet - 1) // per_sheet}\n"
        f"comparison_rows={len(rows)}\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} review rows to {output}")


if __name__ == "__main__":
    main()
