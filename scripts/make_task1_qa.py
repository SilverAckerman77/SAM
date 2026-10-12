from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import image_paths


def tile(path: Path, label: str, size: int = 220) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size - 8, size - 32))
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, 4))
    ImageDraw.Draw(canvas).text((4, size - 24), label[:32], fill="black")
    return canvas


def make_sheet(paths: list[Path], destination: Path, title: str) -> None:
    columns = 5
    rows = max(1, (len(paths) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * 220, rows * 220 + 28), "#dddddd")
    ImageDraw.Draw(sheet).text((6, 6), title, fill="black")
    for index, path in enumerate(paths):
        sheet.paste(tile(path, path.stem), ((index % columns) * 220, 28 + (index // columns) * 220))
    sheet.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--task1", required=True)
    parser.add_argument("--output", default="outputs/task1/qa")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    task1 = Path(args.task1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (task1 / "manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    flagged = [row for row in rows if row["flagged"].lower() == "true"]
    unflagged = [row for row in rows if row["flagged"].lower() != "true"]
    random.Random(args.seed).shuffle(unflagged)
    def overlay_path(row: dict[str, str]) -> Path:
        return task1 / "overlays" / f"{Path(row['filename']).stem}.png"

    flagged_paths = [overlay_path(row) for row in flagged]
    unflagged_paths = [overlay_path(row) for row in unflagged[:20]]
    make_sheet(flagged_paths, output / "flagged_overlays.jpg", "Flagged SAM overlays")
    make_sheet(unflagged_paths, output / "unflagged_sample_overlays.jpg", "Random unflagged SAM overlays")
    (output / "qa_summary.txt").write_text(
        f"flagged={len(flagged)}\nunflagged_sample={len(unflagged_paths)}\n"
        f"flag_reasons={sorted({row['flag_reason'] for row in flagged})}\n",
        encoding="utf-8",
    )
    print(f"Wrote QA sheets to {output}")


if __name__ == "__main__":
    main()
