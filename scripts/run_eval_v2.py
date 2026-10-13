from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classical_seg import segment
from src.metrics import binary_metrics
from src.utils import read_image


def boundary(mask: np.ndarray) -> np.ndarray:
    value = (mask > 0).astype(np.uint8)
    return cv2.morphologyEx(value, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)


def boundary_f1(target: np.ndarray, prediction: np.ndarray, tolerance: int = 2) -> float:
    target_edge = boundary(target)
    prediction_edge = boundary(prediction)
    if not target_edge.any() and not prediction_edge.any():
        return 1.0
    if not target_edge.any() or not prediction_edge.any():
        return 0.0
    target_distance = cv2.distanceTransform((~target_edge).astype(np.uint8), cv2.DIST_L2, 3)
    prediction_distance = cv2.distanceTransform((~prediction_edge).astype(np.uint8), cv2.DIST_L2, 3)
    precision = float((target_distance[prediction_edge] <= tolerance).mean())
    recall = float((prediction_distance[target_edge] <= tolerance).mean())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def hd95(target: np.ndarray, prediction: np.ndarray) -> float:
    target_edge = boundary(target)
    prediction_edge = boundary(prediction)
    if not target_edge.any() or not prediction_edge.any():
        return float("inf")
    target_distance = cv2.distanceTransform((~target_edge).astype(np.uint8), cv2.DIST_L2, 3)
    prediction_distance = cv2.distanceTransform((~prediction_edge).astype(np.uint8), cv2.DIST_L2, 3)
    distances = np.concatenate([target_distance[prediction_edge], prediction_distance[target_edge]])
    return float(np.percentile(distances, 95))


def make_worst_sheet(rows: list[dict[str, object]], output: Path) -> None:
    width, height = 270, 270
    sheet = Image.new("RGB", (width * 3, height * len(rows)), "white")
    for index, row in enumerate(rows):
        image = cv2.imread(str(row["image"]), cv2.IMREAD_COLOR)
        truth = cv2.imread(str(row["truth"]), cv2.IMREAD_GRAYSCALE)
        prediction = cv2.imread(str(row["prediction"]), cv2.IMREAD_GRAYSCALE)
        panels = [image, truth, prediction]
        for column, panel in enumerate(panels):
            if panel.ndim == 2:
                panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2RGB)
            else:
                panel = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(panel)
            pil.thumbnail((width - 8, height - 30))
            canvas = Image.new("RGB", (width, height), "white")
            canvas.paste(pil, ((width - pil.width) // 2, 4))
            ImageDraw.Draw(canvas).text((4, height - 22), ["image", "target", "prediction"][column], fill="black")
            sheet.paste(canvas, (column * width, index * height))
    sheet.save(output)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    split = json.loads((root / "outputs/split/split.json").read_text(encoding="utf-8"))
    frozen = json.loads((root / "outputs/experiments_v2/frozen_config.json").read_text(encoding="utf-8"))
    method = frozen["method"]
    output = root / "outputs/eval_v2"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for split_name in ("dev", "eval"):
        for filename in split[split_name]:
            image_path = root / filename
            truth_path = root / "outputs/task1_gsam/masks" / f"{image_path.stem}.png"
            truth = cv2.imread(str(truth_path), cv2.IMREAD_GRAYSCALE)
            prediction = segment(read_image(image_path), method)
            metrics = binary_metrics(truth, prediction)
            rows.append({
                "split": split_name,
                "filename": filename,
                "method": method,
                **metrics,
                "boundary_f1": boundary_f1(truth, prediction),
                "hd95_pixels": hd95(truth, prediction),
            })
            cv2.imwrite(str(output / f"{image_path.stem}.png"), prediction)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, object] = {"method": method, "source": "outputs/experiments_v2/frozen_config.json"}
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
            "mean_boundary_f1": float(np.mean([row["boundary_f1"] for row in values])),
            "mean_hd95_pixels": float(np.mean([row["hd95_pixels"] for row in values if np.isfinite(row["hd95_pixels"])])),
            "failure_count_iou_below_0_5": sum(float(row["iou"]) < 0.5 for row in values),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    worst = sorted([row for row in rows if row["split"] == "eval"], key=lambda row: float(row["iou"]))[:8]
    with (output / "worst_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(worst[0]))
        writer.writeheader()
        writer.writerows(worst)
    for row in worst:
        image_path = root / row["filename"]
        row["image"] = str(image_path)
        row["truth"] = str(root / "outputs/task1_gsam/masks" / f"{image_path.stem}.png")
        row["prediction"] = str(output / f"{image_path.stem}.png")
    make_worst_sheet(worst, output / "worst_cases.jpg")
    print(f"Wrote {len(rows)} frozen evaluation rows using {method}.")


if __name__ == "__main__":
    main()
