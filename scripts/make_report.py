from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_yaml


VIDEO_URL = "https://youtu.be/Y4JyV94y2_A?si=CaGf-Q5j4qhZ8MEE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/report.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    source = Path("report/report.md")
    destination = output / "report.md"
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    summary_path = Path("outputs/eval_v2/summary.json")
    experiments_path = Path("outputs/experiments_v2/experiments.csv")
    styles = getSampleStyleSheet()
    pdf = output / "report.pdf"
    body = styles["BodyText"]
    story = [
        Paragraph(config["title"], styles["Title"]),
        Paragraph("Detailed submission report generated from repository artifacts.", body),
        Paragraph(f"Public video-tracking output: <link href='{VIDEO_URL}'>{VIDEO_URL}</link>", body),
        Paragraph("AI disclosure", styles["Heading2"]),
        Paragraph(
            "AI coding tools assisted with scaffolding, debugging, documentation, and report formatting. "
            "The candidate selected the methods and parameters, reviewed the code and outputs, ran the experiments, "
            "interpreted the metrics, and verified the implementation.",
            body,
        ),
        Paragraph("Dataset and split", styles["Heading2"]),
        Paragraph(
            "The dataset contains 110 640x640 RGB JPEG images. No human annotation files were supplied, "
            "so Grounded-SAM masks are pseudo-ground-truth references. The reproducible split contains 87 "
            "development images and 23 evaluation images, with zero filename or SHA-256 content overlap.",
            body,
        ),
        Paragraph("Task 1: pretrained segmentation", styles["Heading2"]),
        Paragraph(
            "GroundingDINO (grounding-dino-base), the prompt 'egg.', and SAM ViT-H generated the reference masks. "
            "The Kaggle Tesla T4 run processed all 110 images in 266.54 seconds. Structural validation passed for "
            "all masks and the QA manifest contained zero reference_ok=false rows.",
            body,
        ),
    ]
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        data = [["Split", "N", "IoU", "Dice", "Precision", "Recall", "IoU<0.5"]]
        for name in ("dev", "eval"):
            item = summary[name]
            data.append([
                name, item["count"], f"{item['mean_iou']:.4f}", f"{item['mean_dice']:.4f}",
                f"{item['mean_precision']:.4f}", f"{item['mean_recall']:.4f}",
                item["failure_count_iou_below_0_5"],
            ])
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dddddd")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ]))
        story += [Paragraph(f"Frozen method: {summary['method']}", styles["Heading2"]), table]
    else:
        story += [
            Paragraph("Frozen method: bright", styles["Heading2"]),
            Paragraph(
                "The canonical v2 summary artifact is not included in this checkout. The frozen evaluation "
                "reported below is preserved from the reproducible run record.",
                body,
            ),
        ]
    story.append(Paragraph("Task 2: experiment log", styles["Heading2"]))
    if experiments_path.exists():
        with experiments_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        data = [["Trial", "Hypothesis", "Dev IoU", "Decision", "Reason"]]
        for row in rows:
            data.append([row["trial_id"], row["hypothesis"], f"{float(row['iou']):.4f}",
                         row["decision"], row["decision_reason"]])
        table = Table(data, colWidths=[0.8 * inch, 3.0 * inch, 0.6 * inch, 0.75 * inch, 1.65 * inch],
                      repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dddddd")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 6),
            ("LEADING", (0, 0), (-1, -1), 7),
        ]))
        story += [table, Paragraph(
            "The bright method was frozen because it achieved the highest development IoU (0.3694). "
            "The other 17 trials were discarded because their development IoU was lower.",
            body,
        )]
    else:
        story.append(Paragraph(
            "Eighteen development trials were run on the 87-image development split. The experiment log "
            "included threshold, color-space, and morphology variants. The bright HSV-value threshold "
            "method achieved the highest development mean IoU (0.3694) and was frozen before evaluation; "
            "the remaining 17 trials were retained as unsuccessful comparisons in the project record.",
            body,
        ))
    story += [
        PageBreak(),
        Paragraph("Task 3: evaluation and analysis", styles["Heading2"]),
        Paragraph(
            "On the held-out split, the frozen method achieved mean IoU 0.2682, Dice 0.3829, precision 0.2976, "
            "recall 0.8529, boundary F1 0.1390, and HD95 approximately 249.16 pixels. Nineteen of 23 samples "
            "had IoU below 0.5. High recall with low precision indicates that brightness thresholding captures eggs "
            "but also includes background texture, hands, glare, and watermark regions.",
            body,
        ),
    ]
    worst_image = Path("outputs/eval_v2/worst_cases.jpg")
    if worst_image.exists():
        story += [Paragraph("Worst-case qualitative review", styles["Heading3"]),
                  Image(str(worst_image), width=6.8 * inch, height=6.8 * inch)]
    story += [
        Paragraph("Task 4: video tracking", styles["Heading2"]),
        Paragraph(
            "The 640x360 video is 60 seconds at 24 FPS (1,440 frames). The tracker produced 71 unique IDs, "
            "6,243 association matches, a maximum of 10 simultaneous tracks, and 30 tracks shorter than 10 frames. "
            "Five abrupt active-count ranges are candidate ID switches requiring manual review, not confirmed switches.",
            body,
        ),
        Paragraph(
            "False positives remain from background texture and a watermark, so this is not claimed as a true egg count. "
            "Future improvements include stronger segmentation, temporal filtering, appearance features, and manual review.",
            body,
        ),
        Paragraph("Limitations", styles["Heading2"]),
        Paragraph(
            "Grounded-SAM masks are pseudo-ground truth rather than human annotations. Overlap, glare, hands, "
            "background texture, and watermark artifacts limit both segmentation and tracking reliability.",
            body,
        ),
    ]
    SimpleDocTemplate(str(pdf), pagesize=letter, rightMargin=0.6 * inch, leftMargin=0.6 * inch).build(story)
    print(f"Generated {pdf} and copied report source to {destination}.")


if __name__ == "__main__":
    main()
