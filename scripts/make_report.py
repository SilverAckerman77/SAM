from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_yaml


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
    eval_summary = Path("outputs/eval/summary.json")
    experiments = Path("outputs/experiments/experiments.csv")
    styles = getSampleStyleSheet()
    pdf = output / "report.pdf"
    story = [Paragraph(config["title"], styles["Title"]), Spacer(1, 0.2 * inch)]
    story.append(Paragraph("This report was generated from artifacts produced by the repository scripts.", styles["BodyText"]))
    story.append(Spacer(1, 0.12 * inch))
    if eval_summary.exists():
        summary = json.loads(eval_summary.read_text(encoding="utf-8"))
        data = [["Split", "N", "IoU", "Dice", "Precision", "Recall", "IoU<0.5"]]
        for name in ("dev", "eval"):
            item = summary[name]
            data.append([
                name, item["count"], f"{item['mean_iou']:.4f}", f"{item['mean_dice']:.4f}",
                f"{item['mean_precision']:.4f}", f"{item['mean_recall']:.4f}",
                item["failure_count_iou_below_0_5"],
            ])
        table = Table(data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dddddd")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ]))
        story.extend([Paragraph(f"Frozen method: {summary['method']}", styles["Heading2"]), table])
    if experiments.exists():
        lines = experiments.read_text(encoding="utf-8").splitlines()
        story.extend([Spacer(1, 0.2 * inch), Paragraph("Experiment log", styles["Heading2"])])
        story.append(Paragraph(f"{len(lines) - 1} development trials were executed. See {experiments} for the raw log.", styles["BodyText"]))
    story.extend([
        Spacer(1, 0.2 * inch),
        Paragraph("Limitations", styles["Heading2"]),
        Paragraph("SAM masks are pseudo-ground truth and include flagged large-area cases. Video tracking remains pending a supplied video.", styles["BodyText"]),
        Spacer(1, 0.2 * inch),
        Paragraph("AI tool disclosure", styles["Heading2"]),
        Paragraph("Edit report.md to document the candidate's actual decisions, review, execution, and verification.", styles["BodyText"]),
    ])
    SimpleDocTemplate(str(pdf), pagesize=letter, rightMargin=0.6 * inch, leftMargin=0.6 * inch).build(story)
    print(f"Generated {pdf} and copied report source to {destination}.")


if __name__ == "__main__":
    main()
