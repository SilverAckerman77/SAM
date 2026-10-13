from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def box_iou(left: list[float], right: list[float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    x1, y1 = max(lx1, rx1), max(ly1, ry1)
    x2, y2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def nms(boxes: list[list[float]], scores: list[float], threshold: float) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda index: scores[index], reverse=True)
    kept: list[int] = []
    while order:
        current = order.pop(0)
        kept.append(current)
        order = [index for index in order if box_iou(boxes[current], boxes[index]) <= threshold]
    return kept


def largest_component_and_fill_holes(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = np.where(labels == largest, 255, 0).astype(np.uint8)
    flood = component.copy()
    flood_mask = np.zeros((component.shape[0] + 2, component.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return ((component > 0) | (holes > 0)).astype(np.uint8)


class GroundedSam:
    def __init__(self, config: dict[str, Any], device: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise RuntimeError(
                "Grounded-SAM requires transformers, torch, and segment-anything."
            ) from exc
        selected_device = device
        if selected_device == "auto":
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        if selected_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        checkpoint = Path(config["sam_checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
        self.device = selected_device
        self.config = config
        self.processor = AutoProcessor.from_pretrained(config["detector_model"])
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            config["detector_model"]
        ).to(selected_device)
        sam = sam_model_registry[config["sam_model_type"]](checkpoint=str(checkpoint))
        sam.to(device=selected_device)
        self.predictor = SamPredictor(sam)

    def detect_boxes(self, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        import torch
        image = self.processor(
            images=image_rgb,
            text=self.config["text_prompt"],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.detector(**image)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=float(self.config["box_threshold"]),
            text_threshold=float(self.config["text_threshold"]),
            target_sizes=[image_rgb.shape[:2]],
        )[0]
        boxes = result["boxes"].detach().cpu().tolist()
        scores = result["scores"].detach().cpu().tolist()
        height, width = image_rgb.shape[:2]
        valid_boxes: list[list[float]] = []
        valid_scores: list[float] = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            area_fraction = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)
            if float(self.config["min_area_fraction"]) <= area_fraction <= float(
                self.config["max_area_fraction"]
            ):
                valid_boxes.append([float(x1), float(y1), float(x2), float(y2)])
                valid_scores.append(float(score))
        return [
            {"box": valid_boxes[index], "dino_score": valid_scores[index]}
            for index in nms(valid_boxes, valid_scores, float(self.config["nms_iou"]))
        ]

    def segment(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        detections = self.detect_boxes(image_rgb)
        self.predictor.set_image(image_rgb)
        results = []
        height, width = image_bgr.shape[:2]
        for instance_id, detection in enumerate(detections):
            masks, scores, _ = self.predictor.predict(
                box=np.asarray(detection["box"], dtype=np.float32),
                multimask_output=True,
            )
            best = int(np.argmax(scores))
            mask = largest_component_and_fill_holes(masks[best])
            results.append({
                "instance_id": instance_id,
                "box": detection["box"],
                "dino_score": detection["dino_score"],
                "sam_score": float(scores[best]),
                "mask": mask,
                "area_fraction": float(mask.sum() / (height * width)),
            })
        return results
