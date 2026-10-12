from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


class SamSegmenter:
    """Load SAM once and reuse it for every image in a run."""

    def __init__(self, config: dict, device: str = "auto") -> None:
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as exc:
            raise RuntimeError("Install segment-anything before running Task 1.") from exc
        checkpoint = Path(config["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
        import torch
        selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if selected_device == "auto":
            selected_device = "cpu"
        if selected_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        model = sam_model_registry[config["model_type"]](checkpoint=str(checkpoint))
        model.to(device=selected_device)
        self.generator = SamAutomaticMaskGenerator(model)
        self.config = config
        self.device = selected_device

    def segment(self, image: np.ndarray) -> list[dict]:
        """Run automatic masks and apply simple geometry filters."""
        config = self.config
        height, width = image.shape[:2]
        candidates = []
        for item in self.generator.generate(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)):
            area_fraction = item["area"] / float(height * width)
            if not (config["min_area_fraction"] <= area_fraction <= config["max_area_fraction"]):
                continue
            mask = item["segmentation"].astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = cv2.contourArea(contour) / hull_area if hull_area else 0.0
            if solidity < config["min_solidity"]:
                continue
            candidates.append({"mask": mask, "score": float(item["predicted_iou"]), "area": int(item["area"])})
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[: int(config["max_instances"])]


def segment_image(image: np.ndarray, config: dict, device: str = "auto") -> list[dict]:
    """Compatibility helper for one-off callers."""
    return SamSegmenter(config, device=device).segment(image)
