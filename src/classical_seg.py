from __future__ import annotations

import cv2
import numpy as np


def segment(image: np.ndarray, method: str = "final") -> np.ndarray:
    """Simple explainable baseline; all methods operate only on the image."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if method == "otsu":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method in {"hsv_value", "hsv_value_open3"}:
        _, mask = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method in {"lab_lightness", "lab_lightness_open3", "lab_lightness_open11"}:
        _, mask = cv2.threshold(lab[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 4
        )
    elif method == "hsv_saturation":
        _, mask = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "hsv_value_inv":
        _, mask = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "lab_a_inv":
        _, mask = cv2.threshold(lab[:, :, 1], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "lab_b":
        _, mask = cv2.threshold(lab[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "ycrcb_cr_inv":
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        _, mask = cv2.threshold(ycrcb[:, :, 1], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "ycrcb_cb":
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        _, mask = cv2.threshold(ycrcb[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive_inv":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 4
        )
    elif method == "bright":
        mask = np.where(hsv[:, :, 2] > 150, 255, 0).astype(np.uint8)
    elif method in {"bright_low_sat", "bright_low_sat_strict", "bright_low_sat_loose"}:
        # Bright, low-saturation egg-like pixels, followed by conservative cleanup.
        value_threshold = {"bright_low_sat_strict": 120, "bright_low_sat_loose": 70}.get(method, 90)
        saturation_threshold = {"bright_low_sat_strict": 90, "bright_low_sat_loose": 150}.get(method, 120)
        mask = np.where(
            (hsv[:, :, 2] > value_threshold) & (hsv[:, :, 1] < saturation_threshold),
            255,
            0,
        ).astype(np.uint8)
    elif method == "final":
        # Final frozen candidate selected from the development experiment log.
        mask = np.where((hsv[:, :, 2] > 90) & (hsv[:, :, 1] < 120), 255, 0).astype(np.uint8)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
    kernel_size = {
        "lab_lightness_open3": 3,
        "lab_lightness_open11": 11,
        "hsv_value_open3": 3,
    }.get(method, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
