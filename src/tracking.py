from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

import cv2
import numpy as np


@dataclass
class Detection:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: int


@dataclass
class Track:
    track_id: int
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: int
    velocity: tuple[float, float] = (0.0, 0.0)
    lost: int = 0
    age: int = 1
    trail: list[tuple[int, int]] = field(default_factory=list)

    def predicted_centroid(self) -> tuple[float, float]:
        return (self.centroid[0] + self.velocity[0], self.centroid[1] + self.velocity[1])


def extract_detections(mask: np.ndarray, min_area: int) -> list[Detection]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    detections = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y = int(stats[label, 0]), int(stats[label, 1])
        width, height = int(stats[label, 2]), int(stats[label, 3])
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        detections.append(Detection(
            component, (x, y, width, height),
            (float(centroids[label][0]), float(centroids[label][1])), area,
        ))
    return detections


def bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x1, y1 = max(lx, rx), max(ly, ry)
    x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union else 0.0


class MultiObjectTracker:
    def __init__(self, max_lost: int = 10, max_distance: float = 160.0) -> None:
        self.max_lost = max_lost
        self.max_distance = max_distance
        self.next_id = 1
        self.tracks: list[Track] = []
        self.total_match_count = 0
        self.id_switch_proxy_count = 0
        self.completed_ages: list[int] = []

    def update(self, detections: list[Detection]) -> list[Track]:
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detections)))
        candidates = []
        for track_index, track in enumerate(self.tracks):
            predicted = track.predicted_centroid()
            for detection_index, detection in enumerate(detections):
                distance = hypot(predicted[0] - detection.centroid[0], predicted[1] - detection.centroid[1])
                if distance <= self.max_distance:
                    score = 0.7 * bbox_iou(track.bbox, detection.bbox) + 0.3 * max(
                        0.0, 1.0 - distance / self.max_distance
                    )
                    candidates.append((score, track_index, detection_index))
        for _, track_index, detection_index in sorted(candidates, reverse=True):
            if track_index not in unmatched_tracks or detection_index not in unmatched_detections:
                continue
            track = self.tracks[track_index]
            detection = detections[detection_index]
            old_centroid = track.centroid
            track.velocity = (detection.centroid[0] - old_centroid[0], detection.centroid[1] - old_centroid[1])
            track.centroid, track.bbox, track.area = detection.centroid, detection.bbox, detection.area
            track.lost, track.age = 0, track.age + 1
            track.trail.append((int(track.centroid[0]), int(track.centroid[1])))
            unmatched_tracks.remove(track_index)
            unmatched_detections.remove(detection_index)
            self.total_match_count += 1
        for track_index in unmatched_tracks:
            self.tracks[track_index].lost += 1
            self.tracks[track_index].centroid = self.tracks[track_index].predicted_centroid()
        retained = []
        for track in self.tracks:
            if track.lost <= self.max_lost:
                retained.append(track)
            else:
                self.completed_ages.append(track.age)
        self.tracks = retained
        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            self.tracks.append(Track(
                self.next_id, detection.centroid, detection.bbox, detection.area,
                trail=[(int(detection.centroid[0]), int(detection.centroid[1]))],
            ))
            self.next_id += 1
        return list(self.tracks)


def draw_tracks(image: np.ndarray, tracks: list[Track], mask: np.ndarray) -> np.ndarray:
    output = image.copy()
    output[mask > 0] = (0.65 * output[mask > 0] + np.array([0, 190, 0]) * 0.35).astype(np.uint8)
    for track in tracks:
        if track.lost:
            continue
        x, y, width, height = track.bbox
        color = (255, 180, 0)
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
        cv2.putText(output, f"ID {track.track_id}", (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        for start, end in zip(track.trail, track.trail[1:]):
            cv2.line(output, start, end, (255, 100, 0), 2)
    return output
