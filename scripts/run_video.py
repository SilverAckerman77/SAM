from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classical_seg import segment
from src.tracking import MultiObjectTracker, draw_tracks, extract_detections
from src.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/video.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if not config["input_video"]:
        raise ValueError("Set input_video in configs/video.yaml before running video tracking.")
    capture = cv2.VideoCapture(config["input_video"])
    if not capture.isOpened():
        raise OSError(f"Could not open video: {config['input_video']}")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = Path(config.get("frozen_config", output.parent / "experiments" / "frozen_config.json"))
    method = json.loads(frozen_path.read_text(encoding="utf-8"))["method"] if frozen_path.exists() else "final"
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    tracker = MultiObjectTracker(max_lost=int(config["max_lost"]))
    max_concurrent_tracks = 0
    active_track_counts: list[int] = []
    keyframes = {0, max(0, frame_count // 4), max(0, frame_count // 2),
                 max(0, (3 * frame_count) // 4), max(0, frame_count - 1)}
    keyframe_dir = output / "keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    with (output / "tracks.csv").open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow(["frame", "id", "x", "y", "width", "height", "centroid_x", "centroid_y", "area", "lost"])
        frame = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            mask = segment(image, method)
            detections = extract_detections(mask, int(config["min_instance_area"]))
            tracks = tracker.update(detections)
            active_count = sum(track.lost == 0 for track in tracks)
            active_track_counts.append(active_count)
            max_concurrent_tracks = max(max_concurrent_tracks, active_count)
            annotated = draw_tracks(image, tracks, mask)
            writer.write(annotated)
            if frame in keyframes:
                cv2.imwrite(str(keyframe_dir / f"frame_{frame:06d}.jpg"), annotated)
            for track in tracks:
                x, y, track_width, track_height = track.bbox
                csv_writer.writerow([frame, track.track_id, x, y, track_width, track_height,
                                     track.centroid[0], track.centroid[1], track.area, track.lost])
            frame += 1
    capture.release()
    writer.release()
    stats = {
        "input_video": config["input_video"],
        "method": method,
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frame,
        "duration_seconds": frame / fps if fps else 0.0,
        "unique_track_ids": tracker.next_id - 1,
        "association_matches": tracker.total_match_count,
        "max_concurrent_tracks_observed": max_concurrent_tracks,
        "mean_concurrent_tracks_observed": sum(active_track_counts) / len(active_track_counts) if active_track_counts else 0.0,
        "track_length_distribution": {
            "min": min(tracker.completed_ages + [track.age for track in tracker.tracks], default=0),
            "max": max(tracker.completed_ages + [track.age for track in tracker.tracks], default=0),
            "count": len(tracker.completed_ages) + len(tracker.tracks),
        },
        "short_track_threshold_frames": 10,
        "short_track_count": sum(age < 10 for age in tracker.completed_ages)
        + sum(track.age < 10 for track in tracker.tracks),
        "max_lost_frames": config["max_lost"],
        "id_switches_manual_review_required": True,
    }
    (output / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
