from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/video_v2"
    rows = list(csv.DictReader((output / "tracks.csv").open(encoding="utf-8")))
    by_id: dict[int, list[dict[str, str]]] = defaultdict(list)
    by_frame: dict[int, int] = defaultdict(int)
    for row in rows:
        track_id = int(row["id"])
        frame = int(row["frame"])
        by_id[track_id].append(row)
        if int(row["lost"]) == 0:
            by_frame[frame] += 1
    lengths = []
    for track_id, values in sorted(by_id.items()):
        active_frames = [int(row["frame"]) for row in values if int(row["lost"]) == 0]
        lengths.append({
            "id": track_id,
            "first_frame": min(active_frames),
            "last_frame": max(active_frames),
            "active_frames": len(active_frames),
            "observed_rows": len(values),
            "short_track": len(active_frames) < 10,
        })
    with (output / "track_lengths.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(lengths[0]))
        writer.writeheader()
        writer.writerows(lengths)
    changes = []
    frames = sorted(by_frame)
    for previous, current in zip(frames, frames[1:]):
        if current == previous + 1 and abs(by_frame[current] - by_frame[previous]) >= 3:
            changes.append({"start_frame": previous, "end_frame": current, "reason": "rapid_active_count_change"})
    diagnostics = {
        "track_count": len(lengths),
        "short_track_count": sum(item["short_track"] for item in lengths),
        "active_track_count_min": min(by_frame.values(), default=0),
        "active_track_count_max": max(by_frame.values(), default=0),
        "id_switch_candidate_ranges": changes[:50],
        "manual_review_note": "Candidate ranges indicate abrupt active-track changes; they are not confirmed ID switches.",
    }
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
