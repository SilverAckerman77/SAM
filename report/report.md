# NOSH Egg Segmentation Report

This report is generated only from raw artifacts produced by the scripts.
Task 1, development experiments, frozen evaluation, and video diagnostics are
complete. Existing pre-revision outputs remain preserved.

## AI tool disclosure

**Edit this section before submission.** AI coding tools assisted with
boilerplate, scaffolding, debugging, and report drafting. The candidate must
describe their own decisions, review, execution, and verification here.

## Dataset exploration

The supplied local export contains 110 JPEG images, all 640x640 RGB. Filenames
use the Roboflow `.rf.<hash>` convention. No per-image annotation files were
present; the export contained two dataset README files. The inspected sample
includes single and multiple eggs, eggs held in hands, white and brown eggs,
bowls and metal surfaces, foliage, shadows, and varied indoor backgrounds.
The generated contact sheet and inventory are under `outputs/exploration/`
after running `scripts/inspect_dataset.py`.

The reproducible split contains 87 development images and 23 evaluation
images. `outputs/split/leakage_check.json` reports zero filename overlap and
zero content-hash overlap.

## Task 1 status

The revised Kaggle run completed on a Tesla T4 using GroundingDINO plus SAM
ViT-H, CUDA, seed 42, and 266.54 seconds for 110 images. Validation passed for
all masks, and the A2 review sheets are in `outputs/task1_gsam/qa/`. The
manifest contains zero `reference_ok=false` rows. These remain
pseudo-ground-truth masks rather than human annotations.

## Task 2 and Task 3 status

Eighteen classical development trials were executed. The frozen method was
the `bright` threshold, selected only by development IoU against the revised
pseudo-labels. Development mean IoU was 0.3694. The one-time revised
evaluation reports mean IoU 0.2682, mean Dice 0.3829, boundary F1 0.1390, and
mean HD95 249.16 pixels. Full revised metrics are in
`outputs/eval_v2/metrics.csv` and `outputs/eval_v2/summary.json`. For
transparency, the earlier evaluation remains available and reported mean IoU
0.3235.

## Video tracking status

The supplied YouTube URL was unavailable to yt-dlp, so a local copy was used.
The 360P MP4 is 640x360 at 24 FPS, 1,440 frames, and 60 seconds. The prepared
CPU tracker uses
connected-component instances, IoU plus centroid-distance association,
constant-velocity prediction, missed-track aging, and writes an annotated
video, per-frame CSV, keyframes, and `stats.json`. The revised run produced 71 unique track IDs, 6,243 associations, and at most
10 simultaneous active tracks. There were 30 tracks shorter than 10 frames
and five candidate ID-switch ranges.
This is not a true egg count: keyframes show substantial false positives from
the frozen image segmentation method, including background objects and a
watermark. Candidate ranges require manual review, and no confirmed switch count is claimed.
Replace the placeholder Drive link below after uploading the final video.

## Video link

[GOOGLE_DRIVE_VIDEO_LINK]
