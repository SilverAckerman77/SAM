# NOSH Egg Segmentation Report

This report is generated only from raw artifacts produced by the scripts.
Task 1, the development experiments, and frozen evaluation are complete.
Video tracking remains pending a supplied video.

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
images. Roboflow-derived base names and perceptual-hash near duplicates are
grouped before splitting; the verification run found zero image overlap.

## Task 1 status

The Kaggle run completed on a Tesla T4 using SAM ViT-H, CUDA, seed 42, and
925.57 seconds for 110 images. Validation passed for all masks. Seventeen
images were flagged for suspiciously large mask area and remain questionable
pseudo-ground-truth cases.

## Task 2 and Task 3 status

Thirteen classical development trials were executed. The frozen method was
LAB lightness thresholding, selected only by development IoU. Development
mean IoU was 0.4155 and evaluation mean IoU was 0.3235. Full per-image
metrics are in `outputs/eval/metrics.csv` and aggregate values are in
`outputs/eval/summary.json`.

## Video tracking status

The supplied YouTube URL was unavailable to yt-dlp, so a local copy was used.
The 360P MP4 is 640x360 at 24 FPS, 1,440 frames, and 60 seconds. The prepared
CPU tracker uses
connected-component instances, IoU plus centroid-distance association,
constant-velocity prediction, missed-track aging, and writes an annotated
video, per-frame CSV, keyframes, and `stats.json`. The run produced 89 unique
track IDs, 5,671 associations, and at most 10 simultaneous active tracks.
This is not a true egg count: keyframes show substantial false positives from
the frozen image segmentation method, including background objects and a
watermark. ID switches require manual review, and no switch count is claimed.
Replace the placeholder Drive link below after uploading the final video.

## Video link

[GOOGLE_DRIVE_VIDEO_LINK]
