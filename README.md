# NOSH Egg Segmentation

Reproducible implementation of the NOSH Robotics CV intern task. Data-dependent
results are intentionally not checked in or invented. The dataset URL currently
configured is:

`https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1`

That URL is a Roboflow project page, not a direct export archive. Use a direct
ZIP/export URL or download the dataset locally before running the commands below.

## Setup

Python 3.10–3.13 is recommended for the computer-vision dependencies. Python
3.14.3 was detected during initial setup, but some SAM/OpenCV wheels may not yet
support it. Create an environment and install:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Repository layout

- `configs/`: YAML configuration for data, segmentation, and experiments
- `scripts/`: command-line entry points
- `src/`: readable reusable implementation modules
- `kaggle/`: GPU hand-off notebook for Task 1
- `outputs/`: generated artifacts (gitignored)
- `report/`: generated report sources, figures, and PDF

## Commands

Set `DATASET_ZIP` to a direct ZIP URL or a local ZIP file:

```powershell
python scripts/download_data.py --url $env:DATASET_ZIP --output data/raw
python scripts/inspect_dataset.py --input data/raw --output outputs/exploration
```

### Task 1 — Kaggle GPU hand-off

1. Create a Kaggle notebook and set Accelerator to **GPU (T4 or P100)** and Internet to **ON**.
2. Upload `kaggle/task1_kaggle.ipynb`, set its input switch and fill in the direct dataset URL or attached Kaggle Dataset path.
3. Set `REPO_URL` in the first cell if cloning the repository; the notebook also supports copying the scripts into the notebook.
4. Run All. It skips images that already have masks and writes progress/timing information.
5. Download `/kaggle/working/task1_outputs.zip` from the Output tab.
6. Unzip it into `outputs/task1/` in this repository.

`scripts/run_task1.py` also supports a local CPU run, but SAM is expected to be
slow there:

```powershell
python scripts/run_task1.py --config configs/task1.yaml --input data/raw --output outputs/task1 --device cpu
python scripts/validate_task1.py --input data/raw --task1 outputs/task1
```

Task 1 is a stop gate: inspect flagged overlays and confirm the pseudo-labels
before starting Task 2.

### Remaining tasks

```powershell
python scripts/make_split.py --config configs/split.yaml
python scripts/run_experiments.py --config configs/experiments.yaml
python scripts/evaluate.py --config configs/eval.yaml
python scripts/run_video.py --config configs/video.yaml
python scripts/make_report.py --config configs/report.yaml
```

These commands require real validated Task 1 outputs. They write raw CSV/JSON
artifacts before producing report figures or summaries.

## Video

Provide a local video path or a working `yt-dlp` URL in
`configs/video.yaml`. The supplied YouTube URL was unavailable to yt-dlp, so
the downloaded local copy was used. The 60-second video is 640x360 at 24 FPS.
The implemented tracker extracts connected components, associates detections
using IoU plus centroid distance, maintains constant-velocity predictions,
ages missed tracks, and writes annotated MP4, keyframes, track CSV, and
`stats.json`. The current run produced 89 IDs and at most 10 simultaneous
tracks; manual review is required because the frozen segmentation method
creates visible false positives.

After uploading the result to Google Drive, replace
`[GOOGLE_DRIVE_VIDEO_LINK]` in the report source and README.

## Reproducibility and limitations

Seeds and thresholds live in YAML configuration. The Task 1 masks are SAM
pseudo-ground truth, not human-verified truth. No evaluation number, runtime,
plot, or experiment-log row should be added manually.

## AI disclosure

Edit the marked disclosure section in `report/report.md` to describe which
parts were assisted by AI coding tools and which decisions, runs, reviews, and
verification were completed by the candidate.
