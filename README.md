# NOSH Egg Segmentation & Tracking

This is my submission for the NOSH Robotics CV intern task. I segment eggs with a pretrained model (Task 1), rebuild that segmentation with classical image processing (Task 2), measure the gap between the two (Task 3), and track individual eggs in a video (Task 4).

Every number in my report comes from the scripts in this repo. I did not type any result in by hand.

## Results at a glance

| Task | What I did | Outcome |
|---|---|---|
| 1. Pretrained segmentation | GroundingDINO ("egg" text prompt) gives boxes, SAM ViT-H turns them into masks | All 110 images processed on a Kaggle Tesla T4 in 266.54 s; validation passed for every mask |
| 2. Classical segmentation | 18 logged trials on the dev split, including failed ones | Selected the `bright` method |
| 3. Evaluation | Frozen method scored against the Task 1 masks | Revised eval mean IoU **0.2682** (my earlier version scored 0.3235; see "Limitations") |
| 4. Video tracking | Connected components, IoU + centroid association, constant-velocity prediction | 71 IDs, at most 10 simultaneous tracks. This is **not** a true egg count; see "Video tracking" |

## Setup

I recommend Python 3.10-3.13. My machine had Python 3.14.3, and some SAM/OpenCV wheels may not support it yet, so I used 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Getting the data

The dataset URL I configured is a Roboflow project page, not a direct download:

`https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1`

Download the export yourself (or use a direct ZIP link), then point `DATASET_ZIP` at it, either a URL or a local file:

```powershell
python scripts/download_data.py --url $env:DATASET_ZIP --output data/raw
python scripts/inspect_dataset.py --input data/raw --output outputs/exploration
```

## Running the project

### Task 1: mask generation (Kaggle GPU)

SAM is slow on my laptop's CPU, so I ran Task 1 on a free Kaggle GPU. To reproduce:

1. Create a Kaggle notebook, set Accelerator to **GPU (T4 or P100)** and Internet to **ON**.
2. Upload `kaggle/task1_kaggle.ipynb`, then set the dataset source and the code source (a Kaggle Dataset containing `scripts/`, `src/` and `configs/`, or a cloned `REPO_URL`).
3. Run the 5-image smoke test first and check that only the eggs are highlighted.
4. Run the full pass. It skips images that already have masks and prints progress and timing.
5. Download `task1_gsam_outputs.zip` from the Output tab and unzip it into `outputs/task1_gsam/`.

`scripts/run_task1.py` also runs on a CPU, just slowly:

```powershell
python scripts/run_task1.py --config configs/task1_gsam.yaml --input data/raw --output outputs/task1_gsam --device cpu
python scripts/validate_task1.py --input data/raw --task1 outputs/task1_gsam
```

I treated Task 1 as a checkpoint: I reviewed the overlays and review sheets before starting Task 2.

### Tasks 2 and 3: classical segmentation and evaluation

```powershell
python scripts/run_experiments_v2.py
python scripts/run_eval_v2.py
```

The experiment run records 18 trials with what changed, why, the parameters and the dev results. The evaluation script runs the frozen method on the eval split.

### Task 4: video tracking and report

```powershell
python scripts/run_video.py --config configs/video_v2.yaml
python scripts/analyze_video_v2.py --config configs/video_v2.yaml
python scripts/make_report.py --config configs/report.yaml
```

## Repository layout

- `configs/`: YAML settings for data, segmentation, experiments, video and the report
- `scripts/`: command-line entry points
- `src/`: the reusable modules, written to be readable
- `kaggle/`: the notebook for the Task 1 GPU run
- `outputs/`: generated artifacts (gitignored)
- `report/`: report source, figures and the PDF

## Video tracking

The YouTube URL from the task wouldn't download with `yt-dlp`, so I used a local copy of the video (60 seconds, 640x360, 24 FPS). Set the path or a working URL in `configs/video.yaml`.

My tracker extracts connected components from the segmentation, matches them to existing tracks using IoU plus centroid distance, predicts motion with a constant-velocity model, and drops tracks after a few missed frames. It writes an annotated MP4, keyframes, a track CSV and `stats.json`.

In the revised run it produced 71 IDs, 6,243 associations and at most 10 simultaneous tracks. 30 of the tracks lasted fewer than 10 frames, and the analysis flagged five candidate ID-switch ranges. **Please don't read 71 as the number of eggs.** My classical segmentation also picks up background texture in this video, so many of those IDs are false detections. The ID-switch ranges are candidates, not confirmed switches, and they need a manual check.

The video link is in `report/report.md` and in the PDF:
https://youtu.be/Y4JyV94y2_A?si=CaGf-Q5j4qhZ8MEE

## Limitations

- **The Task 1 masks are pseudo ground truth.** They come from a model, not from a human, so the Task 3 numbers measure agreement with SAM, not with reality.
- **How Task 1 changed.** My first attempt used SAM's automatic mask generator with a solidity filter. It leaked hands, pans and tiles into the masks on 17 images. I tried geometric cleaning, but it kept favoring round non-egg shapes, so I switched to text-prompted detection with box-prompted SAM and applied no egg-shape rules. That keeps the reference from being tuned toward my own shape prior.
- **The eval split was scored twice across versions.** I scored an earlier version of the pipeline on the eval split once (mean IoU 0.3235, still in `outputs/eval/`) before I rebuilt the Task 1 reference. The revised pipeline was then scored once with its frozen configuration (0.2682). I tuned only on the dev split.
- **The video result is weak.** See above.

## Reproducibility

Seeds and thresholds live in the YAML configs. Nothing in the evaluation, runtimes, plots or experiment log was added by hand.

## AI tool disclosure

I used AI coding tools (Google Antigravity and GitHub Copilot) for scaffolding, debugging, documentation and formatting. I chose the methods and parameters, reviewed the outputs, ran the experiments and Kaggle notebook, interpreted the results, and checked the implementation. The full disclosure is in the report.