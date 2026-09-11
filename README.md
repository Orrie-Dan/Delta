# Uniform Detection

Phone-camera **soldier uniform recognition** research prototype: detect people/soldiers in images or video, crop each person, and categorize visible uniform appearance as **our_team**, **known_friendly**, or **unknown**.

## Documentation (what / why / how)

Step-by-step process docs live in [`docs/`](docs/README.md):

| Doc | Topic |
| --- | --- |
| [Overview](docs/00_overview.md) | Goal, pipeline, constraints |
| [MVP-00 environment](docs/01_mvp00_environment.md) | Venv, deps, verification |
| [MVP-01 VisDrone](docs/02_mvp01_visdrone.md) | Place → convert → audit → visualize |
| [Colab workflow](docs/03_colab_workflow.md) | Drive + training notebooks |
| [Architecture decisions](docs/04_architecture_decisions.md) | Why the design looks this way |
| [Step checklist](docs/05_step_checklist.md) | Tick-off runbook |
| [MVP-02 baseline](docs/06_mvp02_person_detector_baseline.md) | YOLO26n Colab training |

## Problem statement

Given images/video from a phone camera (and later aerial/drone imagery), the system should:

1. Detect people/soldiers in each frame.
2. Crop each detected person.
3. Analyze visible uniform appearance with a CNN-based recognition model.
4. Categorize each crop as **our_team**, **known_friendly**, or **unknown**.
5. Later investigate temporal tracking and adaptation to aerial/drone viewpoints.

Person detection and uniform recognition remain separate stages in the initial MVP.

## Current MVP architecture

```
Phone / video
    → person detector
    → person crop
    → uniform recognition model
    → our_team / known_friendly / unknown
```

## Training policy (important)

**All model training happens in Google Colab.**

| Location | Role |
| --- | --- |
| Local / GitHub repo | Dataset preparation, validation, reusable code, Colab notebooks, later inference |
| Google Drive + Colab | YOLO / classifier training, checkpoint storage |

Do **not** train YOLO or the uniform CNN locally as part of the standard workflow.

## Current status

**MVP-02 — Person detector baseline (READY FOR COLAB TRAINING)**

MVP-01 person-only VisDrone conversion is complete (~7,019 images / ~120k person boxes).  
MVP-02 delivers the Colab training notebook + evaluation helpers for a **YOLO26n** baseline.  
**MVP-02 is not COMPLETE until you train on Colab and inspect real metrics.**

| Included | Not included yet |
| --- | --- |
| MVP-00 / MVP-01 data prep | Local YOLO training |
| Colab notebook `02` fully wired for baseline | Uniform classification / ResNet |
| Small-object size-bin recall helpers | Tracking, phone streaming, drone |
| Checkpoint + metrics persistence design | Multi-model comparison / tuning |

## MVP-02 — Person Detector Baseline

### What

Train a **pretrained YOLO26n** (`yolo26n.pt`) person detector on the VisDrone → **person-only** dataset, in **Google Colab**, to establish a reproducible baseline (quality, small-person recall, stability, latency, model size).

### Why

- MVP-01 proved the labels; MVP-02 asks whether a small detector can learn the task.
- ~89% of boxes are very small — overall mAP alone is not enough; we also measure **project-specific** size-bin recall @ IoU 0.5.
- Training stays in Colab so the laptop is not the GPU farm.

### Dataset on Google Drive (not GitHub)

Upload a zip of the processed tree (from `data/processed/visdrone_person/`) to Drive:

```text
MyDrive/uniform-detection/
├── datasets/visdrone_person.zip
├── checkpoints/detector/
└── experiments/
```

Notebook 02 extracts the zip once to `/content/datasets/visdrone_person/` (`EXTRACT_DATASET = True`), then trains from that local copy.

### How to run (Colab)

1. Open [`notebooks/colab/02_train_person_detector.ipynb`](notebooks/colab/02_train_person_detector.ipynb) in Colab.
2. **Runtime → Change runtime type → GPU**.
3. Edit the configuration cell: `REPO_URL`, `DATASET_ZIP`, hyperparameters.
4. Leave `EXTRACT_DATASET = True` and `RUN_TRAINING = False` while you verify extract / YAML / GPU.
5. Set `RUN_TRAINING = True` only when ready (CUDA required; CPU training is blocked).
6. After training, check Drive:

```text
checkpoints/detector/mvp02_yolo26n_visdrone_person_baseline/best.pt
experiments/mvp02_yolo26n_visdrone_person_baseline/metrics.json
```

### Baseline hyperparameters (defaults)

| Setting | Default |
| --- | --- |
| Model | `yolo26n.pt` |
| Image size | 640 |
| Epochs | 50 |
| Batch | 16 |
| Workers | 2 |
| Patience | 10 |
| Seed | 42 |
| Experiment | `mvp02_yolo26n_visdrone_person_baseline` |

### Metrics captured

- Precision / Recall / mAP50 / mAP50-95 (Ultralytics val)
- GT size distribution + size-bin recall @ IoU 0.5 (small / medium / large; **not** COCO bins, **not** size-mAP)
- Inference latency (ms/image, approx FPS) with warm-up
- `best.pt` size (MB) and parameter count
- `metadata.json` (versions, GPU name, counts, seed note)

Detailed what/why/how: [`docs/06_mvp02_person_detector_baseline.md`](docs/06_mvp02_person_detector_baseline.md).

## MVP-01: VisDrone for person detection

### Why VisDrone?

VisDrone provides aerial/drone-view person annotations that are useful for training a **person detector** that must cope with small objects and elevated viewpoints.

### What VisDrone is not

- VisDrone is **not** a uniform-recognition dataset.
- Do **not** create `our_team` / `known_friendly` / `unknown` labels from VisDrone.
- Uniform recognition will use a separate real uniform dataset later.

### Category mapping

Official VisDrone DET categories retained:

| VisDrone ID | Name | Output |
| --- | --- | --- |
| 1 | pedestrian | `0 = person` |
| 2 | people | `0 = person` |

All other VisDrone categories (vehicles, etc.) are ignored.

### Workflow

```
Local / GitHub
  → place raw VisDrone under data/raw/visdrone/
  → convert + audit + visualize
  → push code / use Colab notebooks
Google Drive / Colab
  → train person detector
  → save best.pt / last.pt
  → use checkpoint later in the application
```

### Place the raw VisDrone dataset

Manually download/extract the official **VisDrone2019-DET** release and put it under:

```text
data/raw/visdrone/
  VisDrone2019-DET-train/
    images/
    annotations/
  VisDrone2019-DET-val/
    images/
    annotations/
  VisDrone2019-DET-test-dev/
    images/
    annotations/   # optional; often no usable public GT
```

Flat `train/` / `val/` / `test-dev/` layouts are also detected when present.

Nothing in this repo auto-downloads multi-GB VisDrone archives.

### Convert to YOLO person-only format

```powershell
python -m src.data.visdrone_convert --source data/raw/visdrone --output data/processed/visdrone_person
```

Useful options:

```powershell
python -m src.data.visdrone_convert --split train --limit 50 --overwrite
```

Output:

```text
data/processed/visdrone_person/
  images/{train,val,test}/
  labels/{train,val,test}/
```

If test-dev has no usable ground truth, images are still copied and the converter reports that clearly instead of inventing detections.

### Audit the processed dataset

```powershell
python -m src.data.dataset_audit --dataset data/processed/visdrone_person --report outputs/metrics/visdrone_person_audit.json
```

### Visualize labels

```powershell
python -m src.data.visualize_annotations --dataset data/processed/visdrone_person --split val --count 12
```

Writes under `outputs/visualizations/visdrone_person/` (does not modify source images).

### Ultralytics dataset YAML

`configs/visdrone_person.yaml` defines relative paths and `names: {0: person}`.

In Colab, notebook `02` writes an **absolute-path runtime YAML** so Ultralytics resolves the dataset regardless of working directory.

### Colab notebooks

| Notebook | Purpose |
| --- | --- |
| `notebooks/colab/01_visdrone_prepare.ipynb` | Convert / audit / preview (no training) |
| `notebooks/colab/02_train_person_detector.ipynb` | YOLO training scaffold (`RUN_TRAINING=False` by default) |
| `notebooks/colab/03_train_uniform_classifier.ipynb` | Placeholder for future uniform CNN |
| `notebooks/colab/04_evaluate_models.ipynb` | Placeholder for detector + classifier eval |

Edit the top configuration cell (`REPO_URL`, `DRIVE_ROOT`, paths) before running.

## Environment setup (Windows PowerShell)

Requires **Python 3.11+**.

From the project root (`uniform-detection`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks script activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Verify the environment

```powershell
python -m src.utils.check_environment
pytest
python main.py
```

## Project folder explanation

| Path | Role |
| --- | --- |
| `data/raw/visdrone/` | Official VisDrone DET extract (gitignored contents) |
| `data/processed/visdrone_person/` | YOLO person-only images/labels |
| `data/classification/` | Future uniform crops: `our_team` / `known_friendly` / `unknown` |
| `models/detector/` | Saved person-detector weights (from Colab) |
| `models/classifier/` | Future uniform-recognition weights |
| `notebooks/colab/` | Colab preparation and training scaffolds |
| `src/data/` | VisDrone parse/convert/audit/visualize |
| `src/utils/paths.py` | Portable project-root helpers |
| `configs/visdrone_person.yaml` | Ultralytics dataset config template |
| `outputs/metrics/` | Audit JSON (and later training metrics) |
| `outputs/visualizations/` | Drawn annotation previews |

Empty data/model/output directories are kept in git via `.gitkeep` files; raw/processed VisDrone media and weights are ignored.
