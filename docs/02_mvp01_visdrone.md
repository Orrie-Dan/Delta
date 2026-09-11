# 02 — MVP-01: VisDrone person-detection data

## Goal of this MVP

Turn official **VisDrone2019-DET** annotations into a **YOLO person-only** dataset, then validate and preview it. No local training. No uniform labels.

Code lives under `src/data/`. Colab scaffolds live under `notebooks/colab/`.

---

## Step 1 — Understand why VisDrone is in the project

### What

We use VisDrone **only** as a source of aerial/drone-view people for the **person detector**.

### Why

- The long-term system must handle elevated viewpoints and small people.
- VisDrone already has dense person-like boxes (`pedestrian`, `people`).
- It does **not** encode friendly/hostile/our-team uniforms, so it must never fill `data/classification/`.

### How (conceptually)

```text
VisDrone DET
  → keep pedestrian + people
  → map both to class 0 = person
  → ignore cars, bikes, buses, …
  → write YOLO labels + copy images
```

---

## Step 2 — Obtain and place the raw dataset

### What

Manually download/extract VisDrone2019-DET and put it under:

```text
data/raw/visdrone/
```

Typical layout the code detects:

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
    annotations/    # optional / often incomplete
```

Flat `train/` / `val/` / `test-dev/` layouts are also supported when they contain `images/` (+ `annotations/` when available).

### Why

- Official data stays untouched in `raw/`.
- No silent multi-GB download when you `import` or run normal commands (reproducible, intentional, bandwidth-safe).
- Contents are gitignored so the repo stays small.

### How

1. Download VisDrone2019-DET from the official VisDrone release pages.
2. Extract the archives.
3. Copy/move the folders into `data/raw/visdrone/`.
4. Confirm train images **and** train annotations exist.

Discovery logic: `src/data/visdrone.py` → `discover_split_dirs()` / `validate_source_layout()`.

---

## Step 3 — Parse VisDrone annotations

### What

Each annotation `.txt` line is roughly:

```text
bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion
```

We parse rows safely, reject malformed/invalid boxes, and keep only person-source categories.

### Why

- Explicit category IDs prevent silent mistakes if someone “remembers” wrong numbers.
- Invalid geometry must not become YOLO labels.
- Vehicles would teach the detector the wrong concept if retained.

### How (mapping — audit this anytime)

Defined in `src/data/visdrone.py`:

| VisDrone `object_category` | Name | Action |
| --- | --- | --- |
| 0 | ignored | drop |
| 1 | pedestrian | keep → YOLO `0` (`person`) |
| 2 | people | keep → YOLO `0` (`person`) |
| 3–11 | bicycle, car, … | drop |

API:

```python
from src.data.visdrone import parse_annotation_line, parse_annotation_file, visdrone_to_yolo
```

---

## Step 4 — Convert to YOLO format

### What

Write a processed dataset:

```text
data/processed/visdrone_person/
  images/{train,val,test}/
  labels/{train,val,test}/
```

Each label line:

```text
class_id x_center y_center width height
```

Coordinates are **normalized** by image width/height. `class_id` is always `0` for retained people.

Official train/val/test-dev splits are preserved when detected. If test-dev has no usable GT, the converter copies images and reports that clearly instead of inventing people.

### Why

- Ultralytics YOLO expects this layout and label scheme.
- Normalized boxes are resolution-independent.
- Keeping splits separate enables honest validation later in Colab.

### How

From project root (venv active):

```powershell
python -m src.data.visdrone_convert --source data/raw/visdrone --output data/processed/visdrone_person
```

Useful options:

| Flag | Purpose |
| --- | --- |
| `--split train` | Convert one split only (repeatable) |
| `--limit 50` | Smoke-test on N images per split |
| `--overwrite` | Replace existing outputs |
| `--symlink-images` | Prefer symlinks; falls back to copy |
| `--no-copy-images` | Advanced; still ensures images exist when missing |

Implementation: `src/data/visdrone_convert.py`.

Portable paths: relative args are resolved against the project root via `src/utils/paths.py`.

---

## Step 5 — Audit the processed dataset

### What

`dataset_audit` reports, per split and totals:

- image / label file counts  
- total person boxes, average boxes per image  
- images with zero retained boxes  
- missing labels or images  
- malformed / invalid boxes  
- very small boxes (configurable normalized-area threshold; default `0.001`)  
- width / height / area statistics  
- image resolution statistics  

Writes machine-readable JSON under `outputs/metrics/` (e.g. `visdrone_person_audit.json`) and prints a terminal summary.

### Why

- Aerial people are often tiny; small-box stats matter for detector design (`imgsz`, model size).
- Catch conversion bugs before burning Colab GPU hours.
- JSON is shareable and comparable across runs.

### How

```powershell
python -m src.data.dataset_audit `
  --dataset data/processed/visdrone_person `
  --report outputs/metrics/visdrone_person_audit.json
```

Optional:

```powershell
python -m src.data.dataset_audit --small-box-area 0.0005
```

Implementation: `src/data/dataset_audit.py`.

---

## Step 6 — Visualize annotations

### What

Randomly sample images from a split, draw green `person` boxes, **save** images under:

```text
outputs/visualizations/visdrone_person/
```

No GUI required. Source images are never overwritten.

### Why

Human eyeballing catches systematic errors (shifted boxes, wrong scale, empty labels) that averages miss.

### How

```powershell
python -m src.data.visualize_annotations `
  --dataset data/processed/visdrone_person `
  --split val `
  --count 12
```

Implementation: `src/data/visualize_annotations.py` (OpenCV drawing).

---

## Step 7 — YOLO dataset YAML

### What

`configs/visdrone_person.yaml`:

```yaml
path: data/processed/visdrone_person
train: images/train
val: images/val
test: images/test
names:
  0: person
```

### Why

Ultralytics needs a dataset descriptor. Relative `path` keeps the repo portable.

### How

- Locally: point training at this file **from a known CWD**, or prefer absolute paths.
- In Colab: notebook `02` generates a **runtime absolute YAML** so CWD differences do not break loading.

Do not put `C:\Users\...` into the committed YAML.

---

## Step 8 — Automated tests for data logic

### What

Tests under `tests/` cover:

- parsing valid rows  
- ignoring cars  
- pedestrian/people → person  
- YOLO math  
- invalid box rejection  
- path helpers  
- tiny synthetic VisDrone convert + audit  

No download required.

### Why

Protects the category mapping and conversion math when we refactor later.

### How

```powershell
pytest
```

---

## What success looks like for MVP-01

| Check | Expected |
| --- | --- |
| Code + tests | Green without real VisDrone |
| With real VisDrone placed | Convert finishes; audit JSON written; viz images look correct |
| Classification folders | Still empty of VisDrone-derived uniform labels |
| Local training | Not run |

If real VisDrone is missing, MVP-01 is **complete pending real data**.

Next: [03 — Colab workflow](03_colab_workflow.md)
