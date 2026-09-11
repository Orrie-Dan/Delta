# 03 — Colab workflow

## Goal

Use Google Colab + Drive for **GPU training**, while this GitHub/local repo stays the source of preparation code and notebooks.

```text
Local / GitHub
  → prep VisDrone (or run prep in Colab notebook 01)
  → push / sync code
Google Drive / Colab
  → train person detector
  → save best.pt / last.pt
Later application
  → load checkpoint for inference
```

---

## Step 1 — Put large data on Google Drive

### What

Store raw VisDrone and processed YOLO trees on Drive, for example:

```text
MyDrive/uniform-detection/
  data/raw/visdrone/
  data/processed/visdrone_person/
  datasets/visdrone_person.zip    # notebook 02 trains from this zip
  models/detector/
  outputs/
```

### Why

Colab VMs are ephemeral. Drive persists datasets and checkpoints across sessions.

### How

Upload manually, or sync from a machine that already ran conversion. Edit notebook variables `DRIVE_ROOT`, `VISDRONE_RAW_DIR`, `VISDRONE_PROCESSED_DIR` to match **your** Drive paths (placeholders only in the repo).

---

## Step 2 — Notebook 01: prepare / verify (no training)

**File:** `notebooks/colab/01_visdrone_prepare.ipynb`

### What

Sections:

1. Environment info  
2. Mount Drive  
3. Clone/update GitHub repo  
4. Define project + Drive paths  
5. Install `requirements.txt`  
6. Verify VisDrone layout  
7. Run conversion (calls `src.data.visdrone_convert`)  
8. Run audit (calls `src.data.dataset_audit`)  
9. Show sample visualizations  
10. Print paths ready for YOLO  

Top cell variables: `REPO_URL`, `PROJECT_DIR`, `DRIVE_ROOT`, `VISDRONE_RAW_DIR`, `VISDRONE_PROCESSED_DIR`, …

### Why

Keeps heavy Python logic in `src/` (tested, reusable) and makes Colab a thin orchestration layer. You only edit a small config cell.

### How

1. Open the notebook in Colab.  
2. Edit the configuration cell (repo URL + Drive paths).  
3. Run cells top to bottom.  
4. Set `LIMIT` for a smoke run; `None` for full conversion.

---

## Step 3 — Notebook 02: YOLO26n baseline (MVP-02)

**File:** `notebooks/colab/02_train_person_detector.ipynb`

### What

Canonical MVP-02 training notebook: extract Drive `visdrone_person.zip` once to `/content/datasets/visdrone_person/`, sanity-check, train `yolo26n.pt`, validate, size-bin recall, latency, persist `best.pt` + `metrics.json`.

### Why

Baseline before any architecture comparison or hyperparameter search.

### How

1. Upload `visdrone_person.zip` to `DRIVE_ROOT/datasets/`.
2. Open notebook 02 → GPU runtime.
3. Edit config (`REPO_URL`, `DATASET_ZIP`, `MODEL_NAME="yolo26n.pt"`).
4. Run with `EXTRACT_DATASET=True` and `RUN_TRAINING=False` through dataset checks.
5. Set `RUN_TRAINING=True` (CUDA required).
6. Collect checkpoints under `DRIVE_ROOT/checkpoints/detector/<EXPERIMENT_NAME>/`.

See [06_mvp02_person_detector_baseline.md](06_mvp02_person_detector_baseline.md).

---

## Step 4 — Notebooks 03 and 04 (placeholders)

### What

- `03_train_uniform_classifier.ipynb` — future CNN on real uniforms (`our_team` / `known_friendly` / `unknown`). States VisDrone must not supply those labels.  
- `04_evaluate_models.ipynb` — future detector + classifier evaluation.

### Why

Reserve the workflow slots so later MVPs have an obvious home without implying they are implemented.

### How

Ignore for MVP-01 except to read the TODOs.

---

## Step 5 — Bring checkpoints back into the app later

### What

After Colab training, keep named weights on Drive, e.g. `{EXPERIMENT_NAME}_best.pt`, and eventually copy into `models/detector/` for local/app use.

### Why

Inference and research demos should pin a known checkpoint, not a random Colab `runs/` folder.

### How

Document the experiment name and date when you copy weights. (Application inference code is a later MVP.)

---

## Security / hygiene

- Never commit Drive credentials or personal notebook outputs with secrets.  
- `REPO_URL` and `DRIVE_ROOT` in notebooks are placeholders — replace locally in Colab, do not commit private tokens.  
- Large `.pt` / dataset files stay gitignored.
