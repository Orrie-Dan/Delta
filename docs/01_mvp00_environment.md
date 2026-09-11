# 01 — MVP-00: Environment and repository setup

## Goal of this MVP

Create a clean, importable Python project so every later step has the same folders, configs, and dependency set.

---

## Step 1 — Create the project skeleton

### What

We created `uniform-detection/` with:

- `data/` for raw and processed datasets
- `models/` for checkpoints (empty until Colab produces them)
- `src/` for reusable Python packages
- `configs/` for YAML placeholders
- `notebooks/` for Colab workflows
- `outputs/` for audits, visualizations, metrics
- `tests/` for automated checks
- `requirements.txt`, `.gitignore`, `README.md`, `main.py`

Empty directories are kept in Git with `.gitkeep` files.

### Why

- Git does not track empty folders; `.gitkeep` preserves the intended layout.
- Separating `raw` vs `processed` prevents overwriting originals.
- Separating `detection` vs `classification` data prevents VisDrone from contaminating uniform labels.
- `src/` packages (`detection`, `classification`, `tracking`, `pipeline`) match the future pipeline stages.

### How

Already done in the repo. You only need to clone/open the project. If recreating from scratch, mirror the tree in the root README.

---

## Step 2 — Define semantic class folders (uniforms)

### What

Classification dataset placeholders:

```text
data/classification/{train,val,test}/{our_team,known_friendly,unknown}/
```

Detection config uses `class_name: person` (not “player”).

### Why

The research target is soldier uniforms with three outcomes. Person detection stays a single class so YOLO training stays simple and aligned with VisDrone’s pedestrian/people merge.

### How

Folders already exist. Later you will place real cropped uniform images into these class folders — **not** VisDrone frames labeled as uniforms.

Configs:

- `configs/detector.yaml` → person detector hyperparameters (placeholders)
- `configs/classifier.yaml` → 3-class recognition hyperparameters (placeholders)

---

## Step 3 — Create a virtual environment (Windows PowerShell)

### What

An isolated Python environment under `.venv/`.

### Why

Keeps torch/OpenCV/Ultralytics versions from conflicting with other projects on the machine.

### How

From `uniform-detection/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**GPU note:** For local GPU *inference* later, install a CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/). Training still belongs in Colab per project policy.

---

## Step 4 — Install dependencies

### What

`requirements.txt` lists the stack: PyTorch, Torchvision, Ultralytics, OpenCV, NumPy, Pandas, scikit-learn, Matplotlib, Albumentations, Pillow, PyYAML, tqdm, pytest.

Versions are mostly unpinned for flexibility.

### Why

- Enough for dataset prep, visualization, and later inference.
- Avoids locking every transitive dependency unless compatibility forces it.
- Does **not** add ByteTrack or experiment trackers yet (YAGNI until those MVPs).

### How

```powershell
pip install -r requirements.txt
```

---

## Step 5 — Verify the environment

### What

`src/utils/check_environment.py` prints:

1. Python version  
2. PyTorch version  
3. CUDA availability  
4. GPU name (if any)  
5. OpenCV version  
6. Ultralytics import  
7. Torchvision import  
8. Clear **PASS** / **FAIL**

Missing packages produce readable messages instead of raw stack traces.

### Why

Catches broken installs before you spend time on VisDrone conversion or Colab sync issues.

### How

```powershell
python -m src.utils.check_environment
```

Smoke import tests:

```powershell
pytest tests/test_environment.py
```

Minimal entry point:

```powershell
python main.py
```

---

## Step 6 — Path helpers

### What

`src/utils/paths.py` finds the project root by walking upward for markers (`requirements.txt` + `configs/`, or `main.py` + `src/`) and builds portable paths with `pathlib`.

### Why

Scripts and notebooks must work on Windows, Linux, and Colab without hard-coded `C:\Users\...` or `/content/...` inside library code. Notebooks may set Drive paths as **variables**; library code resolves relative to the repo.

### How

```python
from src.utils.paths import project_root, visdrone_raw_dir, visdrone_processed_dir

print(project_root())
print(visdrone_raw_dir())
```

---

## What MVP-00 deliberately skipped

| Skipped | Why |
| --- | --- |
| Training | Needs GPU workflow in Colab + real data |
| Phone streaming | Application concern after models exist |
| Tracking / drone control | Later research phase |
| Fake uniform labels | Would poison the recognition problem |

Next: [02 — MVP-01 VisDrone](02_mvp01_visdrone.md)
