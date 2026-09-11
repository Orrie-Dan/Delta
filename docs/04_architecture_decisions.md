# 04 — Architecture decisions (what / why)

Short decision log for choices already baked into the repo.

---

## AD-1 — Two-stage pipeline (detect then recognize)

**What:** Person detection and uniform recognition are separate modules and datasets.  
**Why:** Different labels, different failure modes, different data sources. A single end-to-end “uniform detector” would be harder to debug and would abuse VisDrone.  
**How:** `src/detection/` vs `src/classification/`; `data/processed/visdrone_person/` vs `data/classification/`.

---

## AD-2 — Three uniform classes

**What:** `our_team`, `known_friendly`, `unknown`.  
**Why:** Matches the research objective (not sports teams). `unknown` is required for open-world / non-matching uniforms.  
**How:** Folder names under `data/classification/` and `configs/classifier.yaml` (`num_classes: 3`).

---

## AD-3 — Single detection class `person`

**What:** YOLO class `0 = person` only. VisDrone `pedestrian` and `people` merge into it.  
**Why:** For cropping uniforms we need people, not VisDrone’s full taxonomy. Merging avoids splitting scarce aerial person boxes across two near-duplicate classes.  
**How:** `PERSON_SOURCE_CATEGORY_IDS = {1, 2}` in `src/data/visdrone.py`.

---

## AD-4 — Train in Colab, develop locally

**What:** Local repo prepares data and code; Colab trains.  
**Why:** GPU availability, cost, and keeping the laptop free of long training jobs.  
**How:** Policy in README + notebooks with `RUN_TRAINING=False` by default.

---

## AD-5 — Raw vs processed data

**What:** `data/raw/` immutable extracts; `data/processed/` derived YOLO trees.  
**Why:** Re-run conversion safely; never destroy official annotations.  
**How:** Converter writes only under `processed/`; raw is gitignored.

---

## AD-6 — No auto-download of VisDrone

**What:** Conversion fails with a clear error if data is missing.  
**Why:** Multi-GB surprises, license/consent awareness, reproducible “I chose to fetch this.”  
**How:** Explicit CLI / notebook cells; no download on import.

---

## AD-7 — pathlib everywhere in library code

**What:** `Path` objects; project-root discovery.  
**Why:** Windows + Linux + Colab without string-joined OS-specific paths.  
**How:** `src/utils/paths.py`.

---

## AD-8 — Absolute YAML at Colab train time

**What:** Committed `configs/visdrone_person.yaml` stays relative; Colab writes a temporary absolute YAML.  
**Why:** Ultralytics resolves `path` relative to process CWD, which is brittle in notebooks.  
**How:** Notebook `02` section “Load dataset config”.

---

## AD-9 — Audit + visualization before training

**What:** Mandatory-quality tooling ships before the train scaffold is used.  
**Why:** Bad labels waste GPU time and hide small-object issues.  
**How:** `dataset_audit.py`, `visualize_annotations.py`.

---

## AD-10 — Tests use synthetic mini VisDrone

**What:** Unit/integration tests build tiny images + annotation files in temp dirs.  
**Why:** CI and laptops must verify mapping/math without downloading VisDrone.  
**How:** `tests/test_visdrone.py`, `tests/test_dataset_audit.py`.

---

## AD-11 — Ignore datasets and weights in Git

**What:** `.gitignore` drops VisDrone media, processed images/labels, `*.pt`, viz dumps; keeps `.gitkeep` and allows small `outputs/metrics/*.json`.  
**Why:** Repo stays cloneable; metrics JSON is useful documentation of a run.  
**How:** See `.gitignore` VisDrone / metrics / visualizations sections.
