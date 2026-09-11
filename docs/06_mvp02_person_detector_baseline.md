# 06 — MVP-02: Person detector baseline (YOLO26n)

## Goal

Establish a **clean, reproducible** person-detection baseline on the MVP-01 VisDrone → person dataset using **pretrained YOLO26n** in **Google Colab**.

Status after code lands: **READY FOR COLAB TRAINING**  
Status after you train and review metrics: then mark **COMPLETE**.

---

## What / why / how (high level)

| | |
| --- | --- |
| **What** | Colab notebook trains `yolo26n.pt`, validates, analyzes small objects, saves `best.pt` + JSON metrics to Drive |
| **Why** | Confirm the detector learns person-only VisDrone; expose small-person failure modes early; freeze a baseline before tuning |
| **How** | Upload `visdrone_person.zip` to Drive → open notebook 02 → GPU runtime → extract once → set `RUN_TRAINING=True` |

---

## Drive layout

```text
uniform-detection/
├── datasets/visdrone_person.zip
├── checkpoints/detector/<EXPERIMENT_NAME>/best.pt|last.pt
└── experiments/<EXPERIMENT_NAME>/
    ├── metrics.json
    ├── metadata.json
    ├── artifacts/   # results.csv, curves, args.yaml, …
    └── predictions/
```

Zip local `data/processed/visdrone_person/` (train+val is enough). Notebook 02 extracts once to `/content/datasets/visdrone_person/`.

---

## Notebook sections (02)

1. Configuration (`EXTRACT_DATASET=True`, `RUN_TRAINING=False` by default)  
2. Environment / GPU verification  
3. Mount Drive  
4. Clone/update repo  
5. Install dependencies  
6. Extract `visdrone_person.zip` once → `/content/datasets/visdrone_person/`  
7. Validate dataset structure  
8. Build runtime absolute YAML  
9. Load pretrained YOLO  
10. Baseline configuration summary  
11. Train (gated)  
12. Validate  
13. Inspect metrics + size-bin recall  
14. Inference visualization  
15. Save checkpoints to Drive  
16. Save metrics / latency / model size  
17. Final summary  

---

## Small-object evaluation method

**Bins (normalized box area `w*h`, not COCO):**

- small: `area < 0.001`
- medium: `0.001 <= area < 0.01`
- large: `area >= 0.01`

**Recall @ IoU 0.5:** greedy one-to-one matching of predictions to GT by descending IoU; count TP per GT size bin. This is **not** size-specific mAP.

Code: `src/evaluation/size_analysis.py`.

---

## Safety gates

- No auto-train on notebook open (`RUN_TRAINING=False`).
- If `RUN_TRAINING=True` without CUDA → hard stop (no accidental CPU baseline).
- Dataset zip missing on Drive → clear upload instructions (no VisDrone re-download).

---

## Local verification (no training)

```powershell
pytest
```

Does **not** start YOLO training.
