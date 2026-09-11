# 05 — Step checklist

Use this as a runbook. Each item is **What** you are completing; details live in the linked docs.

## A. Environment (MVP-00)

See [01_mvp00_environment.md](01_mvp00_environment.md).

- [ ] Open / clone `uniform-detection`
- [ ] Create `.venv` and activate it (PowerShell)
- [ ] Upgrade pip and `pip install -r requirements.txt`
- [ ] Run `python -m src.utils.check_environment` → PASS
- [ ] Run `pytest`
- [ ] Run `python main.py`

## B. VisDrone person data (MVP-01)

See [02_mvp01_visdrone.md](02_mvp01_visdrone.md).

- [ ] Download VisDrone2019-DET manually
- [ ] Extract under `data/raw/visdrone/` (train images + annotations present)
- [ ] Convert:

```powershell
python -m src.data.visdrone_convert --source data/raw/visdrone --output data/processed/visdrone_person
```

- [ ] Optional smoke first: add `--limit 50`
- [ ] Audit:

```powershell
python -m src.data.dataset_audit --dataset data/processed/visdrone_person --report outputs/metrics/visdrone_person_audit.json
```

- [ ] Visualize:

```powershell
python -m src.data.visualize_annotations --dataset data/processed/visdrone_person --split val --count 12
```

- [ ] Spot-check a few saved images under `outputs/visualizations/visdrone_person/`
- [ ] Confirm you did **not** label VisDrone into `our_team` / `known_friendly` / `unknown`

## C. Colab detector baseline — MVP-02

See [06_mvp02_person_detector_baseline.md](06_mvp02_person_detector_baseline.md) and [03_colab_workflow.md](03_colab_workflow.md).

- [ ] Zip `data/processed/visdrone_person/` (train+val) and upload to Drive `datasets/visdrone_person.zip`
- [ ] Push/sync GitHub repo with MVP-02 code
- [ ] Open `notebooks/colab/02_train_person_detector.ipynb` in Colab
- [ ] Enable GPU runtime
- [ ] Set `REPO_URL`, `DATASET_ZIP`, confirm `MODEL_NAME = "yolo26n.pt"`
- [ ] Run cells with `EXTRACT_DATASET = True` and `RUN_TRAINING = False` through dataset YAML validation
- [ ] Set `RUN_TRAINING = True` and train
- [ ] Confirm Drive has `checkpoints/detector/mvp02_yolo26n_visdrone_person_baseline/best.pt`
- [ ] Review `experiments/.../metrics.json` (mAP + size-bin recall + latency)
- [ ] Spot-check prediction images under the experiment folder

## D. Explicitly not done yet

- [ ] Declare MVP-02 COMPLETE (only after real Colab metrics review)
- [ ] Uniform dataset collection/labeling
- [ ] Classifier training (notebook 03)
- [ ] Joint evaluation (notebook 04)
- [ ] Phone streaming
- [ ] Tracking
- [ ] Production API / frontend

## Quick “why am I blocked?” guide

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| Convert errors “source root does not exist” | VisDrone not placed | Fix `data/raw/visdrone/` layout |
| Convert errors “Train annotations not found” | Incomplete extract | Ensure `annotations/*.txt` beside train images |
| Audit shows many missing labels | Conversion interrupted / partial copy | Re-run convert with `--overwrite` |
| Viz boxes look wrong | Wrong image size read / bad labels | Re-audit; inspect one label file vs image |
| Colab cannot find dataset | Drive path mismatch | Fix `VISDRONE_PROCESSED_DIR`; regenerate absolute YAML |
| Tempted to train locally | Policy + GPU | Use notebook 02 in Colab |
