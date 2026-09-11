# 00 — Project overview

## What this project is

A **soldier uniform recognition** research prototype.

Target pipeline (full system, not all built yet):

```text
Phone / video
  → person detector
  → person crop
  → uniform recognition model
  → our_team / known_friendly / unknown
```

Later work may add temporal tracking and aerial/drone adaptation.

## Why we split detection from uniform recognition

| Stage | Job | Labels |
| --- | --- | --- |
| Person detection | Find people in the frame | one class: `person` |
| Uniform recognition | Decide what the uniform looks like | `our_team`, `known_friendly`, `unknown` |

Mixing these early would force one model to solve two different problems with incompatible datasets. VisDrone can teach “where are people?”; it cannot teach “whose uniform is this?”.

## Where work happens

| Place | What we do there | What we do **not** do there |
| --- | --- | --- |
| Local PC / GitHub | Code, dataset prep, audits, visualization, tests, later inference | Full YOLO / CNN training (policy) |
| Google Colab + Drive | Train detector (and later classifier), store `best.pt` | Become the source of truth for application code |

**Why:** Colab gives GPU access without turning the laptop into a training machine. The repo stays the reusable, versioned codebase.

## MVP map

| MVP | What | Status in repo |
| --- | --- | --- |
| **MVP-00** | Project/environment setup | Done |
| **MVP-01** | VisDrone → YOLO person dataset + audit + Colab scaffolds | Done (real data converted) |
| **MVP-02** | YOLO26n person-detector baseline in Colab | **READY FOR COLAB TRAINING** |
| Later | Uniform dataset, classifier, pipeline, tracking | Not started |

## Hard constraints (do not violate casually)

- Do **not** invent `our_team` / `known_friendly` / `unknown` from VisDrone.
- Do **not** auto-download multi-GB datasets by importing modules.
- Do **not** train YOLO/ResNet as the default local workflow.
- Do **not** hard-code machine-specific paths like `C:\Users\...` in library code.

## How to navigate the rest of the docs

1. Set up the environment → [01_mvp00_environment.md](01_mvp00_environment.md)
2. Prepare person-detection data → [02_mvp01_visdrone.md](02_mvp01_visdrone.md)
3. Train in Colab when ready → [03_colab_workflow.md](03_colab_workflow.md)
4. Understand design choices → [04_architecture_decisions.md](04_architecture_decisions.md)
5. Follow a tick-list → [05_step_checklist.md](05_step_checklist.md)
