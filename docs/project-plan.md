# Project plan: 2 people × ~15 h

Deliverables: **live demo + final presentation** (research question, pipeline, results of
both experiments, failure cases, live demo).

Already in place (scaffold): API, Streamlit frontend with "which card was played", live
tool, Docker, CI, generator v1, export to 5 box variants, training and evaluation scripts,
Label Studio setup. So the hours go into **data, experiments and analysis**, not plumbing.

## Tracks

- **A: Data & transfer to real photos (experiment 1)**
- **B: Box variants, evaluation & demo (experiment 2)**

| # | Task | A | B |
|---|---|---|---|
| 0 | Kickoff: walk through the repo, run the smoke pipeline on both laptops, agree on the capture protocol | 1 h | 1 h |
| 1 | Photograph the 36 cards, extract faces, tune `CORNER_REGION` | 1.5 h | – |
| 2 | Generator realism: background photos, thumb/hand cut-outs, compare synthetic vs real side by side, fix what looks off | 3.5 h | – |
| 3 | Real photos: test sessions + `hands.csv` (both), real-train sessions | 2 h | 2 h |
| 4 | Label ~50 real training photos in Label Studio, convert, check the conversion | 1 h | 2 h |
| 5 | Kaggle notebook working end to end (generate → export → train → download weights) | – | 1.5 h |
| 6 | Experiment 2: train 4 box variants, evaluate on real test, pick the winner | – | 3 h |
| 7 | Experiment 1: aug none/basic/full, + 20 / + 50 real photos | 3 h | – |
| 8 | Error analysis: confusions, failure gallery, effect of known hand size | 1 h | 1.5 h |
| 9 | Demo: best model in the app, demo script (fan → photo → play a card → photo) | – | 1.5 h |
| 10 | Presentation: slides, results tables/plots, rehearsal | 2 h | 1.5 h |
| | **Total** | **15 h** | **15 h** |

Training runs: about 4 (exp. 2) + 4 (exp. 1, one shared) = **~8 runs**, each ~30–60 min on a
Kaggle GPU (estimate). Start them early and let them run while you do other tasks.

## Milestones

| Milestone | Done when |
|---|---|
| M0 Scaffold | stack + pipeline run end to end with placeholder cards *(done)* |
| M1 Data v1 | 36 real card faces, 3,000 synthetic hands, real test set with `hands.csv` |
| M2 Baseline | first model trained on synthetic only, evaluated on real test |
| M3 Experiments | both experiments run, results tables filled in |
| M4 Demo | best model in the app, presentation ready |

## Results tables (fill in)

Experiment 2: box variant (same synthetic data, `--aug full`):

| Variant | card F1 | exact-hand acc. | exact-hand acc. (known count) | val mAP50 |
|---|---|---|---|---|
| full_aabb | | | | |
| full_obb | | | | |
| corner_aabb | | | | |
| corner_obb | | | | |

Experiment 1: training data (best variant):

| Training data | card F1 | exact-hand acc. |
|---|---|---|
| synthetic, aug none | | |
| synthetic, aug basic | | |
| synthetic, aug full | | |
| + 20 real | | |
| + 50 real | | |

## Risks

| Risk | Mitigation |
|---|---|
| Synthetic images look too different from real ones → poor transfer | real backgrounds + hand cut-outs; add real photos (that's experiment 1) |
| Training too slow on the laptops | Kaggle GPU; local only for smoke tests |
| Corner region ambiguous for some cards | that's a finding for experiment 2; full-card variants as fallback |
| Test score too optimistic | split by capture session |
