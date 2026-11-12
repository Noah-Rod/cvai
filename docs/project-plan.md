# Project plan — 2 people × ~15 h

Two tracks that only meet at the `Detector` interface and the dataset format, so both
people can work in parallel from day one. The dummy detector unblocks track B until the
first model exists.

- **A — Data & Model**: data capture, labeling or synthetic data, training, evaluation
- **B — Apps & Platform**: API, frontend, live tool, Docker/CI, demo

| # | Task | A | B |
|---|---|---|---|
| 0 | Kickoff: settle the open decisions in `architecture.md`, agree on the labeling rules | 1 h | 1 h |
| 1 | Capture real photos (several sessions, varied conditions) | 2 h | – |
| 2 | Synthetic data generator (card cut-outs → random backgrounds, auto labels) *or* more manual labeling | 4 h | – |
| 3 | Label the real photos (val/test + some train) | 3 h | 2 h |
| 4 | Baseline training → `models/best.pt`, wired into the API | 2 h | – |
| 5 | Iterate (augmentations, model size, more data where the confusion matrix is weak) | 2 h | – |
| 6 | Evaluation report: test mAP, confusion matrix, failure cases | 1 h | – |
| 7 | API hardening: ONNX backend, input limits, tests against the real model | – | 3 h |
| 8 | Frontend: multi-image upload, result gallery, card overview (all 36 cards, found/missing) | – | 3 h |
| 9 | Live feed: polish the dev tool; optional in-browser live view (streamlit-webrtc) | – | 2 h |
| 10 | CI, image size and build time, optional deployment | – | 2 h |
| 11 | Demo, README/docs, presentation | – | 2 h |
| | **Total** | **15 h** | **15 h** |

## Milestones

| Milestone | Done when |
|---|---|
| M0 Scaffold | stack runs end to end on the dummy model *(this commit)* |
| M1 Decisions | open points in `architecture.md` decided and the doc updated |
| M2 Data v1 | train/val/test exists, ≥ 40 instances per class in train |
| M3 Baseline | real model served by the API, test mAP@50 measured |
| M4 Final | target metric reached (proposal: mAP@50 ≥ 0.90 on real test photos), demo ready |

## Risks

| Risk | Mitigation |
|---|---|
| Labeling takes longer than planned (36 classes!) | synthetic data; label with model assistance after the first baseline |
| Test score is too optimistic because of near-duplicate frames | split by session (see dataset.md) |
| No GPU | Colab/Kaggle free tier; `yolo11n` trains on small data in well under 1 h |
| Under/Ober and 6–9 get confused | targeted extra data; higher input size; two-stage classifier fallback (D1) |
