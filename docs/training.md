# Training: laptop, GPU hub or Kaggle, same code

`tools/train.py` is the same everywhere; only `--device` changes. `tools/run_experiments.py`
chains generate → export → train → evaluate for every experiment run.

| Where | When | Device |
|---|---|---|
| Windows laptop, native with `uv` | smoke tests, small runs, evaluation | `--device cpu` |
| Docker (`tools` service) | if you prefer containers; also CPU only | `--device cpu` |
| **GPU hub (JupyterHub)** | the real experiment runs | first GPU, picked automatically |
| Kaggle notebook | fallback if the GPU hub is unavailable | `--device 0` (GPU) |

Install `uv` on Windows (PowerShell), then work in the repo folder:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Smoke test on a laptop (~10 min on CPU)

```powershell
uv run --extra synth python -m jass_cv.synth --out data/synth/smoke --n 200 --placeholder-cards
uv run --extra synth python -m jass_cv.dataset export --variant corner_obb --train data/synth/smoke/train --val data/synth/smoke/val --out data/exports/smoke-corner_obb
uv run --extra inference python tools/train.py --data data/exports/smoke-corner_obb/data.yaml --model yolo11n-obb.pt --name smoke --epochs 3
```

The first `--extra inference` run installs PyTorch (CPU build on Windows, ~1 GB) and
downloads the pretrained checkpoint. Checkpoints: `yolo11n.pt` for upright boxes,
`yolo11n-obb.pt` for rotated boxes.

CPU is fine for this but too slow for the experiments. Measured in a 4-core Linux container
(`yolo11n-obb`, 640 px, batch 16): **~1.5 min per epoch for 360 images**. A full run
(3,000 images × 50 epochs) would take roughly **10 hours**: possible overnight, but not 8 times.

## Experiment runs on the GPU hub

Everything runs in a **terminal** in JupyterLab (*File → New → Terminal*), not in a
notebook: a run started with `nohup` keeps going when you close the browser tab.

**Once: clone and install**

```bash
git clone https://github.com/Noah-Rod/cvai.git && cd cvai
python -m venv --system-site-packages .venv   # reuses the hub's PyTorch if it has one
source .venv/bin/activate
pip install -e ".[inference]"
nvidia-smi                                    # which GPU, how much memory is free
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

If the repository is private, clone with a GitHub personal access token (read-only) as the
password. Later sessions: `cd cvai && source .venv/bin/activate && git pull`.

**Real photos.** The runs evaluate on `data/real/test/` (images + `hands.csv`) and exp1 adds
`data/real/train/` (Label Studio export, see [dataset.md](dataset.md)). Zip `data/real/` on
your laptop, drag the zip into the JupyterLab file browser, then `unzip real.zip -d data/`.
Without a test set the runs still train; evaluation is skipped and runs again later.

**Start the runs**

```bash
python tools/run_experiments.py --list               # what will run
nohup python tools/run_experiments.py exp2 > exp2.log 2>&1 &
tail -f exp2.log                                      # Ctrl+C stops watching, not the run
```

Then, with the winning box variant of experiment 2:

```bash
nohup python tools/run_experiments.py exp1 --variant corner_obb > exp1.log 2>&1 &
```

The two experiments share data: exp2 trains the four box variants on the same synthetic
hands, and exp1's "aug full" run *is* the exp2 run of that variant, so it isn't trained twice.
Single runs also work: `python tools/run_experiments.py exp2-corner_obb`.

**Model size.** Runs use YOLO11n by default. `--size s` (YOLO11s, ~3x the parameters, often
better on small details such as the index digit) trains the same run as
`<run>-yolo11s` on the same exported data; `--summary` lists it under the n run. A cheap
extra comparison for the best variant:
`nohup python tools/run_experiments.py exp2-corner_obb --size s > size.log 2>&1 &`.
Expect roughly 2x the training time; lower `--batch` if the GPU runs out of memory.

**Interrupted?** JupyterHubs stop servers that are idle or have run too long. Start the
server again and repeat the same command: finished steps are skipped and a training continues
from its last epoch.

**Time and disk (estimates).** Generating 3,000 hands takes ~4 min on 4 CPU cores and
~310 MB per augmentation level (exp1 needs three: ~1 GB). Training is ~30–60 min per run on a
GPU at 50 epochs. Both experiments: 8 runs, roughly an evening. If the GPU runs out of
memory, pass `--batch 8`; `--batch -1` lets Ultralytics pick the largest batch that fits.

**Results**

```bash
python tools/run_experiments.py --summary    # both results tables, ready for project-plan.md
zip -r results.zip models runs/eval runs/*/results.csv runs/*/*.png
```

Download `results.zip` (right-click → *Download*). `models/*.pt` are the weights for the app;
`runs/<run>/` has the training curves and confusion matrices for the slides.

## Experiment runs on Kaggle (fallback, free GPU)

1. Create a Kaggle **Dataset** with your `data/cards/` (and `data/backgrounds/`,
   `data/hands/` if you have them) and one with `data/real/test/`.
2. New notebook → *Settings*: Accelerator **GPU**, Internet **on**; add both datasets as input.
3. Cells:

```python
!git clone https://github.com/Noah-Rod/cvai.git && cd cvai && pip install -q -e ".[inference]"
%cd cvai
!python -m jass_cv.synth --out data/synth/v1 --n 3000 --aug full \
    --cards-dir /kaggle/input/jass-cards/cards --backgrounds-dir /kaggle/input/jass-cards/backgrounds
!python -m jass_cv.dataset export --variant corner_obb \
    --train data/synth/v1/train --val data/synth/v1/val --out data/exports/v1-corner_obb
!python tools/train.py --data data/exports/v1-corner_obb/data.yaml --model yolo11n-obb.pt \
    --name v1-corner_obb --epochs 50 --device 0
!python tools/evaluate.py --weights models/v1-corner_obb.pt --device 0 \
    --images /kaggle/input/jass-real-test/images --hands /kaggle/input/jass-real-test/hands.csv
```

4. Download `models/*.pt`, `runs/<name>/` (plots, confusion matrix) and `runs/eval/*.json`
   from the notebook output.

If the repository is private, add a GitHub token as a Kaggle *Secret* and clone with it, or
upload the repo as a Kaggle dataset.

Kaggle's free GPU quota is limited per week (check the current amount in your account).
~8 runs fit if you keep `--epochs` reasonable (start with 30–50).

## Using a trained model

Put the `.pt` file in `models/` and point the API at it:

```powershell
# .env
JASS_MODEL_PATH=models/v1-corner_obb.pt
```

`docker compose restart api`, or for the live tool:
`uv run --extra live python tools/live_detect.py --weights models/v1-corner_obb.pt`.
Rotated-box models are drawn as rotated boxes in the frontend and the live tool.
