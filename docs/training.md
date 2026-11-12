# Training: Windows laptop or Kaggle, same code

`tools/train.py` is the same everywhere; only `--device` changes.

| Where | When | Device |
|---|---|---|
| Windows laptop, native with `uv` | smoke tests, small runs, evaluation | `--device cpu` |
| Docker (`tools` service) | if you prefer containers; also CPU only | `--device cpu` |
| **Kaggle notebook** | the real experiment runs | `--device 0` (GPU) |

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

## Experiment runs on Kaggle (free GPU)

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
