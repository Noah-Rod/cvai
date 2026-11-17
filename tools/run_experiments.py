"""Run the experiment training runs end to end on a GPU machine (GPU hub, Kaggle, local GPU).

  python tools/run_experiments.py --list                     # what would run
  python tools/run_experiments.py exp2                       # the 4 box variants
  python tools/run_experiments.py exp1 --variant corner_obb  # training data, best variant
  python tools/run_experiments.py exp2-corner_obb            # a single run
  python tools/run_experiments.py --summary                  # results tables (markdown)

Each run: generate synthetic hands → export one box variant → train → evaluate on the real
test set (data/real/test, with and without the known card count). Every step is skipped when
its output exists, so the same command resumes after an interruption; an interrupted training
continues from its last epoch. Runs that share data share it (all exp2 runs use the same
synthetic hands; `exp1` with `--aug full` is the matching exp2 run).

Outputs: data/synth/, data/exports/, models/<run>.pt, runs/<run>/ (training plots, val
metrics), runs/eval/<run>[-known].json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jass_cv.dataset import VARIANTS  # noqa: E402

SYNTH_N = 3000
BOX_VARIANTS = ("full_aabb", "full_obb", "corner_aabb", "corner_obb")  # experiment 2
REAL_TRAIN = ROOT / "data" / "real" / "train"
REAL_TEST = ROOT / "data" / "real" / "test"


@dataclass(frozen=True)
class Run:
    name: str
    variant: str
    aug: str = "full"
    real: int = 0  # number of real training photos added to the synthetic ones

    @property
    def model(self) -> str:
        return "yolo11n-obb.pt" if self.variant.endswith("_obb") else "yolo11n.pt"


def exp2_runs() -> list[Run]:
    return [Run(f"exp2-{v}", v) for v in BOX_VARIANTS]


def exp1_runs(variant: str) -> list[Run]:
    runs = [Run(f"exp1-{variant}-aug_{aug}", variant, aug) for aug in ("none", "basic")]
    runs.append(Run(f"exp2-{variant}", variant))  # aug full = the exp2 run, trained once
    runs += [Run(f"exp1-{variant}-real{n}", variant, real=n) for n in (20, 50)]
    return runs


def all_runs(variant: str) -> dict[str, Run]:
    return {r.name: r for r in exp2_runs() + exp1_runs(variant)}


def select(targets: list[str], variant: str) -> list[Run]:
    runs, known = [], all_runs(variant)
    for t in targets:
        if t == "exp2":
            runs += exp2_runs()
        elif t == "exp1":
            runs += exp1_runs(variant)
        elif t in known:
            runs.append(known[t])
        else:
            raise SystemExit(f"unknown run {t!r}; see --list")
    return list(dict.fromkeys(runs))  # keep order, drop duplicates


# ---------------------------------------------------------------------------- environment


def available_cpus() -> int:
    """CPUs this process may use; respects container CPU limits (cgroup v2), unlike cpu_count."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, n or 1)


def default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sh(*args) -> None:
    cmd = [sys.executable, *map(str, args)]
    print("$", " ".join(cmd[1:]), flush=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


# ---------------------------------------------------------------------------- steps


def synth_dir(aug: str) -> Path:
    return ROOT / "data" / "synth" / f"v1-aug_{aug}"


def ensure_synth(aug: str, workers: int) -> Path:
    out = synth_dir(aug)
    if (out / ".complete").exists():
        return out
    sh("-m", "jass_cv.synth", "--out", out, "--n", SYNTH_N, "--aug", aug, "--workers", workers)
    (out / ".complete").write_text("")
    return out


def real_subset(n: int) -> Path:
    """The first n photos of a fixed shuffle of data/real/train, so real20 ⊂ real50."""
    import random

    src = REAL_TRAIN / "annotations.jsonl"
    if not src.exists():
        raise SystemExit(f"{src} missing: label the real training photos first (docs/dataset.md)")
    records = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line]
    if len(records) < n:
        raise SystemExit(f"need {n} real training photos, {src} has {len(records)}")
    random.Random(0).shuffle(records)
    out = ROOT / "data" / "exports" / "_real" / f"train-{n}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "annotations.jsonl", "w", encoding="utf-8") as fh:
        for r in records[:n]:
            fh.write(json.dumps({**r, "file": str((REAL_TRAIN / r["file"]).resolve())}) + "\n")
    return out


def ensure_export(run: Run, workers: int) -> Path:
    data_yaml = ROOT / "data" / "exports" / run.name / "data.yaml"
    if data_yaml.exists():
        return data_yaml
    synth = ensure_synth(run.aug, workers)
    train = ["--train", synth / "train"]
    if run.real:
        train += ["--train", real_subset(run.real)]
    out = data_yaml.parent
    sh("-m", "jass_cv.dataset", "export", "--variant", run.variant, *train,
       "--val", synth / "val", "--out", out)  # fmt: skip
    return data_yaml


def ensure_model(run: Run, args) -> Path:
    weights = ROOT / "models" / f"{run.name}.pt"
    if weights.exists():
        return weights
    data_yaml = ensure_export(run, args.workers)
    sh("tools/train.py", "--data", data_yaml, "--model", run.model, "--name", run.name,
       "--epochs", args.epochs, "--batch", args.batch, "--imgsz", args.imgsz,
       "--device", args.device, "--workers", min(8, args.workers), "--resume")  # fmt: skip
    return weights


def ensure_eval(run: Run, weights: Path, args) -> None:
    hands = REAL_TEST / "hands.csv"
    if not hands.exists():
        print(f"skip evaluation of {run.name}: {hands} missing", flush=True)
        return
    for known in (False, True):
        out = ROOT / "runs" / "eval" / f"{run.name}{'-known' if known else ''}.json"
        if out.exists():
            continue
        sh("tools/evaluate.py", "--weights", weights, "--images", REAL_TEST / "images",
           "--hands", hands, "--imgsz", args.imgsz, "--device", args.device,
           *(["--known-count"] if known else []))  # fmt: skip


# ---------------------------------------------------------------------------- summary


def val_map50(run_name: str) -> float | None:
    """Best validation mAP50 from the Ultralytics results.csv of a run."""
    path = ROOT / "runs" / run_name / "results.csv"
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(fh)]
    cols = [k for k in (rows[0] if rows else {}) if k.startswith("metrics/mAP50(")]
    return max(float(r[cols[0]]) for r in rows) if cols else None


def eval_result(run_name: str, known: bool = False) -> dict | None:
    path = ROOT / "runs" / "eval" / f"{run_name}{'-known' if known else ''}.json"
    return json.loads(path.read_text()) if path.exists() else None


def summary_table(runs: list[Run]) -> str:
    def fmt(v):
        return "" if v is None else f"{v:.3f}"

    lines = [
        "| Run | card F1 | exact-hand acc. | exact-hand acc. (known count) | val mAP50 |",
        "|---|---|---|---|---|",
    ]
    for r in runs:
        ev, known = eval_result(r.name), eval_result(r.name, known=True)
        lines.append(
            f"| {r.name} | {fmt(ev and ev['card_f1'])} | {fmt(ev and ev['exact_hand_accuracy'])}"
            f" | {fmt(known and known['exact_hand_accuracy'])} | {fmt(val_map50(r.name))} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("targets", nargs="*", help="exp1, exp2 or run names (see --list)")
    ap.add_argument("--variant", default="corner_obb", choices=VARIANTS,
                    help="box variant for exp1 (the winner of exp2)")  # fmt: skip
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16, help="-1 = let Ultralytics pick for the GPU")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="default: first GPU, else mps, else cpu")
    ap.add_argument("--workers", type=int, default=available_cpus())
    ap.add_argument("--list", action="store_true", help="list the runs and their status")
    ap.add_argument("--summary", action="store_true", help="print the results tables")
    args = ap.parse_args(argv)

    if args.list or args.summary:
        groups = {"Experiment 2": exp2_runs(), "Experiment 1": exp1_runs(args.variant)}
        for title, runs in groups.items():
            print(f"\n{title}\n")
            if args.summary:
                print(summary_table(runs))
            else:
                for r in runs:
                    done = (ROOT / "models" / f"{r.name}.pt").exists()
                    print(f"  {'done' if done else '    '}  {r.name:32s} {r.variant:12s} "
                          f"aug={r.aug:5s} real={r.real}")  # fmt: skip
        return
    if not args.targets:
        ap.error("name exp1, exp2 or a run (see --list)")

    args.device = args.device or default_device()
    runs = select(args.targets, args.variant)
    print(f"device={args.device} workers={args.workers} runs: {', '.join(r.name for r in runs)}")
    for i, run in enumerate(runs, start=1):
        print(f"\n=== [{i}/{len(runs)}] {run.name} ===", flush=True)
        ensure_eval(run, ensure_model(run, args), args)
    print("\n" + summary_table(runs))


if __name__ == "__main__":
    main()
