#!/usr/bin/env python3
"""
Run `rgc_train_loss_balanced.py` (triplet_og mode) on every cell in a *triplet* HDF5 dataset.

Each cell is trained in its own subprocess. All cells share one output directory:
- `<output_dir>/for_evaluation/rgc_train_loss_balanced_<cell_id>_<piece>.pkl`  (trained models)
- `<output_dir>/cells_data/cell_<cell_id>-<piece>/`                         (per-cell plots and histories)
- `<output_dir>/deploy_logs/`                                              (one log per cell)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import IO

import h5py


MODE = "triplet_og"
TRAIN_SCRIPT = Path(__file__).resolve().parent / "rgc_train_loss_balanced.py"
EVAL_PICKLE_TAG = "rgc_train_loss_balanced"


def iter_cells(dataset_path: Path):
    with h5py.File(dataset_path, "r") as h5f:
        cells_group = h5f["cells"]
        for cell_key in cells_group.keys():
            g = cells_group[cell_key]
            cell_id = int(g.attrs["cell_id_piece"])
            piece = str(g.attrs["piece"])
            yield cell_key, cell_id, piece


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset_path",
        required=True,
        type=Path,
        help="Path to the triplet HDF5 dataset (filename must contain 'triplet').",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory for all training outputs (created if missing).",
    )
    parser.add_argument(
        "--loss_features",
        type=str,
        default=None,
        help=(
            "Comma-separated loss terms passed to each training run. Choices: sodium_peaks, "
            "diffusion_peaks, potassium_peaks, sodium_velocity, ei_widths_so_po, estim. Default: all six."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for each training run (default: current interpreter).",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index in the dataset cell iteration (for sharding).")
    parser.add_argument("--limit", type=int, default=None, help="Max number of cells to run (for testing / sharding).")
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=0,
        help="Max concurrent training processes. 0 means launch all selected cells at once.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path.resolve()
    output_dir = args.output_dir.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"--dataset_path not found: {dataset_path}")
    if "triplet" not in dataset_path.name.lower():
        raise ValueError(f"--dataset_path filename must contain 'triplet'. Got: {dataset_path.name}")
    if args.max_parallel < 0:
        raise ValueError("--max_parallel must be >= 0")

    logs_dir = output_dir / "deploy_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {dataset_path}")
    print(f"Output dir: {output_dir}")
    print(f"Loss features: {args.loss_features or 'all six (default)'}")
    print(f"Writing logs to {logs_dir}")

    failures = 0
    launched: list[tuple[int, str]] = []
    running: list[tuple[int, str, subprocess.Popen[bytes], IO[bytes]]] = []

    def _drain_finished() -> None:
        nonlocal failures
        still_running: list[tuple[int, str, subprocess.Popen[bytes], IO[bytes]]] = []
        for idx, cell_id_piece, proc, log_f in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((idx, cell_id_piece, proc, log_f))
                continue
            log_f.close()
            if rc != 0:
                failures += 1
                print(f"  - FAILED [{idx}] Cell {cell_id_piece} (exit={rc})", flush=True)
            else:
                print(f"  - OK     [{idx}] Cell {cell_id_piece}", flush=True)
        running[:] = still_running

    for idx, (_cell_key, cell_id, piece) in enumerate(iter_cells(dataset_path)):
        if idx < args.start:
            continue
        if args.limit is not None and len(launched) >= args.limit:
            break

        cell_id_piece = f"{cell_id}-{piece}"
        launched.append((cell_id, piece))
        print(f"[{idx}] Cell {cell_id_piece}", flush=True)

        cmd = [
            args.python,
            str(TRAIN_SCRIPT),
            "--dataset_path",
            str(dataset_path),
            "--output_dir",
            str(output_dir),
            "--mode",
            MODE,
            "--skip_bielec",
            "--piece",
            piece,
            "--cell_id",
            str(cell_id),
        ]
        if args.loss_features:
            cmd += ["--loss_features", args.loss_features]

        log_path = logs_dir / f"{idx:04d}_{cell_id_piece}.log"
        log_f = open(log_path, "wb")
        proc = subprocess.Popen(cmd, env=os.environ.copy(), stdout=log_f, stderr=subprocess.STDOUT)
        running.append((idx, cell_id_piece, proc, log_f))

        if args.max_parallel > 0:
            while len(running) >= args.max_parallel:
                _drain_finished()
                if len(running) >= args.max_parallel:
                    running[0][2].wait()
                    _drain_finished()

    while running:
        running[0][2].wait()
        _drain_finished()

    # A run can exit 0 but fail to write its model pickle (the trainer only warns), so check explicitly.
    for_eval_dir = output_dir / "for_evaluation"
    missing = [
        f"{cell_id}-{piece}"
        for cell_id, piece in launched
        if not (for_eval_dir / f"{EVAL_PICKLE_TAG}_{cell_id}_{piece}.pkl").is_file()
    ]
    print(f"\nDone. cells={len(launched)} failures={failures} missing_model_pickles={len(missing)}")
    if missing:
        print("Cells without a model pickle in for_evaluation/: " + ", ".join(missing))
    return 0 if failures == 0 and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
