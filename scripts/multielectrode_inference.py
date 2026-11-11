#!/usr/bin/env python3
"""
Compute and cache triplet (multi-electrode) response probabilities for all cells in a triplet H5.

- Load per-cell {RGC_kwargs, full_params} from a folder of trained-model pickles
- For each cell, load (stim_elecs, amps_gsort, pos/neg thresholds) from the H5
- Compute model thresholds at stim electrodes via binary search
- Rescale current-adjustment parameters using (model_thresh / gt_thresh)
- Run vmapped multi-electrode inference over the experimental amplitude grid
- Also compute the superposition and independent-site baselines from the H5 thresholds
- Save per-cell results to `<output_dir>/cell_<cell_id>-<piece>_triplet_probs.pkl`

Notes:
- This script ALWAYS recomputes and overwrites outputs (no caching flags).
- It fails loudly if any H5 cell does not have a matching model pickle.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import h5py
import jax.numpy as jnp
import numpy as np

from rgc_models.rgc_model import RGC


# Filenames written by rgc_train_loss_balanced.py into <output_dir>/for_evaluation/.
DEFAULT_PICKLE_REGEX = r"rgc_train_loss_balanced_(?P<cell_id>[0-9]+)_(?P<piece>.+)\.pkl"


@dataclass(frozen=True)
class CellKey:
    cell_id: int
    piece: str

    @property
    def cell_id_piece(self) -> str:
        return f"{self.cell_id}-{self.piece}"


def _iter_cells(dataset_path: str) -> list[str]:
    with h5py.File(dataset_path, "r") as h5f:
        return list(h5f["cells"].keys())


def _read_cell_inputs(dataset_path: str, cell_key: str) -> tuple[CellKey, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns:
      (cell_id/piece, stim_elecs[int32, (3,)], amps[float32/64, (N,3)], gt_pos_thresh[(3,)], gt_neg_thresh[(3,)])
    """
    cell_idx = int(cell_key.split("_")[1])
    with h5py.File(dataset_path, "r") as h5f:
        g = h5f["cells"][f"cell_{cell_idx}"]
        cell_id = int(g.attrs["cell_id_piece"])
        piece = str(g.attrs["piece"])
        stim_elecs = jnp.asarray(g.attrs["stim_elecs"][:], dtype=jnp.int32)
        amps = jnp.asarray(g["amps_gsort"][:])

        # Camera-ready: use thresholds stored in the final triplet H5 (fail loudly if missing).
        assert "pos_thresholds_uA" in g, "Missing dataset pos_thresholds_uA in triplet H5"
        assert "neg_thresholds_uA" in g, "Missing dataset neg_thresholds_uA in triplet H5"
        gt_pos_thresh = jnp.asarray(g["pos_thresholds_uA"][:])
        gt_neg_thresh = jnp.asarray(g["neg_thresholds_uA"][:])

    return CellKey(cell_id=cell_id, piece=piece), stim_elecs, amps, gt_pos_thresh, gt_neg_thresh


def _load_params_pickle(pkl_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if "RGC_kwargs" not in data:
        raise KeyError(f"Missing key 'RGC_kwargs' in {pkl_path}")
    if "full_params" not in data:
        raise KeyError(f"Missing key 'full_params' in {pkl_path}")
    return data["RGC_kwargs"], data["full_params"]


def _get_threshold_crit_voltage_estimates(
    cell_obj: RGC,
    full_params: dict[str, Any],
    stim_elecs: jnp.ndarray,
    max_current: float = 20.0,
    num_iterations: int = 15,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    pos_thresh, neg_thresh = [], []
    for elec in list(map(int, jnp.asarray(stim_elecs))):
        (lower, upper), _ = cell_obj.jitted_binary_search_threshold(
            elec, max_current, num_iterations, full_params, is_positive=True
        )
        pos_thresh.append((lower + upper) / 2.0)

        (lower, upper), _ = cell_obj.jitted_binary_search_threshold(
            elec, max_current, num_iterations, full_params, is_positive=False
        )
        neg_thresh.append((lower + upper) / 2.0)

    return jnp.asarray(pos_thresh), jnp.asarray(neg_thresh)


def _compute_adjusted_triplet_probs(
    cell_obj: RGC,
    full_params: dict[str, Any],
    stim_elecs: jnp.ndarray,
    amps: jnp.ndarray,
    gt_pos_thresh: jnp.ndarray,
    gt_neg_thresh: jnp.ndarray,
) -> jnp.ndarray:
    # Model thresholds
    model_pos_thresh, model_neg_thresh = _get_threshold_crit_voltage_estimates(cell_obj, full_params, stim_elecs)

    # Adjust only the stim electrodes.
    adjusted_full_params = {k: v for k, v in full_params.items()}
    adjusted_full_params["positive_current_adjustment"] = adjusted_full_params["positive_current_adjustment"].at[stim_elecs].set(
        full_params["positive_current_adjustment"][stim_elecs] * (model_pos_thresh / gt_pos_thresh)
    )
    adjusted_full_params["negative_current_adjustment"] = adjusted_full_params["negative_current_adjustment"].at[stim_elecs].set(
        full_params["negative_current_adjustment"][stim_elecs] * (model_neg_thresh / gt_neg_thresh)
    )

    triplet_probs = cell_obj.jitted_multielec_inference_vmapped(adjusted_full_params, stim_elecs, amps)
    triplet_probs.block_until_ready()
    return triplet_probs


def _superposition_model_probs(
    amps: jnp.ndarray,
    *,
    gt_pos_thresh: jnp.ndarray,
    gt_neg_thresh: jnp.ndarray,
    threshold_measurement_max: float = 3.0,
    default_sat_amp: float = 5.0,
) -> np.ndarray:
    """
    Camera-ready version of `evaluate.py::generate_linear_model_probs`, but using thresholds
    provided directly by the H5 rather than `get_triplet_single_elec_thresholds(params_multisite)`.

    Returns float64 array of shape (N,) with 0/1 values.
    """
    amps_np = np.asarray(amps)
    pos_thresh_np = np.asarray(gt_pos_thresh)
    neg_thresh_np = np.asarray(gt_neg_thresh)

    pos_thresh_np = np.where(np.abs(pos_thresh_np) > threshold_measurement_max, default_sat_amp, pos_thresh_np)
    neg_thresh_np = np.where(np.abs(neg_thresh_np) > threshold_measurement_max, -default_sat_amp, neg_thresh_np)

    pos_plane_values = np.sum(amps_np / pos_thresh_np, axis=1)
    neg_plane_values = np.sum(amps_np / neg_thresh_np, axis=1)

    probs = np.where((pos_plane_values >= 1.0) | (neg_plane_values >= 1.0), 1.0, 0.0)
    return probs.astype(np.float64)


def _independent_model_probs(
    amps: jnp.ndarray,
    *,
    gt_pos_thresh: jnp.ndarray,
    gt_neg_thresh: jnp.ndarray,
) -> np.ndarray:
    """
    Camera-ready version of `evaluate.py::generate_nonlinear_model_probs`, but using thresholds
    provided directly by the H5 rather than `get_triplet_single_elec_thresholds(params_multisite)`.

    Returns float64 array of shape (N,) with 0/1 values.
    """
    amps_np = np.asarray(amps)
    pos_thresh_np = np.asarray(gt_pos_thresh)
    neg_thresh_np = np.asarray(gt_neg_thresh)

    above_pos_thresh = amps_np > pos_thresh_np
    below_neg_thresh = amps_np < neg_thresh_np

    probs = np.any(above_pos_thresh | below_neg_thresh, axis=1).flatten()
    assert probs.shape == (amps_np.shape[0],)
    return probs.astype(np.float64)


def _build_pickle_map(params_dir: Path, pickle_regex: str) -> dict[CellKey, Path]:
    """
    Build mapping from (cell_id, piece) -> pickle path, using a regex that MUST contain
    named capture groups (?P<cell_id>...) and (?P<piece>...).
    """
    rx = re.compile(pickle_regex)
    pkl_paths = sorted(p for p in params_dir.iterdir() if p.is_file() and p.suffix == ".pkl")

    mapping: dict[CellKey, Path] = {}
    for p in pkl_paths:
        m = rx.match(p.name)
        if not m:
            continue
        if "cell_id" not in m.groupdict() or "piece" not in m.groupdict():
            raise ValueError(
                "--pickle_regex must include named groups (?P<cell_id>...) and (?P<piece>...). "
                f"Got groups={list(m.groupdict().keys())} for match on {p.name}"
            )
        ck = CellKey(cell_id=int(m.group("cell_id")), piece=str(m.group("piece")))
        if ck in mapping:
            raise ValueError(f"Duplicate pickle match for {ck.cell_id_piece}: {mapping[ck]} and {p}")
        mapping[ck] = p

    return mapping


def _infer_one_cell(
    cell_key: str,
    *,
    dataset_path: str,
    params_by_cell: dict[CellKey, Path],
    out_dir: Path,
) -> tuple[str, bool, str]:
    """
    Returns (cell_key, success, message).
    """
    try:
        ck, stim_elecs, amps, gt_pos_thresh, gt_neg_thresh = _read_cell_inputs(dataset_path, cell_key)
        if ck not in params_by_cell:
            raise FileNotFoundError(
                f"No params pickle found for cell {ck.cell_id_piece} in --models_dir. "
                "Your --pickle_regex likely did not match this cell."
            )

        pkl_path = params_by_cell[ck]
        RGC_kwargs, full_params = _load_params_pickle(pkl_path)
        cell_obj = RGC(**RGC_kwargs)

        triplet_probs = _compute_adjusted_triplet_probs(
            cell_obj=cell_obj,
            full_params=full_params,
            stim_elecs=stim_elecs,
            amps=amps,
            gt_pos_thresh=gt_pos_thresh,
            gt_neg_thresh=gt_neg_thresh,
        )

        superposition_probs = _superposition_model_probs(
            amps,
            gt_pos_thresh=gt_pos_thresh,
            gt_neg_thresh=gt_neg_thresh,
        )
        independent_probs = _independent_model_probs(
            amps,
            gt_pos_thresh=gt_pos_thresh,
            gt_neg_thresh=gt_neg_thresh,
        )

        out_path = out_dir / f"cell_{ck.cell_id_piece}_triplet_probs.pkl"
        payload = {
            "cell_id": ck.cell_id,
            "piece": ck.piece,
            "stim_elecs": jnp.asarray(stim_elecs),
            "triplet_amps_experiment": jnp.asarray(amps),
            "triplet_probs_adjusted_experiment": jnp.asarray(triplet_probs),
            "superposition_model": superposition_probs,
            "independent_model": independent_probs,
            "source_params_pickle": str(pkl_path),
        }
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)

        return cell_key, True, f"wrote {out_path}"
    except Exception as e:
        return cell_key, False, str(e)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_path", required=True, type=str, help="Path to triplet HDF5 dataset.")
    parser.add_argument(
        "--models_dir",
        required=True,
        type=str,
        help=(
            "Directory containing per-cell trained-model pickles with keys {RGC_kwargs, full_params} "
            "(the for_evaluation/ folder written by training)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory for per-cell inference pickles (created if missing).",
    )
    parser.add_argument(
        "--pickle_regex",
        default=DEFAULT_PICKLE_REGEX,
        type=str,
        help=(
            "Regex matched against pickle *filenames* in models_dir. Must include named groups "
            "(?P<cell_id>...) and (?P<piece>...). Default matches rgc_train_loss_balanced.py outputs."
        ),
    )
    parser.add_argument(
        "--num_workers",
        required=False,
        type=int,
        default=min(5, max(1, mp.cpu_count() - 1)),
        help="Max parallel cells (default min(5, cpu_count-1)).",
    )
    parser.add_argument(
        "--allow_missing_cells",
        action="store_true",
        help="Run inference only for cells with param pickles; skip H5 cells without a match.",
    )
    args = parser.parse_args()

    dataset_path = str(Path(args.dataset_path).resolve())
    params_dir = Path(args.models_dir).resolve()
    if not params_dir.is_dir():
        raise NotADirectoryError(f"--models_dir is not a directory: {params_dir}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    params_by_cell = _build_pickle_map(params_dir, args.pickle_regex)
    if len(params_by_cell) == 0:
        raise ValueError(f"No pickles matched --pickle_regex in {params_dir}")

    all_cell_keys = _iter_cells(dataset_path)

    missing: list[str] = []
    cell_keys: list[str] = []
    for cell_key in all_cell_keys:
        ck, *_ = _read_cell_inputs(dataset_path, cell_key)
        if ck in params_by_cell:
            cell_keys.append(cell_key)
        else:
            missing.append(ck.cell_id_piece)

    if missing and not args.allow_missing_cells:
        raise FileNotFoundError(
            "Missing params pickles for the following H5 cells (cell_id-piece):\n"
            + "\n".join(sorted(missing))
        )
    if not cell_keys:
        raise ValueError("No H5 cells have matching param pickles in --models_dir")

    num_workers = int(args.num_workers)
    if num_workers <= 0:
        raise ValueError("--num_workers must be >= 1")

    print(f"Found {len(all_cell_keys)} cells in H5")
    print(f"Matched {len(params_by_cell)} param pickles in {params_dir}")
    if missing:
        print(f"Skipping {len(missing)} cells without param pickles: {', '.join(sorted(missing))}")
    print(f"Writing outputs to {out_dir}")
    print(f"Using num_workers={num_workers}")

    infer_func = partial(
        _infer_one_cell,
        dataset_path=dataset_path,
        params_by_cell=params_by_cell,
        out_dir=out_dir,
    )

    successes = 0
    failures = 0
    with mp.Pool(processes=num_workers) as pool:
        for cell_key, ok, msg in pool.imap_unordered(infer_func, cell_keys):
            if ok:
                successes += 1
            else:
                failures += 1
            print(f"{'OK' if ok else 'FAIL'} {cell_key}: {msg}", flush=True)

    print(f"\nDone. successes={successes} failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())

