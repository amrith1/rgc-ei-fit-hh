"""Evaluate multielectrode inference outputs against triplet H5 ground truth."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

MODEL_PROB_KEY = "triplet_probs_adjusted_experiment"
SUPERPOSITION_PROB_KEY = "superposition_model"
INDEPENDENT_PROB_KEY = "independent_model"
SUPERPOSITION_PREFIX = "superposition"
INDEPENDENT_PREFIX = "independent"


@dataclass
class ClassificationMetrics:
    tp: int
    tn: int
    fp: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def accuracy(self) -> float:
        return 100.0 * (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def balanced_accuracy(self) -> float:
        sens = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        spec = self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0.0
        return 100.0 * (sens + spec) / 2.0

    @property
    def fpr(self) -> float:
        return 100.0 * self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0.0

    @property
    def fnr(self) -> float:
        return 100.0 * self.fn / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    def to_dict(self, prefix: str = "") -> dict[str, float | int]:
        p = f"{prefix}_" if prefix else ""
        return {
            f"{p}accuracy": self.accuracy,
            f"{p}balanced_accuracy": self.balanced_accuracy,
            f"{p}fpr": self.fpr,
            f"{p}fnr": self.fnr,
            f"{p}tp": self.tp,
            f"{p}tn": self.tn,
            f"{p}fp": self.fp,
            f"{p}fn": self.fn,
        }


def _confusion_counts(pred_binary: np.ndarray, gt_binary: np.ndarray) -> ClassificationMetrics:
    pred = (np.asarray(pred_binary) > 0.5).astype(np.int64)
    gt = (np.asarray(gt_binary) > 0.5).astype(np.int64)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    tp = int(np.sum((pred == 1) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    return ClassificationMetrics(tp=tp, tn=tn, fp=fp, fn=fn)


def _add_counts(a: ClassificationMetrics, b: ClassificationMetrics) -> ClassificationMetrics:
    return ClassificationMetrics(
        tp=a.tp + b.tp,
        tn=a.tn + b.tn,
        fp=a.fp + b.fp,
        fn=a.fn + b.fn,
    )


def _load_gt_flipped_probs(cell_group: h5py.Group) -> np.ndarray:
    if "probs_flipped" not in cell_group:
        raise KeyError("Missing dataset probs_flipped in H5 cell group")
    return np.asarray(cell_group["probs_flipped"][:])


def _inference_path_for_cell(inference_dir: Path, cell_id: int, piece: str) -> Path:
    return inference_dir / f"cell_{cell_id}-{piece}_triplet_probs.pkl"


def mean_per_cell_summary(per_cell_df: pd.DataFrame, prefix: str) -> dict[str, float]:
    """Average per-cell classification metrics (values in percent, 0--100)."""
    return {
        "accuracy": float(per_cell_df[f"{prefix}_accuracy"].mean()),
        "balanced_accuracy": float(per_cell_df[f"{prefix}_balanced_accuracy"].mean()),
        "fpr": float(per_cell_df[f"{prefix}_fpr"].mean()),
        "fnr": float(per_cell_df[f"{prefix}_fnr"].mean()),
    }


def format_mean_per_cell_block(label: str, metrics: dict[str, float]) -> str:
    return (
        f"{label}: acc={metrics['accuracy']:.3f}%  "
        f"bal_acc={metrics['balanced_accuracy']:.3f}%  "
        f"fpr={metrics['fpr']:.3f}%  fnr={metrics['fnr']:.3f}%"
    )


def table_metrics(
    mean_per_cell: dict[str, float],
    pooled: dict[str, float | int],
) -> dict[str, float]:
    return {
        "accuracy": float(mean_per_cell["accuracy"]),
        "balanced_accuracy": float(pooled["balanced_accuracy"]),
        "fpr": float(pooled["fpr"]),
        "fnr": float(pooled["fnr"]),
    }


def format_table_metrics_block(label: str, metrics: dict[str, float]) -> str:
    return (
        f"{label}: acc={metrics['accuracy']:.3f}%  "
        f"bal_acc={metrics['balanced_accuracy']:.3f}%  "
        f"fpr={metrics['fpr']:.3f}%  fnr={metrics['fnr']:.3f}%"
    )


def print_evaluation_summary(result: dict[str, Any], *, title: str | None = None) -> None:
    """Print summary metrics for model and baselines."""
    if title:
        print(title)
    print(f"Cells evaluated: {result['n_cells']}")
    if result.get("missing_cells"):
        print(f"Missing cells ({len(result['missing_cells'])}): {', '.join(result['missing_cells'])}")
    print()
    print("Summary:")
    for key, label in (
        ("model", "Model"),
        (SUPERPOSITION_PREFIX, "Superposition"),
        (INDEPENDENT_PREFIX, "Independent"),
    ):
        summary = table_metrics(result["mean_per_cell"][key], result["pooled"][key])
        print("  " + format_table_metrics_block(label, summary))
    print()
    print("Mean per-cell metrics:")
    for key, label in (
        ("model", "Model"),
        (SUPERPOSITION_PREFIX, "Superposition"),
        (INDEPENDENT_PREFIX, "Independent"),
    ):
        print("  " + format_mean_per_cell_block(label, result["mean_per_cell"][key]))
    print()
    print("Pooled metrics (all stimuli concatenated):")
    for key, label in (
        ("model", "Model"),
        (SUPERPOSITION_PREFIX, "Superposition"),
        (INDEPENDENT_PREFIX, "Independent"),
    ):
        pooled = result["pooled"][key]
        print(
            f"  {label}: acc={pooled['accuracy']:.3f}%  "
            f"bal_acc={pooled['balanced_accuracy']:.3f}%  "
            f"fpr={pooled['fpr']:.3f}%  fnr={pooled['fnr']:.3f}%"
        )


def evaluate_multielectrode_inference(
    dataset_path: str | Path,
    inference_dir: str | Path,
    *,
    require_all_cells: bool = True,
) -> dict[str, Any]:
    """Evaluate one multielectrode_inference/ folder against triplet H5 ground truth."""
    dataset_path = Path(dataset_path)
    inference_dir = Path(inference_dir)
    if not inference_dir.is_dir():
        raise NotADirectoryError(f"inference_dir not found: {inference_dir}")

    per_cell_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    pooled_model = ClassificationMetrics(0, 0, 0, 0)
    pooled_superposition = ClassificationMetrics(0, 0, 0, 0)
    pooled_independent = ClassificationMetrics(0, 0, 0, 0)

    with h5py.File(dataset_path, "r") as h5f:
        for cell_key in h5f["cells"].keys():
            g = h5f["cells"][cell_key]
            cell_id = int(g.attrs["cell_id_piece"])
            piece = str(g.attrs["piece"])
            cell_id_piece = f"{cell_id}-{piece}"

            inf_path = _inference_path_for_cell(inference_dir, cell_id, piece)
            if not inf_path.exists():
                missing.append(cell_id_piece)
                continue

            gt_binary = (_load_gt_flipped_probs(g) > 0.5).astype(np.float64)
            with inf_path.open("rb") as f:
                inf = pickle.load(f)

            model_counts = _confusion_counts(inf[MODEL_PROB_KEY], gt_binary)
            superposition_counts = _confusion_counts(inf[SUPERPOSITION_PROB_KEY], gt_binary)
            independent_counts = _confusion_counts(inf[INDEPENDENT_PROB_KEY], gt_binary)

            row: dict[str, Any] = {
                "cell_id_piece": cell_id_piece,
                "cell_id": cell_id,
                "piece": piece,
                "n_stimuli": int(len(gt_binary)),
            }
            row.update(model_counts.to_dict(prefix="model"))
            row.update(superposition_counts.to_dict(prefix=SUPERPOSITION_PREFIX))
            row.update(independent_counts.to_dict(prefix=INDEPENDENT_PREFIX))
            per_cell_rows.append(row)

            pooled_model = _add_counts(pooled_model, model_counts)
            pooled_superposition = _add_counts(pooled_superposition, superposition_counts)
            pooled_independent = _add_counts(pooled_independent, independent_counts)

    if require_all_cells and missing:
        raise FileNotFoundError(
            "Missing inference pickles for cells:\n" + "\n".join(sorted(missing))
        )
    if not per_cell_rows:
        raise ValueError("No cells evaluated (check paths)")

    per_cell_df = pd.DataFrame(per_cell_rows)
    mean_per_cell = {
        "model": mean_per_cell_summary(per_cell_df, "model"),
        SUPERPOSITION_PREFIX: mean_per_cell_summary(per_cell_df, SUPERPOSITION_PREFIX),
        INDEPENDENT_PREFIX: mean_per_cell_summary(per_cell_df, INDEPENDENT_PREFIX),
    }

    return {
        "per_cell_df": per_cell_df,
        "pooled": {
            "model": pooled_model.to_dict(),
            SUPERPOSITION_PREFIX: pooled_superposition.to_dict(),
            INDEPENDENT_PREFIX: pooled_independent.to_dict(),
        },
        "mean_per_cell": mean_per_cell,
        # Backward-compatible alias for model accuracy only.
        "mean_per_cell_accuracy": mean_per_cell["model"]["accuracy"],
        "n_cells": len(per_cell_df),
        "missing_cells": missing,
    }


def _table_metrics_from_result(result: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        prefix: table_metrics(result["mean_per_cell"][prefix], result["pooled"][prefix])
        for prefix in ("model", SUPERPOSITION_PREFIX, INDEPENDENT_PREFIX)
    }


def save_evaluation_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
    *,
    partial: bool = False,
    n_cells_expected: int | None = None,
) -> None:
    """Write per_cell_metrics.csv and evaluation_metrics.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_cell_df = result["per_cell_df"]
    per_cell_df.to_csv(output_dir / "per_cell_metrics.csv", index=False)

    table_metrics = _table_metrics_from_result(result)
    summary = {
        "table_metrics": table_metrics,
        "n_cells": result["n_cells"],
        "mean_per_cell": result["mean_per_cell"],
        "mean_per_cell_accuracy": result["mean_per_cell_accuracy"],
        "pooled_model": result["pooled"]["model"],
        "pooled_superposition": result["pooled"][SUPERPOSITION_PREFIX],
        "pooled_independent": result["pooled"][INDEPENDENT_PREFIX],
        "missing_cells": result["missing_cells"],
        "partial": partial,
    }
    if n_cells_expected is not None:
        summary["n_cells_expected"] = n_cells_expected

    metrics_path = output_dir / "evaluation_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multielectrode inference pickles against triplet H5 ground truth.",
    )
    parser.add_argument("--dataset_path", type=Path, required=True, help="Path to triplet HDF5 dataset.")
    parser.add_argument(
        "--inference_dir",
        type=Path,
        required=True,
        help="Directory of per-cell inference pickles written by multielectrode_inference.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for per_cell_metrics.csv and evaluation_metrics.json (created if missing).",
    )
    parser.add_argument(
        "--allow_missing_cells",
        action="store_true",
        help="Evaluate available cells only (do not fail on missing pickles).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = evaluate_multielectrode_inference(
        args.dataset_path,
        args.inference_dir,
        require_all_cells=not args.allow_missing_cells,
    )
    print_evaluation_summary(
        result,
        title=f"Multielectrode evaluation: {args.inference_dir}",
    )
    save_evaluation_outputs(result, args.output_dir, partial=args.allow_missing_cells)
    print(f"\nWrote metrics to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
