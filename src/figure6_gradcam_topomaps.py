#!/usr/bin/env python3
"""Reproduce Figure 6: group-average Grad-CAM scalp topomaps.

Example
-------
# NPZ input (recommended): one file per subject with left_mi/right_mi arrays
# python src/figure6_gradcam_topomaps.py \
#   --gradcam-dir BCI_Research/results/gradcam \
#   --positions-csv BCI_Research/results/gradcam/channel_positions.csv \
#   --accuracy-csv BCI_Research/results/ablation/ablation_results.csv \
#   --output-dir BCI_Research/results/figure6 --low-n 5 \
#   --scale-mode symmetric --clip-percentile 99

Input contract
--------------
1. Grad-CAM directory: one ``sXX.npz`` or ``sXX.npy`` per subject.
   NPZ (recommended) must contain ``left_mi`` and ``right_mi``. Each value may
   be ``(n_channels,)`` or have extra trial/time dimensions; all dimensions
   except the channel dimension are averaged. Use ``--channel-axis`` when the
   channel dimension is not the last dimension.
   NPY must contain a two-class array. By default class axis 0 means
   ``[Left MI, Right MI]``; change it with ``--class-axis``.
2. Channel-position CSV: columns ``channel,x,y`` (optional ``z``). Rows must be
   in the same channel order as the Grad-CAM arrays. Coordinates should be 2-D
   head/topomap coordinates with x<0 on the left and x>0 on the right.
3. Accuracy CSV: a subject column (auto-detected from subject/sid/subject_id)
   and Fusion accuracy column (auto-detected from acc_fusion/fusion_accuracy/
   fusion). Override with ``--subject-column`` and ``--fusion-column``.

Outputs are ``figure6_gradcam_topomaps.png`` (600 dpi), PDF, and
``figure6_summary.csv`` in ``--output-dir``.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import mne
except ImportError as exc:  # pragma: no cover - dependency check
    raise SystemExit("mne is required. Install it with: pip install mne") from exc


SEED = 42
HIGH_SUBJECTS = ("s03", "s14", "s41", "s43", "s48")
BIAS_SUBJECTS = ("s01", "s05", "s07")
CLASS_NAMES = ("Left MI", "Right MI")
NPZ_KEYS = (("left_mi", "left", "class_0"), ("right_mi", "right", "class_1"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Figure 6 group-average Grad-CAM scalp topomaps."
    )
    parser.add_argument("--gradcam-dir", type=Path, required=True)
    parser.add_argument("--positions-csv", type=Path, required=True)
    parser.add_argument("--accuracy-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--low-n", type=int, default=5)
    parser.add_argument("--subject-column", default=None)
    parser.add_argument("--fusion-column", default=None)
    parser.add_argument("--class-axis", type=int, default=0)
    parser.add_argument(
        "--channel-axis", type=int, default=-1,
        help="Channel axis after selecting a class (default: last axis).",
    )
    parser.add_argument(
        "--scale-mode", choices=("symmetric", "minmax"), default="symmetric"
    )
    parser.add_argument(
        "--clip-percentile", type=float, default=99.0,
        help="Percentile used for color-limit clipping; use 100 to disable.",
    )
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def normalize_subject(value: object) -> str:
    """Convert 3, '03', and 's03' to the canonical ID 's03'."""
    text = str(value).strip().lower()
    match = re.fullmatch(r"s?0*(\d+)(?:\.0+)?", text)
    if not match:
        raise ValueError(f"Cannot parse subject ID: {value!r}")
    return f"s{int(match.group(1)):02d}"


def choose_column(df: pd.DataFrame, requested: str | None, aliases: Sequence[str], kind: str) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"{kind} column {requested!r} not found; columns={list(df.columns)}")
        return requested
    lower_to_original = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias in lower_to_original:
            return str(lower_to_original[alias])
    raise ValueError(f"Could not auto-detect {kind} column; use the corresponding CLI option")


def select_low_performers(
    csv_path: Path, low_n: int, subject_column: str | None, fusion_column: str | None
) -> Tuple[List[str], pd.DataFrame]:
    if low_n < 1:
        raise ValueError("--low-n must be at least 1")
    df = pd.read_csv(csv_path)
    sid_col = choose_column(df, subject_column, ("subject", "sid", "subject_id"), "subject")
    acc_col = choose_column(
        df, fusion_column, ("acc_fusion", "fusion_accuracy", "fusion"), "Fusion accuracy"
    )
    work = df[[sid_col, acc_col]].copy()
    work["subject"] = work[sid_col].map(normalize_subject)
    work["fusion_accuracy"] = pd.to_numeric(work[acc_col], errors="coerce")
    work = work.dropna(subset=["fusion_accuracy"])
    excluded = set(HIGH_SUBJECTS) | set(BIAS_SUBJECTS)
    eligible = work.loc[~work["subject"].isin(excluded)]
    eligible = eligible.sort_values(["fusion_accuracy", "subject"], kind="stable")
    lows = eligible.drop_duplicates("subject").head(low_n)["subject"].tolist()
    if len(lows) < low_n:
        logging.warning("Requested %d low performers, but only %d were eligible", low_n, len(lows))
    return lows, work[["subject", "fusion_accuracy"]]


def load_positions(path: Path) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(path)
    lookup = {str(c).strip().lower(): c for c in df.columns}
    missing = [c for c in ("channel", "x", "y") if c not in lookup]
    if missing:
        raise ValueError(f"Position CSV is missing columns: {missing}")
    names = df[lookup["channel"]].astype(str).str.strip().tolist()
    pos = df[[lookup["x"], lookup["y"]]].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if len(names) < 4 or len(set(names)) != len(names):
        raise ValueError("Position CSV needs at least four uniquely named channels")
    if not np.isfinite(pos).all():
        raise ValueError("Channel positions contain NaN or infinite values")
    if np.ptp(pos[:, 0]) == 0 or np.ptp(pos[:, 1]) == 0:
        raise ValueError("Channel positions must span both x and y dimensions")
    return names, pos


def reduce_to_channels(array: np.ndarray, n_channels: int, channel_axis: int) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 0:
        raise ValueError("Grad-CAM value is scalar")
    axis = channel_axis if channel_axis >= 0 else arr.ndim + channel_axis
    if not 0 <= axis < arr.ndim:
        raise ValueError(f"Invalid channel axis {channel_axis} for shape {arr.shape}")
    if arr.shape[axis] != n_channels:
        candidates = [i for i, size in enumerate(arr.shape) if size == n_channels]
        if len(candidates) == 1:
            logging.warning("Using inferred channel axis %d instead of %d for shape %s", candidates[0], axis, arr.shape)
            axis = candidates[0]
        else:
            raise ValueError(f"Expected {n_channels} channels in shape {arr.shape}; candidates={candidates}")
    arr = np.moveaxis(arr, axis, -1).reshape(-1, n_channels)
    if not np.isfinite(arr).any():
        raise ValueError("Grad-CAM array has no finite values")
    return np.nanmean(arr, axis=0)


def _npz_value(archive: np.lib.npyio.NpzFile, aliases: Iterable[str]) -> np.ndarray:
    lower_keys = {key.lower(): key for key in archive.files}
    for alias in aliases:
        if alias in lower_keys:
            return archive[lower_keys[alias]]
    raise KeyError(f"None of keys {tuple(aliases)} found; available={archive.files}")


def load_subject_file(
    path: Path, n_channels: int, class_axis: int, channel_axis: int
) -> np.ndarray:
    """Return a (2, n_channels) array."""
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            values = [_npz_value(archive, aliases) for aliases in NPZ_KEYS]
    elif path.suffix.lower() == ".npy":
        raw = np.load(path, allow_pickle=False)
        axis = class_axis if class_axis >= 0 else raw.ndim + class_axis
        if not 0 <= axis < raw.ndim or raw.shape[axis] != 2:
            raise ValueError(f"NPY class axis {class_axis} must have size 2; shape={raw.shape}")
        values = [np.take(raw, idx, axis=axis) for idx in range(2)]
    else:
        raise ValueError(f"Unsupported extension: {path.suffix}")
    return np.stack([reduce_to_channels(v, n_channels, channel_axis) for v in values])


def find_subject_file(directory: Path, subject: str) -> Path | None:
    candidates = [directory / f"{subject}.npz", directory / f"{subject}.npy"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    by_lower_name = {p.name.lower(): p for p in directory.iterdir() if p.is_file()}
    for candidate in candidates:
        if candidate.name.lower() in by_lower_name:
            return by_lower_name[candidate.name.lower()]
    return None


def load_groups(
    directory: Path, requested: Mapping[str, Sequence[str]], n_channels: int,
    class_axis: int, channel_axis: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]], Dict[str, List[str]]]:
    group_means: Dict[str, np.ndarray] = {}
    loaded: Dict[str, List[str]] = {}
    skipped: Dict[str, List[str]] = {}
    for group, subjects in requested.items():
        arrays, loaded[group], skipped[group] = [], [], []
        for subject in subjects:
            path = find_subject_file(directory, subject)
            if path is None:
                logging.warning("[%s] Missing Grad-CAM file for %s; skipping", group, subject)
                skipped[group].append(subject)
                continue
            try:
                arrays.append(load_subject_file(path, n_channels, class_axis, channel_axis))
                loaded[group].append(subject)
            except Exception:
                logging.error("[%s] Failed to load %s; skipping\n%s", group, subject, traceback.format_exc())
                skipped[group].append(subject)
        if not arrays:
            raise RuntimeError(f"No usable Grad-CAM files for group {group!r}")
        group_means[group] = np.nanmean(np.stack(arrays), axis=0)
    return group_means, loaded, skipped


def color_limits(means: Mapping[str, np.ndarray], mode: str, percentile: float) -> Tuple[float, float]:
    if not 0 < percentile <= 100:
        raise ValueError("--clip-percentile must be in (0, 100]")
    values = np.concatenate([v.ravel() for v in means.values()])
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("All group-average values are non-finite")
    if mode == "symmetric":
        vmax = float(np.percentile(np.abs(values), percentile))
        vmax = vmax if vmax > 0 else float(np.finfo(float).eps)
        return -vmax, vmax
    tail = (100.0 - percentile) / 2.0
    vmin, vmax = np.percentile(values, [tail, 100.0 - tail]).astype(float)
    if vmin == vmax:
        vmax = vmin + float(np.finfo(float).eps)
    return vmin, vmax


def make_summary(
    means: Mapping[str, np.ndarray], loaded: Mapping[str, Sequence[str]], positions: np.ndarray
) -> pd.DataFrame:
    left_mask, right_mask = positions[:, 0] < 0, positions[:, 0] > 0
    if not left_mask.any() or not right_mask.any():
        raise ValueError("Lateralization requires channels with both negative and positive x coordinates")
    rows = []
    for group, values in means.items():
        metrics = []
        for class_idx, class_name in enumerate(CLASS_NAMES):
            data = values[class_idx]
            left_mean = float(np.nanmean(data[left_mask]))
            right_mean = float(np.nanmean(data[right_mask]))
            denom = abs(left_mean) + abs(right_mean)
            raw_li = (left_mean - right_mean) / denom if denom > 0 else 0.0
            metrics.append((left_mean, right_mean, raw_li))
        # Positive switching strength means the expected contralateral change:
        # right-dominant Left MI (negative LI) -> left-dominant Right MI (positive LI).
        switching = (metrics[1][2] - metrics[0][2]) / 2.0
        for class_idx, class_name in enumerate(CLASS_NAMES):
            left_mean, right_mean, raw_li = metrics[class_idx]
            rows.append({
                "group": group,
                "class": class_name,
                "n": len(loaded[group]),
                "subjects": ";".join(loaded[group]),
                "mean_activation": float(np.nanmean(values[class_idx])),
                "left_roi_mean": left_mean,
                "right_roi_mean": right_mean,
                "lateralization_index": raw_li,
                "switching_strength": switching,
            })
    return pd.DataFrame(rows)


def plot_figure(
    means: Mapping[str, np.ndarray], loaded: Mapping[str, Sequence[str]], positions: np.ndarray,
    limits: Tuple[float, float], cmap: str, output_dir: Path, dpi: int,
) -> None:
    groups = ("High performers", "Low performers", "Bias subjects")
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.0), constrained_layout=True)
    image = None
    for row, class_name in enumerate(CLASS_NAMES):
        for col, group in enumerate(groups):
            ax = axes[row, col]
            image, _ = mne.viz.plot_topomap(
                means[group][row], positions, axes=ax, show=False, cmap=cmap,
                vlim=limits, sensors=True, outlines="head", contours=6,
                extrapolate="head", image_interp="cubic",
            )
            ax.set_title(f"{group} (n={len(loaded[group])})", fontsize=11, pad=7)
            if col == 0:
                ax.set_ylabel(class_name, fontsize=12, fontweight="bold", labelpad=13)
    if image is None:
        raise RuntimeError("No topomap was drawn")
    colorbar = fig.colorbar(image, ax=axes, orientation="vertical", shrink=0.78, pad=0.025)
    colorbar.set_label("Mean Grad-CAM activation", fontsize=10)
    fig.suptitle("Class-dependent Grad-CAM scalp topographies", fontsize=14, fontweight="bold")
    fig.savefig(output_dir / "figure6_gradcam_topomaps.png", dpi=max(600, dpi), bbox_inches="tight")
    fig.savefig(output_dir / "figure6_gradcam_topomaps.pdf", bbox_inches="tight")
    plt.close(fig)


def log_reproduction(
    requested: Mapping[str, Sequence[str]], loaded: Mapping[str, Sequence[str]],
    skipped: Mapping[str, Sequence[str]], summary: pd.DataFrame, limits: Tuple[float, float],
) -> None:
    logging.info("Shared color limits: [%.6g, %.6g]", *limits)
    for group in requested:
        logging.info("%s requested: %s", group, ", ".join(requested[group]))
        logging.info("%s loaded (%d): %s", group, len(loaded[group]), ", ".join(loaded[group]))
        logging.info("%s skipped (%d): %s", group, len(skipped[group]), ", ".join(skipped[group]) or "none")
        subset = summary[summary["group"] == group]
        for row in subset.to_dict("records"):
            logging.info(
                "%s | %s: LI(L-R)=%.4f, mean=%.4f", group, row["class"],
                row["lateralization_index"], row["mean_activation"],
            )
        logging.info("%s switching strength: %.4f", group, subset["switching_strength"].iloc[0])


def main() -> int:
    args = parse_args()
    np.random.seed(SEED)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        channel_names, positions = load_positions(args.positions_csv)
        logging.info("Loaded %d channel positions: %s", len(channel_names), ", ".join(channel_names))
        lows, _ = select_low_performers(
            args.accuracy_csv, args.low_n, args.subject_column, args.fusion_column
        )
        requested = {
            "High performers": list(HIGH_SUBJECTS),
            "Low performers": lows,
            "Bias subjects": list(BIAS_SUBJECTS),
        }
        means, loaded, skipped = load_groups(
            args.gradcam_dir, requested, len(channel_names), args.class_axis, args.channel_axis
        )
        limits = color_limits(means, args.scale_mode, args.clip_percentile)
        summary = make_summary(means, loaded, positions)
        summary.to_csv(args.output_dir / "figure6_summary.csv", index=False, float_format="%.8f")
        plot_figure(means, loaded, positions, limits, args.cmap, args.output_dir, args.dpi)
        log_reproduction(requested, loaded, skipped, summary, limits)
        logging.info("Saved Figure 6 outputs to %s", args.output_dir.resolve())
        return 0
    except Exception:
        logging.critical("Figure 6 generation failed\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
