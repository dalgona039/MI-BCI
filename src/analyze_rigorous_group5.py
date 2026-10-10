"""Analyze paired five-fold results with fold-cluster uncertainty.

The five independently trained outer folds, rather than 49 test subjects,
are the resampling units. With only five clusters, intervals are exploratory.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_rigorous_loso import CONDITIONS, METRICS, MODELS, load_complete

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "BCI_Research/results/rigorous_group5"


def summarize_delta(delta, fold_by_sid, bootstrap):
    """Average paired subject effects; resample whole outer folds for CI."""
    if not delta.index.equals(fold_by_sid.index):
        raise ValueError("Subject and fold order differs")
    fold_ids = sorted(fold_by_sid.unique())
    if fold_ids != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected five outer folds, found {fold_ids}")
    values = [delta[fold_by_sid == fold].to_numpy(dtype=float) for fold in fold_ids]
    rng = np.random.default_rng(42)
    draws = np.empty(bootstrap, dtype=float)
    for i in range(bootstrap):
        chosen = rng.integers(0, 5, size=5)
        draws[i] = np.concatenate([values[j] for j in chosen]).mean()
    return {
        "n_subjects": len(delta), "n_outer_folds": 5,
        "mean_paired_delta": float(delta.mean()),
        "fold_mean_deltas": [float(part.mean()) for part in values],
        "fold_cluster_bootstrap_95ci": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "positive_subjects": int((delta > 0).sum()),
        "negative_subjects": int((delta < 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or args.bootstrap < 100:
        parser.error("Seeds must be unique and bootstrap must be at least 100")
    df = load_complete(args.out_dir, args.seeds)
    if "outer_fold" not in df.columns:
        raise ValueError("Expected group5 results with outer_fold")
    if df.groupby("sid").outer_fold.nunique().max() != 1:
        raise ValueError("Outer folds differ between paired runs")
    folds = df.groupby("sid").outer_fold.first().sort_index()
    if sorted(df.outer_fold.unique()) != [1, 2, 3, 4, 5]:
        raise ValueError("Incomplete outer folds")

    per_subject = df.groupby(["condition", "model_type", "sid"])[list(METRICS)].mean()
    analyses = {}
    for metric in METRICS:
        def pair(a, b):
            first = per_subject.xs(a)[metric].sort_index()
            second = per_subject.xs(b)[metric].sort_index()
            if not first.index.equals(second.index):
                raise ValueError("Unpaired test subjects")
            return summarize_delta(first - second, folds, args.bootstrap)

        for model in MODELS:
            analyses[f"training_cleanup_{model}_{metric}"] = pair(
                ("clean_train", model), ("dirty_train", model))
        for condition in CONDITIONS:
            analyses[f"fusion_gain_{condition}_{metric}"] = pair(
                (condition, "fusion"), (condition, "eeg_only"))
        clean_gain = (per_subject.xs(("clean_train", "fusion"))[metric]
                      - per_subject.xs(("clean_train", "eeg_only"))[metric]).sort_index()
        dirty_gain = (per_subject.xs(("dirty_train", "fusion"))[metric]
                      - per_subject.xs(("dirty_train", "eeg_only"))[metric]).sort_index()
        analyses[f"difference_in_fusion_gain_{metric}"] = summarize_delta(
            clean_gain - dirty_gain, folds, args.bootstrap)

    result = {
        "protocol": "fixed five-fold subject-disjoint evaluation; clean test in both conditions",
        "seeds": args.seeds,
        "uncertainty": "Resample five outer folds. Exploratory with only five clusters; do not treat 49 test subjects as independent fitted models.",
        "analyses": analyses,
    }
    output = args.out_dir / "paired_analysis_group5.json"
    output.write_text(json.dumps(result, indent=2))
    print(f"Saved {output}")
    for name, value in analyses.items():
        print(f"{name}: delta={value['mean_paired_delta']:+.4f}, "
              f"fold-cluster 95% CI={value['fold_cluster_bootstrap_95ci']}")


if __name__ == "__main__":
    main()
