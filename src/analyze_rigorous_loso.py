"""Analyze complete paired rigorous_loso runs without treating seeds as subjects."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "BCI_Research/results/rigorous_loso"
CONDITIONS = ("dirty_train", "clean_train")
MODELS = ("eeg_only", "emg_only", "fusion")
METRICS = ("accuracy", "kappa")


def load_complete(out_dir, seeds, expected_subjects=49):
    frames = []
    for condition in CONDITIONS:
        for model in MODELS:
            for seed in seeds:
                path = out_dir / f"{condition}_{model}_seed{seed}.csv"
                if not path.is_file():
                    raise FileNotFoundError(f"Required run is missing: {path}")
                df = pd.read_csv(path)
                if (len(df) != expected_subjects or df.sid.nunique() != expected_subjects
                        or set(df.condition) != {condition}
                        or set(df.model_type) != {model}
                        or set(df.seed) != {seed}):
                    raise ValueError(f"Incomplete or inconsistent run: {path}")
                frames.append(df)
    all_rows = pd.concat(frames, ignore_index=True)
    ids = {tuple(sorted(f.sid)) for f in frames}
    if len(ids) != 1:
        raise ValueError("Conditions do not contain the same test subjects")
    grouped = all_rows.groupby(["seed", "sid"])
    if grouped.validation_sid.nunique().max() != 1:
        raise ValueError("Paired runs selected different validation subjects")
    if grouped.n_trials.nunique().max() != 1:
        raise ValueError("Paired runs used different clean test-trial counts")
    return all_rows


def paired_result(df, numerator, denominator, metric, bootstrap=10000):
    """Average each subject across seeds, then test paired subject differences."""
    per_subject = df.groupby(["condition", "model_type", "sid"])[metric].mean()
    a = per_subject.xs(numerator[:2]).sort_index()
    b = per_subject.xs(denominator[:2]).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("Subject pairing failed")
    delta = (a - b).to_numpy(dtype=float)
    p = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
    rng = np.random.default_rng(42)
    draws = rng.choice(delta, size=(bootstrap, len(delta)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"n_subjects": len(delta), "mean_a": float(a.mean()),
            "mean_b": float(b.mean()), "mean_paired_delta": float(delta.mean()),
            "bootstrap_95ci": [float(low), float(high)], "wilcoxon_p": p,
            "positive_subjects": int((delta > 0).sum()),
            "negative_subjects": int((delta < 0).sum()),
            "zero_subjects": int((delta == 0).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or args.bootstrap < 100:
        parser.error("Seeds must be unique and bootstrap must be at least 100")
    df = load_complete(args.out_dir, args.seeds)

    analyses = {}
    for metric in METRICS:
        for model in MODELS:
            key = f"training_cleanup_{model}_{metric}"
            analyses[key] = paired_result(
                df, ("clean_train", model), ("dirty_train", model),
                metric, args.bootstrap)
        for condition in CONDITIONS:
            key = f"fusion_gain_{condition}_{metric}"
            analyses[key] = paired_result(
                df, (condition, "fusion"), (condition, "eeg_only"),
                metric, args.bootstrap)

    # Difference in fusion advantage attributable to training-trial cleanup.
    for metric in METRICS:
        pivot = df.groupby(["sid", "condition", "model_type"])[metric].mean().unstack(
            ["condition", "model_type"])
        dirty = pivot[("dirty_train", "fusion")] - pivot[("dirty_train", "eeg_only")]
        clean = pivot[("clean_train", "fusion")] - pivot[("clean_train", "eeg_only")]
        delta = (clean - dirty).to_numpy(dtype=float)
        rng = np.random.default_rng(42)
        draws = rng.choice(delta, size=(args.bootstrap, len(delta)), replace=True).mean(axis=1)
        analyses[f"difference_in_fusion_gain_{metric}"] = {
            "n_subjects": len(delta), "dirty_gain_mean": float(dirty.mean()),
            "clean_gain_mean": float(clean.mean()),
            "change_mean": float(delta.mean()),
            "bootstrap_95ci": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
            "wilcoxon_p": float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0,
        }

    result = {"protocol": "subject-disjoint validation; clean outer test in both conditions",
              "seeds": args.seeds, "analyses": analyses}
    output = args.out_dir / "paired_analysis.json"
    output.write_text(json.dumps(result, indent=2))
    print(f"Saved {output}")
    for name, value in analyses.items():
        effect = value.get("mean_paired_delta", value.get("change_mean"))
        print(f"{name}: delta={effect:+.4f}, 95% CI={value['bootstrap_95ci']}, "
              f"p={value['wilcoxon_p']:.4g}")


if __name__ == "__main__":
    main()
