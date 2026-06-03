"""
statistical_tests.py — Wilcoxon Signed-rank Test + ITR Bootstrap CI
====================================================================
ablation_results.csv 를 입력으로 받아 조건 간 통계 검정을 수행합니다.

사용법:
  # 로컬 (.venv에 scipy + pandas 설치됨)
  python src/statistical_tests.py

  # Colab (ablation 셀 5 완료 후)
  python /content/statistical_tests.py \\
      --drive_root /content/drive/MyDrive/BCI_Research

출력:
  results/ablation/wilcoxon_results.json   ← Wilcoxon 검정 결과
  results/ablation/itr_bootstrap.json      ← ITR 95% CI bootstrap
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SEED = 42


# ════════════════════════════════════════════════════════════════
#  Wilcoxon effect size
# ════════════════════════════════════════════════════════════════

def _wilcoxon_r(x: np.ndarray, y: np.ndarray):
    """Wilcoxon signed-rank test + effect size r = |Z| / sqrt(N_nonzero)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    diff  = x - y
    nz    = int((diff != 0).sum())
    if nz < 2:
        return None, None, None, None

    result = wilcoxon(x, y, alternative="two-sided")
    stat   = float(result.statistic)
    p      = float(result.pvalue)

    # Z via normal approximation of W+
    mu    = nz * (nz + 1) / 4.0
    sigma = np.sqrt(nz * (nz + 1) * (2 * nz + 1) / 24.0)
    z     = (stat - mu) / sigma
    r     = abs(z) / np.sqrt(nz)

    return stat, p, float(z), float(r)


# ════════════════════════════════════════════════════════════════
#  ITR Bootstrap 95% CI
# ════════════════════════════════════════════════════════════════

def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000,
                  ci: float = 0.95, seed: int = SEED):
    """Percentile bootstrap CI for the mean."""
    rng   = np.random.default_rng(seed)
    vals  = np.asarray(values, float)
    n     = len(vals)
    boot  = np.array([
        rng.choice(vals, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = 1.0 - ci
    lo    = float(np.percentile(boot, 100 * alpha / 2))
    hi    = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return float(vals.mean()), float(vals.std(ddof=1)), lo, hi


# ════════════════════════════════════════════════════════════════
#  메인 분석
# ════════════════════════════════════════════════════════════════

def run(drive_root: str):
    abl_dir  = os.path.join(drive_root, "results", "ablation")
    csv_path = os.path.join(abl_dir, "ablation_results.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"ablation_results.csv 없음: {csv_path}\n"
            "Colab 셀 5 (--merge) 를 먼저 실행하세요."
        )

    df = pd.read_csv(csv_path)
    N  = len(df)
    print(f"  피험자 수: N = {N}")

    CONDITIONS = ["eeg_only", "emg_only", "fusion"]
    PAIRS = [
        ("eeg_only", "fusion",   "EEG-only vs Fusion"),
        ("emg_only", "fusion",   "sEMG-only vs Fusion"),
        ("eeg_only", "emg_only", "EEG-only vs sEMG-only"),
    ]
    ALPHA       = 0.05
    BONFERRONI  = ALPHA / len(PAIRS)

    # ── Wilcoxon Signed-rank Tests ──────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Wilcoxon Signed-rank Tests  "
          f"(Bonferroni α = {BONFERRONI:.4f})")
    print(f"{'='*65}")

    wilcoxon_out = {
        "N":                N,
        "alpha":            ALPHA,
        "bonferroni_alpha": round(BONFERRONI, 4),
        "pairs":            {},
    }

    for a, b, label in PAIRS:
        pair_res = {}
        for metric in ["acc", "kappa"]:
            xa = df[f"{metric}_{a}"].values
            xb = df[f"{metric}_{b}"].values
            stat, p, z, r = _wilcoxon_r(xa, xb)
            if p is None:
                sig = "?"
            elif p < BONFERRONI:
                sig = "**"
            elif p < ALPHA:
                sig = "*"
            else:
                sig = "ns"

            pair_res[metric] = {
                "statistic": round(stat, 4) if stat is not None else None,
                "p_value":   round(p,    6) if p    is not None else None,
                "z":         round(z,    4) if z    is not None else None,
                "effect_r":  round(r,    4) if r    is not None else None,
                "sig":       sig,
            }
            print(f"  {label:<37} [{metric:5s}]  "
                  f"W={stat:6.1f}  p={p:.4f}  z={z:+.3f}  r={r:.3f}  {sig}")

        wilcoxon_out["pairs"][f"{a}_vs_{b}"] = {
            "label": label,
            **pair_res,
        }

    os.makedirs(abl_dir, exist_ok=True)
    wil_path = os.path.join(abl_dir, "wilcoxon_results.json")
    with open(wil_path, "w") as f:
        json.dump(wilcoxon_out, f, indent=2)
    print(f"\n  저장: {wil_path}")

    # ── ITR Bootstrap 95% CI ────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  ITR Bootstrap 95% CI  (n_boot=1000, seed={SEED})")
    print(f"{'='*65}")

    itr_out = {}
    for cond in CONDITIONS:
        col  = f"itr_{cond}"
        if col not in df.columns:
            print(f"  {cond}: 컬럼 없음 — 건너뜀")
            continue
        itrs = df[col].values
        mean, std, lo, hi = _bootstrap_ci(itrs)
        itr_out[cond] = {
            "mean":     round(mean, 4),
            "std":      round(std,  4),
            "ci95_lo":  round(lo,   4),
            "ci95_hi":  round(hi,   4),
        }
        print(f"  {cond:<12}: {mean:.4f} ± {std:.4f}  "
              f"95% CI [{lo:.4f}, {hi:.4f}]  bits/min")

    itr_path = os.path.join(abl_dir, "itr_bootstrap.json")
    with open(itr_path, "w") as f:
        json.dump(itr_out, f, indent=2)
    print(f"\n  저장: {itr_path}")

    return wilcoxon_out, itr_out


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Wilcoxon Signed-rank + ITR Bootstrap",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--drive_root", type=str,
        default=str(Path(__file__).resolve().parents[1] / "BCI_Research"),
        help="BCI_Research 루트 경로 (로컬 또는 Colab Drive)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.drive_root)
