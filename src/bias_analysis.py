"""
bias_analysis.py — Right MI Bias 분석
=======================================
Right MI 과분류(bias) 피험자를 식별하고,
EEG-only 대비 Fusion 조건에서 bias 개선 여부를 분석합니다.

Bias 정의: right_recall - left_recall > 0.3  (Fusion 조건 기준)

사용법:
  # 로컬 (scipy + pandas 설치됨)
  python src/bias_analysis.py

  # Colab
  python /content/bias_analysis.py \\
      --drive_root /content/drive/MyDrive/BCI_Research

출력:
  results/bias_analysis.json
  results/ablation/bias_subjects.csv
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

SEED          = 42
BIAS_THRESHOLD = 0.30   # right_recall - left_recall > 이 값이면 bias 피험자


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
    print(f"  전체 피험자: N = {N}")

    # ── Bias 피험자 식별 (Fusion 조건 right_recall - left_recall) ──
    df["recall_diff_fusion"] = (
        df["right_recall_fusion"] - df["left_recall_fusion"]
    )
    bias_mask    = df["recall_diff_fusion"] > BIAS_THRESHOLD
    bias_sids    = df.loc[bias_mask,  "sid"].tolist()
    nonbias_sids = df.loc[~bias_mask, "sid"].tolist()

    n_bias    = int(bias_mask.sum())
    n_nonbias = N - n_bias

    print(f"\n  Bias 임계값: right_recall - left_recall > {BIAS_THRESHOLD}")
    print(f"  Bias 피험자:     {n_bias}명  {sorted(bias_sids)}")
    print(f"  Non-bias 피험자: {n_nonbias}명")

    # ── Bias vs Non-bias 그룹 지표 비교 ─────────────────────────
    groups = {
        "bias":     df[bias_mask],
        "non_bias": df[~bias_mask],
    }

    group_stats = {}
    print(f"\n{'='*65}")
    print(f"  그룹별 Accuracy 평균 (EEG-only / sEMG-only / Fusion)")
    print(f"{'='*65}")

    for grp, sub in groups.items():
        row = {}
        for cond in ["eeg_only", "emg_only", "fusion"]:
            vals       = sub[f"acc_{cond}"].values
            row[cond]  = {
                "mean": round(float(vals.mean()), 4),
                "std":  round(float(vals.std()),  4),
                "n":    int(len(vals)),
            }
            print(f"  [{grp:<9}] {cond:<12}: "
                  f"{vals.mean():.4f} ± {vals.std():.4f}")
        group_stats[grp] = row

    # ── Mann-Whitney U: bias vs non-bias (Fusion 정확도) ──────────
    print(f"\n{'='*65}")
    print(f"  Mann-Whitney U — Fusion Accuracy: bias vs non-bias")
    print(f"{'='*65}")

    mw_results = {}
    for cond in ["eeg_only", "emg_only", "fusion"]:
        col  = f"acc_{cond}"
        a    = df.loc[bias_mask,  col].values
        b    = df.loc[~bias_mask, col].values
        if len(a) < 2 or len(b) < 2:
            print(f"  {cond:<12}: 표본 부족 — 건너뜀")
            continue
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        # effect size r = Z / sqrt(N)
        # approx Z from U
        n1, n2   = len(a), len(b)
        mu_u     = n1 * n2 / 2
        sigma_u  = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z        = (stat - mu_u) / sigma_u
        r        = abs(z) / np.sqrt(n1 + n2)
        sig      = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        print(f"  {cond:<12}: U={stat:.1f}  p={p:.4f}  z={z:+.3f}  r={r:.3f}  {sig}")
        mw_results[cond] = {
            "U": round(float(stat), 2),
            "p": round(float(p),    6),
            "z": round(float(z),    4),
            "r": round(float(r),    4),
            "sig": sig,
        }

    # ── Fusion vs EEG-only improvement in bias group ──────────────
    print(f"\n{'='*65}")
    print(f"  Bias 그룹 내: Fusion - EEG-only Accuracy 개선")
    print(f"{'='*65}")

    bias_df   = df[bias_mask].copy()
    bias_df["improvement_fusion_vs_eeg"] = (
        bias_df["acc_fusion"] - bias_df["acc_eeg_only"]
    )
    bias_df["improvement_fusion_vs_emg"] = (
        bias_df["acc_fusion"] - bias_df["acc_emg_only"]
    )

    imp_fe = bias_df["improvement_fusion_vs_eeg"].values
    imp_fm = bias_df["improvement_fusion_vs_emg"].values

    print(f"  Fusion - EEG-only : "
          f"{imp_fe.mean():+.4f} ± {imp_fe.std():.4f}  "
          f"(+{(imp_fe > 0).sum()}/{n_bias} 피험자 개선)")
    print(f"  Fusion - sEMG-only: "
          f"{imp_fm.mean():+.4f} ± {imp_fm.std():.4f}  "
          f"(+{(imp_fm > 0).sum()}/{n_bias} 피험자 개선)")

    # ── 저장 ─────────────────────────────────────────────────────
    out_data = {
        "bias_threshold":   BIAS_THRESHOLD,
        "N_total":          N,
        "N_bias":           n_bias,
        "N_non_bias":       n_nonbias,
        "bias_sids":        sorted(bias_sids),
        "non_bias_sids":    sorted(nonbias_sids),
        "group_stats":      group_stats,
        "mannwhitney":      mw_results,
        "bias_improvement": {
            "fusion_vs_eeg_mean":  round(float(imp_fe.mean()), 4),
            "fusion_vs_eeg_std":   round(float(imp_fe.std()),  4),
            "fusion_vs_emg_mean":  round(float(imp_fm.mean()), 4),
            "fusion_vs_emg_std":   round(float(imp_fm.std()),  4),
            "n_improved_vs_eeg":   int((imp_fe > 0).sum()),
            "n_improved_vs_emg":   int((imp_fm > 0).sum()),
        },
    }

    os.makedirs(os.path.join(drive_root, "results"), exist_ok=True)
    json_path = os.path.join(drive_root, "results", "bias_analysis.json")
    with open(json_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n  저장: {json_path}")

    bias_csv_path = os.path.join(abl_dir, "bias_subjects.csv")
    bias_df[[
        "sid", "acc_eeg_only", "acc_emg_only", "acc_fusion",
        "left_recall_fusion", "right_recall_fusion", "recall_diff_fusion",
        "improvement_fusion_vs_eeg", "improvement_fusion_vs_emg",
    ]].to_csv(bias_csv_path, index=False)
    print(f"  저장: {bias_csv_path}")

    return out_data


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Right MI Bias Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--drive_root", type=str,
        default=str(Path(__file__).resolve().parents[1] / "BCI_Research"),
        help="BCI_Research 루트 경로",
    )
    p.add_argument(
        "--threshold", type=float, default=BIAS_THRESHOLD,
        help="Bias 임계값 (right_recall - left_recall)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    global BIAS_THRESHOLD
    BIAS_THRESHOLD = args.threshold
    run(args.drive_root)
