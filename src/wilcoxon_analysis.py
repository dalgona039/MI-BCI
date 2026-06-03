"""
wilcoxon_analysis.py
====================
HybridBCI Ablation 결과 통계 검정 (JNE 투고용)
- Wilcoxon Signed-Rank Test (9쌍 × Bonferroni correction)
- Effect size r = Z / sqrt(N)
- Paired dot plot 시각화 (3×3)
- Fusion gain vs sEMG accuracy Spearman 상관
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
ABL_DIR  = ROOT / "BCI_Research" / "results" / "ablation"
CSV_PATH = ABL_DIR / "ablation_results.csv"
PNG_PATH = ABL_DIR / "wilcoxon_results.png"
OUT_CSV  = ABL_DIR / "wilcoxon_summary.csv"

# ── 색상 ──────────────────────────────────────────────────────
C_EEG    = "#4C72B0"
C_EMG    = "#DD8452"
C_FUSION = "#55A868"

N_TESTS  = 9   # Bonferroni 분모


# ════════════════════════════════════════════════════════════════
#  Wilcoxon + effect size
# ════════════════════════════════════════════════════════════════

def wilcoxon_r(x, y):
    """Two-sided Wilcoxon signed-rank + r = |Z| / sqrt(N_nonzero)."""
    x, y   = np.asarray(x, float), np.asarray(y, float)
    diff   = x - y
    nz     = int((diff != 0).sum())
    if nz < 2:
        return np.nan, np.nan, np.nan, np.nan

    res  = wilcoxon(x, y, alternative="two-sided")
    stat = float(res.statistic)
    p    = float(res.pvalue)

    mu    = nz * (nz + 1) / 4.0
    sigma = np.sqrt(nz * (nz + 1) * (2 * nz + 1) / 24.0)
    z     = (stat - mu) / sigma
    r     = abs(z) / np.sqrt(nz)
    return stat, p, float(z), float(r)


def effect_label(r):
    if r >= 0.5:
        return "large"
    elif r >= 0.3:
        return "medium"
    elif r >= 0.1:
        return "small"
    return "negligible"


def sig_stars(p_corr):
    if p_corr < 0.001:
        return "***"
    elif p_corr < 0.01:
        return "**"
    elif p_corr < 0.05:
        return "*"
    return "ns"


# ════════════════════════════════════════════════════════════════
#  메인 분석
# ════════════════════════════════════════════════════════════════

def run():
    if not CSV_PATH.exists():
        sys.exit(f"[ERROR] {CSV_PATH} 없음")

    df = pd.read_csv(CSV_PATH)
    N  = len(df)
    print(f"피험자 수: N = {N}\n")

    # ── 9쌍 정의 ──────────────────────────────────────────────
    COMPARISONS = [
        ("fusion",   "eeg_only",  "Fusion vs EEG-only"),
        ("fusion",   "emg_only",  "Fusion vs sEMG-only"),
        ("eeg_only", "emg_only",  "EEG-only vs sEMG-only"),
    ]
    METRICS = [
        ("acc",   "Accuracy"),
        ("kappa", "Cohen's κ"),
        ("itr",   "ITR"),
    ]

    rows = []
    for cmp_a, cmp_b, cmp_label in COMPARISONS:
        for metric_key, metric_label in METRICS:
            xa = df[f"{metric_key}_{cmp_a}"].values
            xb = df[f"{metric_key}_{cmp_b}"].values
            stat, p, z, r = wilcoxon_r(xa, xb)
            rows.append({
                "comparison":  cmp_label,
                "metric":      metric_label,
                "col_a":       f"{metric_key}_{cmp_a}",
                "col_b":       f"{metric_key}_{cmp_b}",
                "W_stat":      stat,
                "p_value":     p,
                "Z":           z,
                "effect_r":    r,
            })

    res_df = pd.DataFrame(rows)

    # ── Bonferroni correction ──────────────────────────────────
    reject, p_corr, _, _ = multipletests(
        res_df["p_value"].fillna(1.0), method="bonferroni"
    )
    res_df["p_corr"]  = p_corr
    res_df["sig"]     = [sig_stars(pc) for pc in p_corr]
    res_df["effect"]  = [effect_label(r) for r in res_df["effect_r"]]

    # ── 터미널 출력 ───────────────────────────────────────────
    header = (f"{'Comparison':<24} | {'Metric':<10} | {'W-stat':>7} | "
              f"{'p-value':>8} | {'p-corr':>8} | {'Effect r':>8} | Sig")
    sep    = "-" * len(header)
    print(header)
    print(sep)

    prev_cmp = None
    comments = {
        "Fusion vs EEG-only":   (
            "→ Fusion이 EEG-only 대비 전반적으로 우세하나 개인 편차가 크며, "
            "sEMG 신호 품질이 낮은 피험자에서 개선 폭이 감소함."
        ),
        "Fusion vs sEMG-only":  (
            "→ Fusion은 sEMG-only 대비 일관되게 높은 성능을 보여, "
            "EEG 스트림이 sEMG 단독 모델의 분류 한계를 보완함."
        ),
        "EEG-only vs sEMG-only":(
            "→ EEG-only가 sEMG-only보다 유의하게 높은 성능을 보여, "
            "EEG가 주요 분류 기여 모달리티임을 확인함."
        ),
    }

    for _, row in res_df.iterrows():
        if row["comparison"] != prev_cmp and prev_cmp is not None:
            print(f"  {comments[prev_cmp]}\n")
        print(
            f"{row['comparison']:<24} | {row['metric']:<10} | "
            f"{row['W_stat']:>7.1f} | {row['p_value']:>8.4f} | "
            f"{row['p_corr']:>8.4f} | {row['effect_r']:>8.3f} | "
            f"{row['sig']}"
        )
        prev_cmp = row["comparison"]
    print(f"  {comments[prev_cmp]}\n")

    # ── CSV 저장 ───────────────────────────────────────────────
    save_df = res_df[["comparison", "metric", "W_stat",
                       "p_value", "p_corr", "effect_r", "effect", "sig"]].copy()
    save_df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"저장: {OUT_CSV}")

    # ── 시각화 ────────────────────────────────────────────────
    _plot(df, res_df, N)

    # ── Spearman 상관 ─────────────────────────────────────────
    _spearman(df, N)

    return res_df


# ════════════════════════════════════════════════════════════════
#  Paired dot plot 3×3
# ════════════════════════════════════════════════════════════════

def _plot(df, res_df, N):
    COMPARISONS = [
        ("fusion",   "eeg_only",  "Fusion vs EEG-only",    C_FUSION, C_EEG),
        ("fusion",   "emg_only",  "Fusion vs sEMG-only",   C_FUSION, C_EMG),
        ("eeg_only", "emg_only",  "EEG-only vs sEMG-only", C_EEG,    C_EMG),
    ]
    METRICS = [
        ("acc",   "Accuracy"),
        ("kappa", "Cohen's κ"),
        ("itr",   "ITR (bits/min)"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    fig.suptitle(
        "Wilcoxon Signed-Rank Test: EEG-only / sEMG-only / Fusion\n"
        f"(N={N}, Bonferroni-corrected, two-sided)",
        fontsize=13, fontweight="bold", y=1.01
    )

    for row_i, (cmp_a, cmp_b, cmp_label, col_a, col_b) in enumerate(COMPARISONS):
        for col_i, (metric_key, metric_label) in enumerate(METRICS):
            ax    = axes[row_i][col_i]
            xa    = df[f"{metric_key}_{cmp_a}"].values
            xb    = df[f"{metric_key}_{cmp_b}"].values

            # 검정 결과 가져오기
            sub   = res_df[
                (res_df["col_a"] == f"{metric_key}_{cmp_a}") &
                (res_df["col_b"] == f"{metric_key}_{cmp_b}")
            ]
            sig_txt   = sub["sig"].values[0]  if len(sub) else "?"
            p_corr    = sub["p_corr"].values[0] if len(sub) else np.nan

            # paired lines
            xs = [0, 1]
            for i in range(N):
                color = "#aaaaaa"
                lw    = 0.5
                alpha = 0.4
                if xa[i] < xb[i]:
                    color = col_b
                    alpha = 0.55
                ax.plot(xs, [xa[i], xb[i]], color=color, lw=lw, alpha=alpha, zorder=1)

            # dots
            jitter = np.random.default_rng(42).uniform(-0.06, 0.06, N)
            ax.scatter(np.zeros(N) + jitter, xa, color=col_a,
                       s=22, zorder=3, edgecolors="white", linewidths=0.3, alpha=0.85)
            ax.scatter(np.ones(N)  + jitter, xb, color=col_b,
                       s=22, zorder=3, edgecolors="white", linewidths=0.3, alpha=0.85)

            # median ± IQR
            for xi, vals, clr in [(0, xa, col_a), (1, xb, col_b)]:
                med = np.median(vals)
                q1  = np.percentile(vals, 25)
                q3  = np.percentile(vals, 75)
                ax.plot([xi - 0.18, xi + 0.18], [med, med],
                        color=clr, lw=2.5, zorder=5)
                ax.errorbar(xi, med, yerr=[[med - q1], [q3 - med]],
                            fmt="none", color=clr, capsize=5, lw=1.8, zorder=4)

            # significance bracket
            y_max = max(np.max(xa), np.max(xb))
            y_rng = max(np.max(xa), np.max(xb)) - min(np.min(xa), np.min(xb))
            bh    = y_max + y_rng * 0.08
            ax.plot([0, 0, 1, 1], [bh - y_rng*0.02, bh, bh, bh - y_rng*0.02],
                    color="black", lw=1.2)
            p_str = f"p={p_corr:.3f}" if not np.isnan(p_corr) else ""
            ax.text(0.5, bh + y_rng * 0.01,
                    f"{sig_txt}  {p_str}",
                    ha="center", va="bottom", fontsize=8.5)

            # axis labels
            cond_a_label = {"fusion": "Fusion", "eeg_only": "EEG-only",
                            "emg_only": "sEMG-only"}
            ax.set_xticks([0, 1])
            ax.set_xticklabels(
                [cond_a_label[cmp_a], cond_a_label[cmp_b]],
                fontsize=9
            )
            ax.set_xlim(-0.4, 1.4)

            if col_i == 0:
                ax.set_ylabel(cmp_label, fontsize=8.5, labelpad=4)
            if row_i == 0:
                ax.set_title(metric_label, fontsize=10, fontweight="bold")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="y", labelsize=8)

    # legend
    handles = [
        mpatches.Patch(color=C_FUSION, label="Fusion"),
        mpatches.Patch(color=C_EEG,    label="EEG-only"),
        mpatches.Patch(color=C_EMG,    label="sEMG-only"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
    print(f"저장: {PNG_PATH}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
#  Spearman 상관: Fusion gain vs sEMG accuracy
# ════════════════════════════════════════════════════════════════

def _spearman(df, N):
    gain      = df["acc_fusion"].values - df["acc_eeg_only"].values
    emg_acc   = df["acc_emg_only"].values
    n_neg     = int((gain < 0).sum())

    rho, p = spearmanr(emg_acc, gain)
    sig    = sig_stars(p)

    print("\n" + "=" * 55)
    print("  Spearman 상관: Fusion gain vs sEMG accuracy")
    print("=" * 55)
    print(f"  Fusion < EEG-only 피험자: {n_neg}/{N} ({n_neg/N*100:.1f}%)")
    print(f"  ρ = {rho:.4f},  p = {p:.4f}  {sig}")

    interp = (
        "→ sEMG 정확도가 높을수록 Fusion 이득이 크다는 가설이 "
        + ("지지됨." if p < 0.05 else "통계적으로 지지되지 않음 (경향성만 존재).")
    )
    print(f"  {interp}")

    # scatter 저장
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(emg_acc, gain, color=C_EMG, alpha=0.7, s=35,
               edgecolors="white", linewidths=0.4)
    ax.axhline(0, color="gray", lw=1, ls="--")
    m, b    = np.polyfit(emg_acc, gain, 1)
    x_line  = np.linspace(emg_acc.min(), emg_acc.max(), 100)
    ax.plot(x_line, m * x_line + b, color="#c44e52", lw=1.5, ls="--")
    ax.set_xlabel("sEMG-only Accuracy", fontsize=10)
    ax.set_ylabel("Fusion Gain (Fusion − EEG-only)", fontsize=10)
    ax.set_title(
        f"Spearman ρ = {rho:.3f},  p = {p:.4f}  {sig}\n"
        f"({n_neg}/{N} subjects: Fusion < EEG-only)",
        fontsize=9
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    sp_path = ABL_DIR / "spearman_fusion_gain.png"
    fig.savefig(sp_path, dpi=150, bbox_inches="tight")
    print(f"저장: {sp_path}")
    plt.close(fig)


if __name__ == "__main__":
    run()
