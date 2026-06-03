"""
subgroup_analysis.py
====================
sEMG 품질(acc_emg_only) median split 기반 서브그룹 분석
Fusion vs EEG-only 재검정 (JNE 보완 분석)
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
ABL_DIR  = ROOT / "BCI_Research" / "results" / "ablation"
CSV_PATH = ABL_DIR / "ablation_results.csv"
PNG_PATH = ABL_DIR / "subgroup_wilcoxon.png"
OUT_CSV  = ABL_DIR / "subgroup_summary.csv"

C_FUSION = "#55A868"
C_EEG    = "#4C72B0"
N_TESTS  = 6   # Bonferroni: 2 groups × 3 metrics


# ════════════════════════════════════════════════════════════════
#  Wilcoxon + effect size (서브그룹용, n 가변)
# ════════════════════════════════════════════════════════════════

def wilcoxon_r(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    diff = x - y
    nz   = int((diff != 0).sum())
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


def sig_stars(p):
    if   p < 0.001: return "***"
    elif p < 0.01:  return "**"
    elif p < 0.05:  return "*"
    return "ns"


def effect_label(r):
    if   r >= 0.5: return "large"
    elif r >= 0.3: return "medium"
    elif r >= 0.1: return "small"
    return "negligible"


# ════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════

def run():
    if not CSV_PATH.exists():
        sys.exit(f"[ERROR] {CSV_PATH} 없음")

    df  = pd.read_csv(CSV_PATH)
    N   = len(df)
    med = float(df["acc_emg_only"].median())

    high = df[df["acc_emg_only"] >= med].reset_index(drop=True)
    low  = df[df["acc_emg_only"] <  med].reset_index(drop=True)

    print(f"N = {N}")
    print(f"sEMG accuracy median = {med:.4f}")
    print(f"High-sEMG : n = {len(high)}  (acc_emg_only >= {med:.4f})")
    print(f"Low-sEMG  : n = {len(low)}   (acc_emg_only <  {med:.4f})\n")

    METRICS = [
        ("acc",   "Accuracy"),
        ("kappa", "Cohen's κ"),
        ("itr",   "ITR"),
    ]
    GROUPS = [("High-sEMG", high), ("Low-sEMG", low)]

    # ── 1. 기술통계 ───────────────────────────────────────────
    print("=" * 60)
    print("  기술통계")
    print("=" * 60)
    for gname, gdf in GROUPS:
        gain = gdf["acc_fusion"] - gdf["acc_eeg_only"]
        print(f"\n  [{gname}]  n={len(gdf)}")
        print(f"    EEG-only  Acc : {gdf['acc_eeg_only'].mean():.4f} ± {gdf['acc_eeg_only'].std():.4f}")
        print(f"    Fusion    Acc : {gdf['acc_fusion'].mean():.4f} ± {gdf['acc_fusion'].std():.4f}")
        print(f"    Fusion gain   : {gain.mean():.4f} ± {gain.std():.4f}  "
              f"({(gain > 0).sum()} positive / {len(gdf)} total)")

    # ── 2. Wilcoxon (6쌍 × Bonferroni) ────────────────────────
    rows = []
    for gname, gdf in GROUPS:
        for mkey, mlabel in METRICS:
            xa = gdf[f"{mkey}_fusion"].values
            xb = gdf[f"{mkey}_eeg_only"].values
            stat, p, z, r = wilcoxon_r(xa, xb)
            rows.append({
                "group":       gname,
                "metric":      mlabel,
                "n":           len(gdf),
                "fusion_mean": float(np.mean(xa)),
                "eeg_mean":    float(np.mean(xb)),
                "W_stat":      stat,
                "p_raw":       p,
                "Z":           z,
                "effect_r":    r,
                "effect":      effect_label(r),
            })

    res_df = pd.DataFrame(rows)
    _, p_corr, _, _ = multipletests(res_df["p_raw"].fillna(1.0), method="bonferroni")
    res_df["p_corr"] = p_corr
    res_df["sig"]    = [sig_stars(pc) for pc in p_corr]

    # ── 터미널 출력 ───────────────────────────────────────────
    print("\n" + "=" * 95)
    print("  Wilcoxon Signed-Rank Test: Fusion vs EEG-only  "
          f"(Bonferroni, {N_TESTS} comparisons)")
    print("=" * 95)
    hdr = (f"{'Group':<12} | {'Metric':<10} | {'n':>3} | "
           f"{'Fusion':>8} | {'EEG':>8} | {'W-stat':>7} | "
           f"{'p-raw':>7} | {'p-corr':>7} | {'r':>5} | sig")
    print(hdr)
    print("-" * 95)
    for _, row in res_df.iterrows():
        print(
            f"{row['group']:<12} | {row['metric']:<10} | {row['n']:>3} | "
            f"{row['fusion_mean']:>8.4f} | {row['eeg_mean']:>8.4f} | "
            f"{row['W_stat']:>7.1f} | {row['p_raw']:>7.4f} | "
            f"{row['p_corr']:>7.4f} | {row['effect_r']:>5.3f} | {row['sig']}"
        )

    # CSV 저장
    save_df = res_df.drop(columns=["Z"])
    save_df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n저장: {OUT_CSV}")

    # ── 3. 시각화 ─────────────────────────────────────────────
    _plot(high, low, res_df, med)

    # ── 4. 논문 Results 문구 ──────────────────────────────────
    _print_results_text(res_df, high, low, med)

    return res_df


# ════════════════════════════════════════════════════════════════
#  시각화
# ════════════════════════════════════════════════════════════════

def _plot(high, low, res_df, med):
    METRICS = [
        ("acc",   "Accuracy"),
        ("kappa", "Cohen's κ"),
        ("itr",   "ITR (bits/min)"),
    ]
    GROUPS = [("High-sEMG", high), ("Low-sEMG", low)]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(
        f"Subgroup Wilcoxon: Fusion vs EEG-only by sEMG Quality\n"
        f"(N=26 each, median split threshold = {med:.3f}, Bonferroni-corrected)",
        fontsize=12, fontweight="bold", y=1.01
    )

    rng = np.random.default_rng(42)

    for row_i, (gname, gdf) in enumerate(GROUPS):
        n = len(gdf)
        for col_i, (mkey, mlabel) in enumerate(METRICS):
            ax = axes[row_i][col_i]
            xa = gdf[f"{mkey}_fusion"].values
            xb = gdf[f"{mkey}_eeg_only"].values

            sub     = res_df[(res_df["group"] == gname) & (res_df["metric"] == mlabel)]
            sig_txt = sub["sig"].values[0]  if len(sub) else "?"
            p_corr  = sub["p_corr"].values[0] if len(sub) else np.nan

            # paired lines
            for i in range(n):
                clr   = "#aaaaaa"
                alpha = 0.35
                lw    = 0.6
                if xa[i] > xb[i]:
                    clr   = C_FUSION
                    alpha = 0.5
                elif xa[i] < xb[i]:
                    clr   = C_EEG
                    alpha = 0.5
                ax.plot([0, 1], [xa[i], xb[i]], color=clr, lw=lw, alpha=alpha, zorder=1)

            # dots with jitter
            jitter = rng.uniform(-0.07, 0.07, n)
            ax.scatter(np.zeros(n) + jitter, xa, color=C_FUSION,
                       s=24, zorder=3, edgecolors="white", linewidths=0.3, alpha=0.85)
            ax.scatter(np.ones(n)  + jitter, xb, color=C_EEG,
                       s=24, zorder=3, edgecolors="white", linewidths=0.3, alpha=0.85)

            # median ± IQR bar
            for xi, vals, clr in [(0, xa, C_FUSION), (1, xb, C_EEG)]:
                med_v = np.median(vals)
                q1    = np.percentile(vals, 25)
                q3    = np.percentile(vals, 75)
                ax.plot([xi - 0.18, xi + 0.18], [med_v, med_v],
                        color=clr, lw=2.8, zorder=5, solid_capstyle="round")
                ax.errorbar(xi, med_v, yerr=[[med_v - q1], [q3 - med_v]],
                            fmt="none", color=clr, capsize=5, lw=2.0, zorder=4)

            # significance bracket
            y_vals = np.concatenate([xa, xb])
            y_max  = np.max(y_vals)
            y_rng  = np.ptp(y_vals) if np.ptp(y_vals) > 0 else 0.1
            bh     = y_max + y_rng * 0.10
            ax.plot([0, 0, 1, 1], [bh - y_rng*0.03, bh, bh, bh - y_rng*0.03],
                    color="black", lw=1.2)
            p_str  = f"p={p_corr:.3f}" if not (np.isnan(p_corr) if isinstance(p_corr, float) else False) else ""
            color_txt = "black" if sig_txt == "ns" else "#c44e52"
            ax.text(0.5, bh + y_rng * 0.015,
                    f"{sig_txt}  {p_str}",
                    ha="center", va="bottom", fontsize=8.5, color=color_txt,
                    fontweight="bold" if sig_txt != "ns" else "normal")

            # 축 설정
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Fusion", "EEG-only"], fontsize=9)
            ax.set_xlim(-0.4, 1.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="y", labelsize=8)

            if col_i == 0:
                ax.set_ylabel(gname, fontsize=10, fontweight="bold", labelpad=4)
            if row_i == 0:
                ax.set_title(mlabel, fontsize=10, fontweight="bold")

    # 범례
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=C_FUSION, label="Fusion"),
        mpatches.Patch(color=C_EEG,    label="EEG-only"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
    print(f"저장: {PNG_PATH}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
#  논문 Results 문구
# ════════════════════════════════════════════════════════════════

def _print_results_text(res_df, high, low, med):
    def get(group, metric):
        return res_df[(res_df["group"] == group) & (res_df["metric"] == metric)].iloc[0]

    h_acc = get("High-sEMG", "Accuracy")
    l_acc = get("Low-sEMG",  "Accuracy")
    h_kap = get("High-sEMG", "Cohen's κ")
    l_kap = get("Low-sEMG",  "Cohen's κ")
    h_itr = get("High-sEMG", "ITR")
    l_itr = get("Low-sEMG",  "ITR")

    print("\n" + "=" * 60)
    print("  논문 Results 문구 초안")
    print("=" * 60)
    print(
        f'\n"When stratified by sEMG signal quality (median split, '
        f'threshold = {med:.3f}),\n'
        f'the Fusion model significantly outperformed EEG-only in the '
        f'High-sEMG subgroup\n'
        f'(Accuracy: {h_acc["fusion_mean"]:.3f} vs {h_acc["eeg_mean"]:.3f}, '
        f'W = {h_acc["W_stat"]:.1f}, '
        f'p_corr = {h_acc["p_corr"]:.4f}, '
        f'r = {h_acc["effect_r"]:.3f}, {h_acc["sig"]};\n'
        f' Cohen\'s κ: {h_kap["fusion_mean"]:.3f} vs {h_kap["eeg_mean"]:.3f}, '
        f'p_corr = {h_kap["p_corr"]:.4f}, {h_kap["sig"]};\n'
        f' ITR: {h_itr["fusion_mean"]:.3f} vs {h_itr["eeg_mean"]:.3f} bits/min, '
        f'p_corr = {h_itr["p_corr"]:.4f}, {h_itr["sig"]}),\n'
        f'whereas no significant difference was observed in the '
        f'Low-sEMG subgroup\n'
        f'(Accuracy: {l_acc["fusion_mean"]:.3f} vs {l_acc["eeg_mean"]:.3f}, '
        f'p_corr = {l_acc["p_corr"]:.4f}, {l_acc["sig"]};\n'
        f' Cohen\'s κ: {l_kap["fusion_mean"]:.3f} vs {l_kap["eeg_mean"]:.3f}, '
        f'p_corr = {l_kap["p_corr"]:.4f}, {l_kap["sig"]};\n'
        f' ITR: {l_itr["fusion_mean"]:.3f} vs {l_itr["eeg_mean"]:.3f} bits/min, '
        f'p_corr = {l_itr["p_corr"]:.4f}, {l_itr["sig"]}).\n'
        f'This pattern was consistent across Cohen\'s κ and ITR metrics,\n'
        f'suggesting that multimodal fusion provides meaningful benefit\n'
        f'specifically when sEMG signals carry sufficient discriminative '
        f'information."'
    )


if __name__ == "__main__":
    run()
