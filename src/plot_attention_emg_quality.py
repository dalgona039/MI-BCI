"""Render the publication figure for attention weights and EMG signal quality."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "BCI_Research/results/attention/attention_weights_per_subject.csv"
OUT = ROOT / "BCI_Research/results/attention/figures"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#5B5B5B"
LIGHT_GRAY = "#D9D9D9"


def mean_ci(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray):
    """OLS mean-response line and 95% confidence interval."""
    slope, intercept, _, _, _ = stats.linregress(x, y)
    fit = intercept + slope * x_grid
    residual = y - (intercept + slope * x)
    s_err = np.sqrt(np.sum(residual**2) / (len(x) - 2))
    se = s_err * np.sqrt(1 / len(x) + (x_grid - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    tcrit = stats.t.ppf(0.975, len(x) - 2)
    return fit, fit - tcrit * se, fit + tcrit * se


def verify(df: pd.DataFrame) -> None:
    """Fail loudly if manuscript annotations drift from the stored results."""
    assert len(df) == 52
    np.testing.assert_allclose(
        [df.w_eeg_mean.mean(), df.w_eeg_mean.std(), df.w_emg_mean.mean(), df.w_emg_mean.std()],
        [0.657643, 0.173965, 0.342357, 0.173965],
        atol=5e-6,
    )
    emg_dom = df.loc[df.w_emg_mean > 0.5, "sid"].tolist()
    assert emg_dom == [6, 8, 17, 19, 20, 27, 30, 33, 37, 44]
    rho, p = stats.spearmanr(df.emg_snr_db, df.w_emg_mean)
    np.testing.assert_allclose([rho, p], [0.0606164091, 0.6694679548], atol=1e-9)
    quartile = pd.qcut(df.emg_snr_db, 4, labels=False)
    bottom = df.loc[quartile == 0, "w_emg_mean"]
    top = df.loc[quartile == 3, "w_emg_mean"]
    u, p_u = stats.mannwhitneyu(top, bottom, alternative="two-sided")
    np.testing.assert_allclose(
        [top.mean(), top.std(), bottom.mean(), bottom.std(), u, p_u],
        [0.3588684615, 0.1643296736, 0.3505207692, 0.1444784690, 87.0, 0.9183089345],
        atol=1e-9,
    )


def main() -> None:
    df = pd.read_csv(DATA)
    verify(df)
    OUT.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.2, 5.9), constrained_layout=False)
    fig.patch.set_facecolor("white")

    # Panel A: exact subject-level attention weights.
    eeg = df.w_eeg_mean.to_numpy()
    emg = df.w_emg_mean.to_numpy()
    groups = [eeg, emg]
    colors = [BLUE, ORANGE]
    positions = [0, 1]

    violins = ax_a.violinplot(groups, positions=positions, widths=0.62, showextrema=False)
    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.14)
        body.set_linewidth(1.0)

    box = ax_a.boxplot(
        groups,
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": GRAY, "linewidth": 0.9},
        capprops={"color": GRAY, "linewidth": 0.9},
        boxprops={"linewidth": 1.2},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor("white")
        patch.set_edgecolor(color)

    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.105, 0.105, len(df))
    for pos, values, color in zip(positions, groups, colors):
        ax_a.scatter(
            pos + jitter,
            values,
            s=18,
            facecolor=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.72,
            zorder=3,
        )
        ax_a.scatter(pos, values.mean(), marker="D", s=42, color="black", edgecolor="white", linewidth=0.5, zorder=5)

    ax_a.text(0, 0.965, r"$w_{EEG}$ = 0.658 ± 0.174", ha="center", va="top", fontsize=9)
    ax_a.text(1, 0.965, r"$w_{EMG}$ = 0.342 ± 0.174", ha="center", va="top", fontsize=9)
    dominance = (
        "EEG-dominant ($w_{EEG}>0.50$): 42/52 (81%)\n"
        "sEMG-dominant ($w_{EMG}>0.50$): 10/52 (19%)\n"
        "sEMG-dominant IDs: s06, s08, s17, s19, s20,\n"
        "s27, s30, s33, s37, s44"
    )
    ax_a.text(
        0.03,
        0.035,
        dominance,
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.1,
        linespacing=1.28,
        bbox={"boxstyle": "square,pad=0.35", "facecolor": "white", "edgecolor": LIGHT_GRAY, "linewidth": 0.8, "alpha": 0.95},
        zorder=10,
    )
    ax_a.set_xlim(-0.55, 1.55)
    ax_a.set_ylim(0, 1.0)
    ax_a.set_xticks(positions, [r"$w_{EEG}$", r"$w_{EMG}$"])
    ax_a.set_yticks(np.arange(0, 1.01, 0.2))
    ax_a.set_ylabel("Attention weight")
    ax_a.set_title("A   Attention weight distribution", loc="left", fontweight="bold", pad=11)

    # Panel B: exact EMG-SNR scatter, OLS trend, and 95% mean CI.
    x = df.emg_snr_db.to_numpy()
    y = df.w_emg_mean.to_numpy()
    grid = np.linspace(x.min(), x.max(), 250)
    fit, lo, hi = mean_ci(x, y, grid)
    ax_b.fill_between(grid, lo, hi, color=ORANGE, alpha=0.14, linewidth=0, label="95% CI")
    ax_b.plot(grid, fit, color=ORANGE, linewidth=1.7, label="Linear trend")
    ax_b.scatter(x, y, s=30, color=ORANGE, edgecolor="white", linewidth=0.45, alpha=0.8, zorder=3)
    ax_b.text(
        0.03,
        0.96,
        r"Spearman $\rho$ = +0.061, $p$ = 0.669, ns",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    quartile_text = (
        "Top SNR quartile (n=13): 0.359 ± 0.164\n"
        "Bottom SNR quartile (n=13): 0.351 ± 0.145\n"
        "Mann–Whitney U = 87.0, p = 0.918, ns"
    )
    ax_b.text(
        0.97,
        0.95,
        quartile_text,
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        linespacing=1.3,
        bbox={"boxstyle": "square,pad=0.35", "facecolor": "white", "edgecolor": LIGHT_GRAY, "linewidth": 0.8, "alpha": 0.96},
        zorder=10,
    )
    ax_b.text(
        0.03,
        0.04,
        r"$w_{EMG}$ range: 0.002 (s40) to 0.677 (s17)",
        transform=ax_b.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        color=GRAY,
    )
    ax_b.set_xlim(0.4, 4.3)
    ax_b.set_ylim(0, 0.75)
    ax_b.set_xlabel("EMG SNR (dB)")
    ax_b.set_ylabel(r"Mean $w_{EMG}$")
    ax_b.set_title(r"B   Relationship between $w_{EMG}$ and EMG-SNR", loc="left", fontweight="bold", pad=11)

    for ax in (ax_a, ax_b):
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.7, linestyle=(0, (2, 2)), zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3.5)

    fig.suptitle(
        "Attention Weights Distribution and Relationship with EMG Signal Quality (N=52, LOSO)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.055,
        "Per-subject mean attention weights were extracted from best subject-specific fusion checkpoints; "
        "statistics reported for N=52 under LOSO evaluation.",
        ha="center",
        va="center",
        fontsize=8.5,
    )
    fig.text(
        0.5,
        0.025,
        "Attention weights were subject-variable, but aggregate $w_{EMG}$ showed no significant monotonic relationship with EMG-SNR.",
        ha="center",
        va="center",
        fontsize=8.5,
        style="italic",
        color=GRAY,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.17, wspace=0.28)

    stem = OUT / "attention_weights_emg_snr_figure"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
