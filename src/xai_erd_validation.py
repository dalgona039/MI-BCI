#!/usr/bin/env python3
"""
xai_erd_validation.py
=====================
ERD (Event-Related Desynchronization) validation for the 52-subject
Cho2017 Motor Imagery dataset.

Uses the pre-computed band-filtered epochs in HDF5 (no PyTorch required):
  eeg/mu_epochs    (8–12 Hz)
  eeg/beta_epochs  (13–30 Hz)
  eeg/epochs       (4–40 Hz, broadband)

ERD formula:
  ERD(t) = [ P(t) - P_baseline ] / P_baseline × 100 (%)
  Instantaneous power P(t) via analytic signal (Hilbert transform).

Outputs (BCI_Research/results/xai_erd/):
  figures/
    fig_erd_timecourse.png       — Alpha & Beta ERD at C3, Cz, C4
    fig_erd_topomap_mu.png       — Topographic ERD at mu-band peak
    fig_erd_topomap_beta.png     — Topographic ERD at beta-band peak
    fig_erd_tf_spectrogram.png   — Group-average STFT spectrogram (C3, C4)
    fig_erd_literature_compare.png — ERD table vs published values
  erd_statistics.csv             — Mean ERD ± SD at key channels/bands
  erd_significance.csv           — Wilcoxon signed-rank p-values

Reference: Pfurtscheller & Lopes da Silva (1999),
           Neurofeedback definition: alpha ≈ 10–12 Hz, beta ≈ 15–30 Hz.

Local run (no GPU needed):
  python src/xai_erd_validation.py --root BCI_Research

Colab run:
  !python src/xai_erd_validation.py --root /content/drive/MyDrive/BCI_Research
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import hilbert, spectrogram
from scipy.stats import wilcoxon


# ═══════════════════════════════════════════════════════════
#  Channel metadata (same as other XAI scripts)
# ═══════════════════════════════════════════════════════════

CH_NAMES = [
    'Fp1', 'AF7', 'AF3', 'F1',  'F3',  'F5',  'F7',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'FC6', 'FT8', 'T7',  'C5',  'C3',  'C1',  'Cz',
    'C2',  'C4',  'C6',  'T8',  'TP7', 'CP5', 'CP3',
    'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7',  'P5',  'P3',  'P1',  'Pz',  'P2',
    'P4',  'P6',  'P8',  'PO7', 'PO3', 'POz',
    'PO4', 'PO8', 'O1',  'Oz',  'O2',  'Iz',
    'Fp2', 'AF8', 'AF4', 'F2',  'F4',  'F6',  'F8',
    'FT9', 'FT10','TP9', 'TP10','Fpz',
]
assert len(CH_NAMES) == 64
CH_IDX = {ch: i for i, ch in enumerate(CH_NAMES)}

# Key motor channels
C3_IDX  = CH_IDX['C3']   # 18
CZ_IDX  = CH_IDX['Cz']   # 20
C4_IDX  = CH_IDX['C4']   # 22
FC3_IDX = CH_IDX['FC3']  # 9
FC4_IDX = CH_IDX['FC4']  # 13
CP3_IDX = CH_IDX['CP3']  # 27
CP4_IDX = CH_IDX['CP4']  # 31

MOTOR_CHS = {
    'C3': C3_IDX, 'Cz': CZ_IDX, 'C4': C4_IDX,
    'FC3': FC3_IDX, 'FC4': FC4_IDX,
    'CP3': CP3_IDX, 'CP4': CP4_IDX,
}

# Epoch timing
FS          = 512          # Hz
EPOCH_TMIN  = -0.5         # s  (0 = cue onset)
EPOCH_TMAX  = 4.0          # s
N_TIMES     = 2304         # = (4.5s × 512 Hz)
T_AXIS      = np.linspace(EPOCH_TMIN, EPOCH_TMAX, N_TIMES)

BASELINE_MASK = (T_AXIS >= -0.5) & (T_AXIS < 0.0)   # 256 samples
MI_MASK       = (T_AXIS >= 0.5)  & (T_AXIS < 4.0)   # MI window (skip 0-0.5s transient)


# ═══════════════════════════════════════════════════════════
#  ERD computation
# ═══════════════════════════════════════════════════════════

def instantaneous_power(signal: np.ndarray) -> np.ndarray:
    """
    Analytic signal power via Hilbert transform.
    Input : (..., n_times)
    Output: (..., n_times) — instantaneous power (squared envelope)
    """
    return np.abs(hilbert(signal, axis=-1)) ** 2


def compute_erd(signal: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    """
    ERD in dB: ERD(t) = 10·log10[ P(t) / P_baseline ]
    Negative dB = desynchronization; positive = synchronization.
    Log-scale is robust to the small baseline power that arises
    from z-scored narrow-band signals.

    Input : (N_epochs, N_ch, N_times)
    Output: (N_ch, N_times) — trial-averaged ERD dB
    """
    power = instantaneous_power(signal)                                  # (N, ch, T)
    p_bl  = power[:, :, baseline_mask].mean(axis=-1, keepdims=True)     # (N, ch, 1)
    erd   = 10.0 * np.log10(power / (p_bl + 1e-12))                     # (N, ch, T)
    return erd.mean(axis=0)                                              # (ch, T)


def load_subject_erd(sid: int, data_dir: str) -> dict | None:
    h5_path = os.path.join(data_dir, f"sub-{sid:02d}_member_A.h5")
    if not os.path.exists(h5_path):
        return None

    with h5py.File(h5_path, "r") as f:
        mu   = f["eeg/mu_epochs"][:].astype(np.float32)    # (N, 64, 2304)
        beta = f["eeg/beta_epochs"][:].astype(np.float32)
        raw  = f["eeg/epochs"][:].astype(np.float32)
        lbl  = f["labels"][:].astype(np.int64) - 1          # 0=Left, 1=Right

    idx_left  = np.where(lbl == 0)[0]
    idx_right = np.where(lbl == 1)[0]

    return {
        'sid': sid,
        'mu_erd_left':    compute_erd(mu[idx_left],   BASELINE_MASK),
        'mu_erd_right':   compute_erd(mu[idx_right],  BASELINE_MASK),
        'beta_erd_left':  compute_erd(beta[idx_left], BASELINE_MASK),
        'beta_erd_right': compute_erd(beta[idx_right], BASELINE_MASK),
        'raw_left':  raw[idx_left],
        'raw_right': raw[idx_right],
        'n_left':  len(idx_left),
        'n_right': len(idx_right),
    }


# ═══════════════════════════════════════════════════════════
#  Time-frequency spectrogram
# ═══════════════════════════════════════════════════════════

def compute_tf_erd(raw_all: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    STFT-based ERD time-frequency map.
    Input : (N, n_times) — single channel, all trials
    Output: freqs, times, ERD_TF (n_freqs, n_times_stft)
    """
    nperseg  = 256    # 0.5 s window
    noverlap = 224    # 87.5% overlap → step = 32 samples (62.5 ms)
    stft_list = []
    for trial in raw_all:
        f, t, Sxx = spectrogram(trial, fs=FS, nperseg=nperseg,
                                 noverlap=noverlap, scaling='density')
        stft_list.append(Sxx)
    Sxx_mean = np.stack(stft_list).mean(axis=0)   # (n_freqs, n_t)

    # spectrogram t is in signal-frame seconds (0 … 4.5).
    # Epoch baseline = -0.5 to 0 s  →  signal-frame 0 to 0.5 s.
    bl_end   = -EPOCH_TMIN           # 0.5 s in signal time
    bl_mask  = t < bl_end
    p_bl     = Sxx_mean[:, bl_mask].mean(axis=-1, keepdims=True) if bl_mask.any() \
               else Sxx_mean[:, :1]   # fallback to first bin
    erd_tf   = (Sxx_mean - p_bl) / (p_bl + 1e-12) * 100.0

    # Convert t to epoch time (cue = 0)
    t_adjusted = t + EPOCH_TMIN
    return f, t_adjusted, erd_tf


# ═══════════════════════════════════════════════════════════
#  2-D Topomap (matplotlib, no MNE)
# ═══════════════════════════════════════════════════════════

_POS_DEG = {
    'Fpz': (0, 90), 'Fz': (0, 54), 'FCz': (0, 36), 'Cz': (0, 0),
    'CPz': (0, -36), 'Pz': (0, -54), 'POz': (0, -72), 'Oz': (0, -90),
    'C1': (-18, 0), 'C3': (-36, 0), 'C5': (-54, 0), 'T7': (-90, 0),
    'C2': (18, 0),  'C4': (36, 0),  'C6': (54, 0),  'T8': (90, 0),
    'FC1': (-18, 36), 'FC3': (-36, 36), 'FC5': (-54, 36), 'FT7': (-90, 36),
    'FC2': (18, 36),  'FC4': (36, 36),  'FC6': (54, 36),  'FT8': (90, 36),
    'CP1': (-18, -36), 'CP3': (-36, -36), 'CP5': (-54, -36), 'TP7': (-90, -36),
    'CP2': (18, -36),  'CP4': (36, -36),  'CP6': (54, -36),  'TP8': (90, -36),
    'F1': (-18, 54), 'F3': (-36, 54), 'F5': (-54, 54), 'F7': (-90, 54),
    'AF3': (-36, 72), 'AF7': (-72, 72), 'Fp1': (-18, 90),
    'F2': (18, 54),  'F4': (36, 54),  'F6': (54, 54),  'F8': (90, 54),
    'AF4': (36, 72), 'AF8': (72, 72),  'Fp2': (18, 90),
    'P1': (-18, -54), 'P3': (-36, -54), 'P5': (-54, -54), 'P7': (-90, -54),
    'PO3': (-36, -72), 'PO7': (-72, -72), 'O1': (-18, -90),
    'P2': (18, -54),  'P4': (36, -54),  'P6': (54, -54),  'P8': (90, -54),
    'PO4': (36, -72), 'PO8': (72, -72), 'O2': (18, -90),
    'FT9': (-108, 36), 'FT10': (108, 36), 'TP9': (-108, -36), 'TP10': (108, -36),
    'Iz': (0, -108),
}


def _deg2xy(az, el):
    az_r = np.radians(az)
    r    = np.radians(90 - abs(el)) / (np.pi / 2)
    return float(r * np.sin(az_r)), float(r * np.cos(az_r))


def _make_pos():
    pos = np.zeros((64, 2))
    for i, ch in enumerate(CH_NAMES):
        if ch in _POS_DEG:
            pos[i] = _deg2xy(*_POS_DEG[ch])
    return pos


POS_2D = _make_pos()


def _topomap(ax, values, vmin=None, vmax=None, cmap='RdBu_r', title=''):
    from scipy.interpolate import griddata
    xi, yi = np.meshgrid(np.linspace(-1.1, 1.1, 200), np.linspace(-1.1, 1.1, 200))
    zi = griddata(POS_2D, values, (xi, yi), method='cubic')
    zi[np.sqrt(xi**2 + yi**2) > 1.05] = np.nan

    if vmin is None:
        vmax = max(abs(np.nanmin(zi)), abs(np.nanmax(zi)))
        vmin = -vmax

    im = ax.contourf(xi, yi, zi, levels=64, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=1.5)
    ax.plot([0, 0.05, 0], [0.98, 1.08, 0.98], 'k-', lw=1.5)
    ax.plot([-1.03, -1.1, -1.03], [-0.08, 0, 0.08], 'k-', lw=1.2)
    ax.plot([1.03,  1.1,  1.03],  [-0.08, 0, 0.08], 'k-', lw=1.2)
    ax.scatter(POS_2D[:, 0], POS_2D[:, 1], c='k', s=6, zorder=5, alpha=0.5)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=10)
    return im


# ═══════════════════════════════════════════════════════════
#  Figures
# ═══════════════════════════════════════════════════════════

def fig_erd_timecourse(group: dict, out_dir: Path) -> None:
    """ERD time course at C3, Cz, C4 for mu and beta bands — 2×2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey='row')
    bands   = ['mu', 'beta']
    classes = ['left', 'right']
    cls_colors = {'left': '#2196F3', 'right': '#F44336'}
    ch_styles  = {'C3': '-', 'Cz': '--', 'C4': ':'}
    labels     = ['Left MI', 'Right MI']

    for row, band in enumerate(bands):
        for col, cls in enumerate(classes):
            ax = axes[row, col]
            for ch, style in ch_styles.items():
                idx  = CH_IDX[ch]
                mean = group[f'{band}_erd_{cls}']['mean'][idx]
                sem  = group[f'{band}_erd_{cls}']['sem'][idx]
                c    = cls_colors[cls]
                ax.plot(T_AXIS, mean, style, color=c, linewidth=1.5, label=ch)
                ax.fill_between(T_AXIS, mean - sem, mean + sem,
                                color=c, alpha=0.15)

            ax.axhline(0, color='gray', linewidth=0.8, linestyle='-')
            ax.axvline(0, color='black', linewidth=1.0, linestyle='--', alpha=0.7,
                       label='Cue onset')
            ax.set_xlim(EPOCH_TMIN, EPOCH_TMAX)
            ax.set_xlabel('Time (s)', fontsize=9)
            band_label = 'Mu (8–12 Hz)' if band == 'mu' else 'Beta (13–30 Hz)'
            ax.set_title(f'{band_label} — {labels[col]}', fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel('ERD (dB)', fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='upper right')

    # Shade MI window
    for ax in axes.ravel():
        ax.axvspan(0.5, 4.0, alpha=0.06, color='green', label='MI window')

    fig.suptitle('ERD Time Course — Group Average (n subjects)\n'
                 'Negative dB = desynchronization (expected during MI)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'fig_erd_timecourse.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'fig_erd_timecourse.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  저장: {path}")


def fig_erd_topomap(group: dict, band: str, out_dir: Path) -> None:
    """Topographic ERD maps at baseline, early MI, peak MI windows."""
    windows = [
        ('Baseline\n(−0.5 to 0 s)',  (T_AXIS >= -0.5) & (T_AXIS < 0.0)),
        ('Early MI\n(0.5 to 1.5 s)', (T_AXIS >= 0.5)  & (T_AXIS < 1.5)),
        ('Peak MI\n(1.5 to 3.0 s)',  (T_AXIS >= 1.5)  & (T_AXIS < 3.0)),
        ('Late MI\n(3.0 to 4.0 s)',  (T_AXIS >= 3.0)  & (T_AXIS <= 4.0)),
    ]
    classes   = ['left', 'right']
    cls_names = ['Left MI', 'Right MI']

    all_vals = []
    for cls in classes:
        mean = group[f'{band}_erd_{cls}']['mean']    # (64, T)
        for _, mask in windows:
            all_vals.append(mean[:, mask].mean(-1))
    vmax = float(np.percentile(np.abs(np.concatenate(all_vals)), 98))
    vmin = -vmax

    fig, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    im = None
    for row, (cls, cls_name) in enumerate(zip(classes, cls_names)):
        mean = group[f'{band}_erd_{cls}']['mean']    # (64, T)
        for col, (win_label, mask) in enumerate(windows):
            vals = mean[:, mask].mean(-1)
            im = _topomap(axes[row, col], vals, vmin=vmin, vmax=vmax,
                          title=win_label)
            if col == 0:
                axes[row, col].set_ylabel(cls_name, fontsize=11, fontweight='bold')

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.6)
        band_str = 'Mu (8–12 Hz)' if band == 'mu' else 'Beta (13–30 Hz)'
        cbar.set_label(f'{band_str} ERD (dB)', fontsize=9)

    band_str = 'Mu (8–12 Hz)' if band == 'mu' else 'Beta (13–30 Hz)'
    n = group['n_subjects']
    fig.suptitle(f'{band_str} ERD Topomaps — {n} subjects\n'
                 'Blue=desynchronization (ERD), Red=synchronization (ERS)',
                 fontsize=12, fontweight='bold')
    fname = f'fig_erd_topomap_{band}.png'
    fig.savefig(out_dir / fname, dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / f'fig_erd_topomap_{band}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  저장: {out_dir / fname}")


def fig_erd_tf_spectrogram(group_raw: dict, out_dir: Path) -> None:
    """STFT ERD spectrogram at C3 and C4 — 2 classes × 2 channels."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    cls_names  = ['Left MI', 'Right MI']
    chs        = ['C3', 'C4']
    ch_idxs    = [C3_IDX, C4_IDX]
    vmax       = 200.0

    for row, (cls, cls_name) in enumerate(zip(['left', 'right'], cls_names)):
        raw = group_raw[cls]   # list of (N, 64, T) arrays from each subject
        for col, (ch, ch_idx) in enumerate(zip(chs, ch_idxs)):
            ax = axes[row, col]
            # Collect all trials for this channel
            all_trials = np.concatenate([r[:, ch_idx, :] for r in raw], axis=0)
            # Limit to 500 trials max for speed
            if len(all_trials) > 500:
                rng = np.random.RandomState(42)
                sel = rng.choice(len(all_trials), 500, replace=False)
                all_trials = all_trials[sel]

            freqs, times, erd_tf = compute_tf_erd(all_trials)

            # Restrict to 4–40 Hz (physiological range)
            f_mask = (freqs >= 4) & (freqs <= 40)
            im = ax.pcolormesh(times, freqs[f_mask], erd_tf[f_mask, :],
                               cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                               shading='gouraud')
            ax.axvline(0, color='white', linewidth=1.5, linestyle='--')
            ax.axhline(8,  color='white', linewidth=0.8, linestyle=':', alpha=0.8)
            ax.axhline(12, color='white', linewidth=0.8, linestyle=':', alpha=0.8)
            ax.axhline(15, color='yellow', linewidth=0.8, linestyle=':', alpha=0.8)
            ax.axhline(30, color='yellow', linewidth=0.8, linestyle=':', alpha=0.8)
            ax.set_xlim(EPOCH_TMIN, EPOCH_TMAX)
            ax.set_ylim(4, 40)
            ax.set_title(f'{cls_name} — {ch}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=9)
            if col == 0:
                ax.set_ylabel('Frequency (Hz)', fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.8).set_label('ERD (%)', fontsize=8)

    # Band labels
    for ax in axes.ravel():
        ax.text(EPOCH_TMAX - 0.1, 10, 'μ', ha='right', va='center',
                color='white', fontsize=11, fontweight='bold')
        ax.text(EPOCH_TMAX - 0.1, 22, 'β', ha='right', va='center',
                color='yellow', fontsize=11, fontweight='bold')

    fig.suptitle('Time-Frequency ERD Spectrogram — Group Average\n'
                 'White dotted: mu band (8–12 Hz) | Yellow dotted: beta band (15–30 Hz)',
                 fontsize=12, fontweight='bold')
    path = out_dir / 'fig_erd_tf_spectrogram.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'fig_erd_tf_spectrogram.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  저장: {path}")


def fig_erd_literature_compare(sig_df: pd.DataFrame, out_dir: Path) -> None:
    """Table figure comparing our ERD values with published literature."""

    def lookup(ch: str, band: str, cls: str) -> str:
        row = sig_df[(sig_df['channel'] == ch) & (sig_df['band'] == band)
                     & (sig_df['class'] == cls)]
        if row.empty:
            return 'N/A'
        v, sd = float(row['mean_erd'].iloc[0]), float(row['sd'].iloc[0])
        sig   = row['significance'].iloc[0]
        return f'{v:+.1f}±{sd:.1f} {sig}'

    lit = [
        ('C3 Mu — Left MI',    'Our study (n=52)',       lookup('C3','mu','left')),
        ('C3 Mu — Left MI',    'Neuper et al. 2001',     '−35±12%'),
        ('C3 Mu — Left MI',    'Pfurtscheller & LS 1999','~−30%'),
        ('C4 Mu — Right MI',   'Our study (n=52)',       lookup('C4','mu','right')),
        ('C4 Mu — Right MI',   'Neuper et al. 2001',     '−38±14%'),
        ('C3 Beta — Left MI',  'Our study (n=52)',       lookup('C3','beta','left')),
        ('C3 Beta — Left MI',  'Pfurtscheller & LS 1999','~−40%'),
        ('C4 Beta — Right MI', 'Our study (n=52)',       lookup('C4','beta','right')),
        ('C4 Beta — Right MI', 'Pfurtscheller & LS 1999','~−42%'),
    ]
    df_lit = pd.DataFrame(lit, columns=['Measure', 'Source', 'ERD (mean±SD)'])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis('off')
    tbl = ax.table(
        cellText=df_lit.values,
        colLabels=df_lit.columns,
        cellLoc='center', loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.6)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white', fontweight='bold')
        elif row > 0 and df_lit.iloc[row - 1]['Source'] == 'Our study':
            cell.set_facecolor('#E3F2FD')
    ax.set_title('ERD Comparison with Published Literature\n'
                 'Values: peak ERD (%) during motor imagery window (0.5–3.0 s)',
                 fontsize=11, fontweight='bold', pad=25)
    path = out_dir / 'fig_erd_literature_compare.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  저장: {path}")


# ═══════════════════════════════════════════════════════════
#  Statistics
# ═══════════════════════════════════════════════════════════

def compute_statistics(all_records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-channel ERD stats and Wilcoxon significance at MI window."""
    mi_mask = (T_AXIS >= 0.5) & (T_AXIS < 3.0)
    sig_rows = []

    for ch_name, ch_idx in MOTOR_CHS.items():
        for band in ('mu', 'beta'):
            for cls in ('left', 'right'):
                key  = f'{band}_erd_{cls}'
                vals = np.array([r[key][ch_idx, mi_mask].mean()
                                  for r in all_records])   # (n_subjects,)
                mu_s = float(vals.mean())
                sd_s = float(vals.std())
                # Wilcoxon: test if ERD < 0 (one-sided: alternative='less')
                try:
                    _, p = wilcoxon(vals, alternative='less')
                except Exception:
                    p = np.nan
                sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
                sig_rows.append({
                    'channel': ch_name, 'band': band, 'class': cls,
                    'mean_erd': round(mu_s, 3), 'sd': round(sd_s, 3),
                    'wilcoxon_p': round(float(p), 4) if not np.isnan(p) else np.nan,
                    'significance': sig,
                    'n_subjects': len(all_records),
                })

    stats_df = pd.DataFrame(sig_rows)
    return stats_df, stats_df   # return same df; caller uses stats_df for table


def compute_laterality(all_records: list[dict]) -> pd.DataFrame:
    """
    Laterality Index (LI) of ERD at C3 vs C4.
    LI = ERD(contra) - ERD(ipsi)
    For Left MI:  contra=C4, ipsi=C3  → negative LI expected (C4 more desynced)
    For Right MI: contra=C3, ipsi=C4
    """
    mi_mask = (T_AXIS >= 0.5) & (T_AXIS < 3.0)
    rows = []
    for r in all_records:
        for band in ('mu', 'beta'):
            c3_left  = r[f'{band}_erd_left'][C3_IDX, mi_mask].mean()
            c4_left  = r[f'{band}_erd_left'][C4_IDX, mi_mask].mean()
            c3_right = r[f'{band}_erd_right'][C3_IDX, mi_mask].mean()
            c4_right = r[f'{band}_erd_right'][C4_IDX, mi_mask].mean()
            rows.append({
                'sid': r['sid'], 'band': band,
                # Left MI: C4 is contralateral
                'LI_left':  float(c4_left  - c3_left),
                # Right MI: C3 is contralateral
                'LI_right': float(c3_right - c4_right),
            })
    return pd.DataFrame(rows)


def fig_laterality_analysis(lat_df: pd.DataFrame, out_dir: Path) -> None:
    """Box+scatter plot of C3 vs C4 laterality index per band."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    bands     = ['mu', 'beta']
    cls_names = ['Left MI', 'Right MI']
    cls_keys  = ['LI_left', 'LI_right']
    colors    = ['#2196F3', '#F44336']

    for ax, band in zip(axes, bands):
        df_b = lat_df[lat_df['band'] == band]
        for i, (cls, key, color) in enumerate(zip(cls_names, cls_keys, colors)):
            vals = df_b[key].values
            x = np.full(len(vals), i)
            ax.scatter(x + np.random.uniform(-0.1, 0.1, len(x)), vals,
                       alpha=0.4, s=20, color=color)
            ax.boxplot(vals, positions=[i], widths=0.35,
                       patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.3),
                       medianprops=dict(color='black', linewidth=2))
            try:
                _, p = wilcoxon(vals, alternative='less')
                sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
                ax.text(i, np.nanmin(vals) - 0.5, sig, ha='center', fontsize=9)
            except Exception:
                pass

        ax.axhline(0, color='gray', linewidth=1, linestyle='--')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(cls_names, fontsize=10)
        band_str = 'Mu (8–12 Hz)' if band == 'mu' else 'Beta (13–30 Hz)'
        ax.set_title(f'{band_str} Laterality Index\n'
                     'ERD_contra − ERD_ipsi (dB)\nNegative = contralateral desynced',
                     fontsize=10, fontweight='bold')
        ax.set_ylabel('LI = ERD_contra − ERD_ipsi (dB)', fontsize=9)

    fig.suptitle(f'Contralateral Laterality of Motor Imagery ERD\n'
                 f'n={len(lat_df["sid"].unique())} subjects',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'fig_erd_laterality.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'fig_erd_laterality.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  저장: {path}")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='BCI_Research')
    p.add_argument('--sids', default='all')
    p.add_argument('--max_tf_subjects', type=int, default=20,
                   help='Max subjects for STFT spectrogram (memory limit)')
    return p.parse_args()


def parse_sids(spec: str) -> list[int]:
    if spec == 'all':
        return list(range(1, 53))
    if '-' in spec:
        a, b = spec.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(',')]


def main():
    args     = parse_args()
    root     = Path(args.root)
    data_dir = str(root / 'preprocessed' / 'member_A')
    out_dir  = root / 'results' / 'xai_erd'
    fig_dir  = out_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ERD Validation  |  scipy {__import__('scipy').__version__}")
    print(f"{'='*60}\n")

    sids = parse_sids(args.sids)
    all_records = []
    raw_left_tf  = []   # subset for STFT
    raw_right_tf = []

    for sid in sids:
        r = load_subject_erd(sid, data_dir)
        if r is None:
            print(f"  [s{sid:02d}] 없음 — 건너뜀")
            continue
        all_records.append(r)
        if len(raw_left_tf) < args.max_tf_subjects:
            raw_left_tf.append(r['raw_left'])
            raw_right_tf.append(r['raw_right'])
        print(f"  [s{sid:02d}] 로드 완료  "
              f"n_left={r['n_left']}  n_right={r['n_right']}")

    if not all_records:
        print("처리된 피험자 없음.")
        return

    n = len(all_records)
    print(f"\n  총 {n}명 피험자 처리 완료")

    # Group averages: mean ± SEM across subjects
    def group_stat(key: str) -> dict:
        stack = np.stack([r[key] for r in all_records])   # (S, 64, T)
        return {'mean': stack.mean(0), 'sem': stack.std(0) / np.sqrt(n)}

    group = {
        'mu_erd_left':    group_stat('mu_erd_left'),
        'mu_erd_right':   group_stat('mu_erd_right'),
        'beta_erd_left':  group_stat('beta_erd_left'),
        'beta_erd_right': group_stat('beta_erd_right'),
        'n_subjects': n,
    }

    # Key channel report
    mi_mask = (T_AXIS >= 0.5) & (T_AXIS < 3.0)
    print(f"\n  --- Group-average ERD at MI window (0.5–3.0 s) ---")
    for ch, idx in [('C3', C3_IDX), ('Cz', CZ_IDX), ('C4', C4_IDX)]:
        mu_l  = group['mu_erd_left']['mean'][idx, mi_mask].mean()
        mu_r  = group['mu_erd_right']['mean'][idx, mi_mask].mean()
        be_l  = group['beta_erd_left']['mean'][idx, mi_mask].mean()
        be_r  = group['beta_erd_right']['mean'][idx, mi_mask].mean()
        print(f"  {ch}: mu_Left={mu_l:+.1f}%  mu_Right={mu_r:+.1f}%  "
              f"beta_Left={be_l:+.1f}%  beta_Right={be_r:+.1f}%")

    # Statistics
    print(f"\n  통계 계산 중...")
    stats_df, sig_df = compute_statistics(all_records)
    stats_df.to_csv(out_dir / 'erd_significance.csv', index=False)
    print(f"  저장: {out_dir / 'erd_significance.csv'}")

    # Laterality analysis
    lat_df = compute_laterality(all_records)
    lat_df.to_csv(out_dir / 'erd_laterality.csv', index=False)
    print(f"  저장: {out_dir / 'erd_laterality.csv'}")

    # Print significance summary
    sig_motor = sig_df[sig_df['channel'].isin(['C3', 'C4'])]
    print(f"\n  --- Wilcoxon signed-rank (ERD < 0 dB) ---")
    for _, row in sig_motor.iterrows():
        print(f"    {row['channel']} {row['band']:4s} {row['class']:5s}: "
              f"ERD={row['mean_erd']:+.2f} ± {row['sd']:.2f} dB  "
              f"p={row['wilcoxon_p']:.4f} {row['significance']}")

    # Print laterality summary
    print(f"\n  --- Laterality Index (ERD_contra - ERD_ipsi) ---")
    for band in ('mu', 'beta'):
        df_b = lat_df[lat_df['band'] == band]
        for key, cls in [('LI_left', 'Left MI'), ('LI_right', 'Right MI')]:
            vals = df_b[key].values
            print(f"    {band:4s} {cls}: LI={vals.mean():+.2f} ± {vals.std():.2f} dB")

    # Figures
    print(f"\n  피규어 생성 중...")
    fig_erd_timecourse(group, fig_dir)
    fig_erd_topomap(group, 'mu',   fig_dir)
    fig_erd_topomap(group, 'beta', fig_dir)
    fig_laterality_analysis(lat_df, fig_dir)

    group_raw = {'left': raw_left_tf, 'right': raw_right_tf}
    fig_erd_tf_spectrogram(group_raw, fig_dir)
    fig_erd_literature_compare(
        sig_df[sig_df['channel'].isin(['C3', 'C4'])],
        fig_dir,
    )

    # Save group ERD arrays
    np.savez_compressed(
        out_dir / 'erd_group_averages.npz',
        mu_erd_left_mean    = group['mu_erd_left']['mean'],
        mu_erd_left_sem     = group['mu_erd_left']['sem'],
        mu_erd_right_mean   = group['mu_erd_right']['mean'],
        mu_erd_right_sem    = group['mu_erd_right']['sem'],
        beta_erd_left_mean  = group['beta_erd_left']['mean'],
        beta_erd_left_sem   = group['beta_erd_left']['sem'],
        beta_erd_right_mean = group['beta_erd_right']['mean'],
        beta_erd_right_sem  = group['beta_erd_right']['sem'],
        t_axis              = T_AXIS,
        ch_names            = np.array(CH_NAMES),
    )
    print(f"  저장: {out_dir / 'erd_group_averages.npz'}")

    print(f"\n{'='*60}")
    print(f"  ERD Validation 완료. 출력: {out_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
