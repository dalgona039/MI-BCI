#!/usr/bin/env python3
"""
xai_gradcam_eeg.py
==================
Grad-CAM channel saliency maps from EEGNet block1 activations.

Registers a forward/backward hook on EEGNetEncoder.block1 and computes
per-channel importance:
  CAM[ch] = ReLU( sum_filter( mean_time( grad × activation ) ) )

Outputs:
  BCI_Research/results/gradcam/
    sXX.npz                   — left_mi / right_mi (64,) arrays per subject
    channel_positions.csv     — (x, y) 2D topomap coords for figure6
  BCI_Research/results/xai_gradcam/
    figures/
      fig_gradcam_group.png   — group-average topomaps (High / All / Low perf.)
      fig_gradcam_laterality.png — C3 vs C4 Grad-CAM bar chart

Then run figure6_gradcam_topomaps.py for the full JNE-quality figure.

Colab A100:
  !pip install mne -q   # optional but improves topomap rendering
  !python src/xai_gradcam_eeg.py --root /content/drive/MyDrive/BCI_Research
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════
#  Channel metadata
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

# Standard 2-D topomap coordinates (azimuth projection, unit head circle).
# x<0: left hemisphere; y>0: anterior; y<0: posterior.
_POS_DEG = {
    # Midline
    'Fpz': (0, 90), 'Fz': (0, 54), 'FCz': (0, 36), 'Cz': (0, 0),
    'CPz': (0, -36), 'Pz': (0, -54), 'POz': (0, -72), 'Oz': (0, -90),
    # Left C-row
    'C1': (-18, 0), 'C3': (-36, 0), 'C5': (-54, 0), 'T7': (-90, 0),
    # Right C-row
    'C2': (18, 0),  'C4': (36, 0),  'C6': (54, 0),  'T8': (90, 0),
    # Fronto-central
    'FC1': (-18, 36), 'FC3': (-36, 36), 'FC5': (-54, 36), 'FT7': (-90, 36),
    'FC2': (18, 36),  'FC4': (36, 36),  'FC6': (54, 36),  'FT8': (90, 36),
    # Centro-parietal
    'CP1': (-18, -36), 'CP3': (-36, -36), 'CP5': (-54, -36), 'TP7': (-90, -36),
    'CP2': (18, -36),  'CP4': (36, -36),  'CP6': (54, -36),  'TP8': (90, -36),
    # Frontal
    'F1': (-18, 54), 'F3': (-36, 54), 'F5': (-54, 54), 'F7': (-90, 54),
    'AF3': (-36, 72), 'AF7': (-72, 72), 'Fp1': (-18, 90),
    'F2': (18, 54),  'F4': (36, 54),  'F6': (54, 54),  'F8': (90, 54),
    'AF4': (36, 72), 'AF8': (72, 72),  'Fp2': (18, 90),
    # Parietal
    'P1': (-18, -54), 'P3': (-36, -54), 'P5': (-54, -54), 'P7': (-90, -54),
    'PO3': (-36, -72), 'PO7': (-72, -72), 'O1': (-18, -90),
    'P2': (18, -54),  'P4': (36, -54),  'P6': (54, -54),  'P8': (90, -54),
    'PO4': (36, -72), 'PO8': (72, -72), 'O2': (18, -90),
    # Other
    'FT9': (-108, 36), 'FT10': (108, 36), 'TP9': (-108, -36), 'TP10': (108, -36),
    'Iz': (0, -108), 'C1 ': (-18, 0),
}


def _deg_to_xy(az_deg: float, el_deg: float) -> tuple[float, float]:
    """Azimuth-elevation to 2-D head surface projection."""
    az = np.radians(az_deg)
    el = np.radians(90 - abs(el_deg))
    r  = el / (np.pi / 2)        # normalized radius 0..1
    return float(r * np.sin(az)), float(r * np.cos(az))


def build_positions() -> np.ndarray:
    """Return (64, 2) array of (x, y) topomap positions."""
    pos = np.zeros((64, 2))
    for i, ch in enumerate(CH_NAMES):
        if ch in _POS_DEG:
            az, el = _POS_DEG[ch]
            pos[i] = _deg_to_xy(az, el)
        else:
            # Fallback: place at (0, 0) — will show as center dot
            pos[i] = (0.0, 0.0)
    return pos


def save_channel_positions(path: Path) -> None:
    pos = build_positions()
    df  = pd.DataFrame({'channel': CH_NAMES, 'x': pos[:, 0], 'y': pos[:, 1]})
    df.to_csv(path, index=False)


# ═══════════════════════════════════════════════════════════
#  Model definition (identical to inference.py)
# ═══════════════════════════════════════════════════════════

DEFAULT_CFG = {
    "n_eeg_ch": 64, "n_emg_ch": 4, "n_times": 2304, "n_classes": 2,
    "emg_ds_factor": 8, "eegnet_F1": 8, "eegnet_D": 2, "eegnet_kern_len": 256,
    "eegnet_dropout": 0.5, "lstm_hidden": 128, "lstm_layers": 2,
    "lstm_dropout": 0.3, "clf_dropout": 0.3, "feat_dim": 256,
}
DEFAULT_CFG["n_times_emg"] = DEFAULT_CFG["n_times"] // DEFAULT_CFG["emg_ds_factor"]


class EEGNetEncoder(nn.Module):
    def __init__(self, n_ch, n_times, F1=8, D=2, kern_len=256, dropout=0.5, feat_dim=256):
        super().__init__()
        F2 = F1 * D
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F2, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        flat = self._flat(n_ch, n_times)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(flat, feat_dim), nn.ELU())

    def _flat(self, n_ch, n_times):
        with torch.no_grad():
            return self.block3(self.block2(self.block1(torch.zeros(1, 1, n_ch, n_times)))).numel()

    def forward(self, x):
        return self.fc(self.block3(self.block2(self.block1(x.unsqueeze(1)))))


class EMGBiLSTMEncoder(nn.Module):
    def __init__(self, n_ch=4, hidden=128, n_layers=2, dropout=0.3, feat_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(n_ch, hidden, n_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden * 2)
        self.fc   = nn.Sequential(nn.Linear(hidden * 2, feat_dim), nn.ELU())

    def forward(self, x):
        out, _ = self.lstm(x.permute(0, 2, 1))
        return self.fc(self.norm(out[:, -1, :]))


class SoftmaxAttentionFusion(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        self.W_eeg = nn.Linear(feat_dim, feat_dim)
        self.W_emg = nn.Linear(feat_dim, feat_dim)
        self.attn  = nn.Linear(feat_dim * 2, 2)

    def forward(self, h_eeg, h_emg):
        w = F.softmax(self.attn(torch.cat([h_eeg, h_emg], dim=-1)), dim=-1)
        return w[:, 0:1] * self.W_eeg(h_eeg) + w[:, 1:2] * self.W_emg(h_emg), w


class HybridBCIModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        fd = cfg["feat_dim"]
        self.eeg_enc = EEGNetEncoder(cfg["n_eeg_ch"], cfg["n_times"], cfg["eegnet_F1"],
                                      cfg["eegnet_D"], cfg["eegnet_kern_len"],
                                      cfg["eegnet_dropout"], fd)
        self.emg_enc = EMGBiLSTMEncoder(cfg["n_emg_ch"], cfg["lstm_hidden"],
                                         cfg["lstm_layers"], cfg["lstm_dropout"], fd)
        self.fusion  = SoftmaxAttentionFusion(fd)
        self.clf     = nn.Sequential(nn.Linear(fd, 128), nn.ELU(),
                                      nn.Dropout(cfg["clf_dropout"]),
                                      nn.Linear(128, cfg["n_classes"]))

    def forward(self, eeg, emg):
        fused, w = self.fusion(self.eeg_enc(eeg), self.emg_enc(emg))
        return self.clf(fused), w


# ═══════════════════════════════════════════════════════════
#  Grad-CAM hook
# ═══════════════════════════════════════════════════════════

class GradCAMHook:
    """Hook on EEGNetEncoder.block1; captures activations & gradients."""

    def __init__(self, layer: nn.Module):
        self._act  = None
        self._grad = None
        self._fwd_h = layer.register_forward_hook(self._save_act)
        self._bwd_h = layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self._act = out.detach()           # (B, F1, 64, T')

    def _save_grad(self, module, grad_in, grad_out):
        self._grad = grad_out[0].detach()  # (B, F1, 64, T')

    def remove(self):
        self._fwd_h.remove()
        self._bwd_h.remove()

    def channel_cam(self) -> torch.Tensor:
        """
        Returns (B, 64) per-channel Grad-CAM importance.
        Formula: ReLU( sum_F1( mean_T( grad × act ) ) )
        """
        # (B, F1, 64, T')  →  mean over T'  →  (B, F1, 64)
        weights = (self._grad * self._act).mean(dim=-1)  # (B, F1, 64)
        cam     = weights.sum(dim=1)                      # (B, 64)
        return F.relu(cam)


# ═══════════════════════════════════════════════════════════
#  Per-subject Grad-CAM
# ═══════════════════════════════════════════════════════════

def gradcam_subject(sid: int, data_dir: str, ckpt_dir: str,
                    cfg: dict, device: torch.device,
                    batch_size: int = 32) -> dict | None:
    h5_path   = os.path.join(data_dir, f"sub-{sid:02d}_member_A.h5")
    ckpt_path = os.path.join(ckpt_dir, f"best_s{sid:02d}.pt")
    for p in (h5_path, ckpt_path):
        if not os.path.exists(p):
            print(f"  [s{sid:02d}] 없음: {os.path.basename(p)} — 건너뜀")
            return None

    ds = cfg["emg_ds_factor"]
    with h5py.File(h5_path, "r") as f:
        eeg = f["eeg/epochs"][:].astype(np.float32)
        emg = f["emg/epochs"][:, :, ::ds].astype(np.float32)
        lbl = f["labels"][:].astype(np.int64) - 1   # 0=Left, 1=Right

    model = HybridBCIModel(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.train()   # need gradients through BN

    hook = GradCAMHook(model.eeg_enc.block1)

    # CAM per class: accumulate (n_trials, 64) then average
    cam_per_class = {0: [], 1: []}
    n = len(lbl)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b_eeg = torch.tensor(eeg[start:end], device=device, requires_grad=False)
        b_emg = torch.tensor(emg[start:end], device=device, requires_grad=False)
        b_lbl = lbl[start:end]

        for cls in range(2):
            model.zero_grad()
            logits, _ = model(b_eeg, b_emg)             # (B, 2)
            score = logits[:, cls].sum()                 # scalar
            score.backward(retain_graph=(cls == 0))
            cam = hook.channel_cam().cpu().numpy()       # (B, 64)
            cam_per_class[cls].append(cam)

    hook.remove()

    left_mi  = np.concatenate(cam_per_class[0], axis=0).mean(axis=0)   # (64,)
    right_mi = np.concatenate(cam_per_class[1], axis=0).mean(axis=0)

    # Normalize to [0, 1]
    left_mi  = (left_mi  - left_mi.min())  / (left_mi.max()  - left_mi.min()  + 1e-8)
    right_mi = (right_mi - right_mi.min()) / (right_mi.max() - right_mi.min() + 1e-8)

    print(f"  [s{sid:02d}] Grad-CAM 완료  "
          f"Left top: {CH_NAMES[left_mi.argmax()]}  "
          f"Right top: {CH_NAMES[right_mi.argmax()]}")

    return {'sid': sid, 'left_mi': left_mi, 'right_mi': right_mi, 'labels': lbl}


# ═══════════════════════════════════════════════════════════
#  Topomap (matplotlib only, no MNE dependency)
# ═══════════════════════════════════════════════════════════

def _topomap_ax(ax: plt.Axes, values: np.ndarray, pos: np.ndarray,
                cmap: str = 'RdBu_r', title: str = '') -> None:
    """Simple 2D scattered topomap with head outline."""
    from scipy.interpolate import griddata

    xi = np.linspace(-1.1, 1.1, 200)
    yi = np.linspace(-1.1, 1.1, 200)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata(pos, values, (xi, yi), method='cubic')

    # Mask outside unit circle
    mask = np.sqrt(xi**2 + yi**2) > 1.05
    zi[mask] = np.nan

    im = ax.contourf(xi, yi, zi, levels=64, cmap=cmap, extend='both')
    ax.contour(xi, yi, zi, levels=10, colors='k', linewidths=0.3, alpha=0.3)

    # Head outline
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1.5)
    # Nose
    ax.plot([0, 0.05, 0], [0.98, 1.08, 0.98], 'k-', linewidth=1.5)
    # Ears
    ax.plot([-1.03, -1.1, -1.03], [-0.08, 0, 0.08], 'k-', linewidth=1.2)
    ax.plot([1.03,  1.1,  1.03],  [-0.08, 0, 0.08], 'k-', linewidth=1.2)

    # Channel dots
    ax.scatter(pos[:, 0], pos[:, 1], c='k', s=8, zorder=5, alpha=0.6)

    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=10)
    return im


def fig_gradcam_group(group_means: dict, pos: np.ndarray, out_dir: Path) -> None:
    """2×3 topomap grid: rows=class, cols=group."""
    groups    = ['High performers', 'All subjects', 'Low performers']
    cls_names = ['Left MI', 'Right MI']
    cls_keys  = ['left_mi', 'right_mi']

    all_vals = np.concatenate([v.ravel() for v in group_means.values()])
    vmax = float(np.percentile(np.abs(all_vals[np.isfinite(all_vals)]), 99))

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    im = None
    for row, (key, cls) in enumerate(zip(cls_keys, cls_names)):
        for col, grp in enumerate(groups):
            if grp not in group_means:
                axes[row, col].axis('off')
                continue
            vals = group_means[grp][key]
            im = _topomap_ax(axes[row, col], vals, pos,
                             cmap='RdBu_r', title=f'{grp}')
            if col == 0:
                axes[row, col].set_ylabel(cls, fontsize=11, fontweight='bold')

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, orientation='vertical',
                            shrink=0.6, pad=0.02)
        cbar.set_label('Normalized Grad-CAM activation', fontsize=9)
    fig.suptitle('Grad-CAM EEG Channel Topomaps — HybridBCIModel EEGNet',
                 fontsize=13, fontweight='bold')
    path = out_dir / 'fig_gradcam_group.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'fig_gradcam_group.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  저장: {path}")


def fig_gradcam_laterality(records: list[dict], out_dir: Path) -> None:
    """C3 vs C4 Grad-CAM bar chart."""
    c3, c4 = 18, 22  # CH_NAMES indices
    left_c3  = np.array([r['left_mi'][c3]  for r in records])
    left_c4  = np.array([r['left_mi'][c4]  for r in records])
    right_c3 = np.array([r['right_mi'][c3] for r in records])
    right_c4 = np.array([r['right_mi'][c4] for r in records])

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(4)
    vals = [left_c3, left_c4, right_c3, right_c4]
    colors = ['#2196F3', '#2196F3', '#F44336', '#F44336']
    alphas = [0.9, 0.5, 0.9, 0.5]
    labels = ['Left MI\nC3 (ipsi)', 'Left MI\nC4 (contra)',
              'Right MI\nC3 (contra)', 'Right MI\nC4 (ipsi)']
    for i, (v, c, a, l) in enumerate(zip(vals, colors, alphas, labels)):
        ax.bar(x[i], v.mean(), color=c, alpha=a, yerr=v.std(), capsize=4,
               ecolor='black', label=l if i < 2 else '_nolegend_')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Mean Grad-CAM activation (normalized)')
    ax.set_title('Contralateral Laterality in Grad-CAM\n'
                 'Expected: Left MI → C4 > C3 | Right MI → C3 > C4',
                 fontweight='bold')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='#2196F3', label='Left MI'),
                        Patch(facecolor='#F44336', label='Right MI')])
    plt.tight_layout()
    path = out_dir / 'fig_gradcam_laterality.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'fig_gradcam_laterality.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  저장: {path}")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='BCI_Research')
    p.add_argument('--sids', default='all')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--batch', type=int, default=32)
    return p.parse_args()


def parse_sids(spec: str) -> list[int]:
    if spec == 'all':
        return list(range(1, 53))
    if '-' in spec:
        a, b = spec.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(',')]


def main():
    args = parse_args()
    root = Path(args.root)
    data_dir = str(root / 'preprocessed' / 'member_A')
    ckpt_dir = str(root / 'results' / 'checkpoints_A')
    acc_csv  = root / 'results' / 'ablation' / 'ablation_results.csv'

    cam_dir  = root / 'results' / 'gradcam'
    out_dir  = root / 'results' / 'xai_gradcam'
    fig_dir  = out_dir / 'figures'
    cam_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  Grad-CAM  |  device={device}")
    print(f"{'='*60}\n")

    cfg  = DEFAULT_CFG.copy()
    sids = parse_sids(args.sids)

    # Save channel positions CSV (needed by figure6_gradcam_topomaps.py)
    pos_csv = cam_dir / 'channel_positions.csv'
    save_channel_positions(pos_csv)
    print(f"  채널 위치 저장: {pos_csv}")

    pos = build_positions()   # (64, 2)

    records = []
    for sid in sids:
        r = gradcam_subject(sid, data_dir, ckpt_dir, cfg, device, args.batch)
        if r is None:
            continue
        records.append(r)
        # Save per-subject NPZ for figure6_gradcam_topomaps.py
        np.savez_compressed(
            cam_dir / f"s{sid:02d}.npz",
            left_mi=r['left_mi'],
            right_mi=r['right_mi'],
        )

    if not records:
        print("처리된 피험자 없음.")
        sys.exit(1)

    print(f"\n  완료: {len(records)}/{len(sids)} 피험자")
    print(f"  NPZ 저장 위치: {cam_dir}")

    # Build group means
    try:
        acc_df = pd.read_csv(acc_csv)
        high_sids = set(acc_df.nlargest(5, 'acc_fusion')['sid'].tolist())
        low_sids  = set(acc_df.nsmallest(5, 'acc_fusion')['sid'].tolist())
    except Exception:
        high_sids = set(range(1, 6))
        low_sids  = set(range(48, 53))

    def group_mean(key: str, sids_set: set) -> np.ndarray:
        arr = [r[key] for r in records if r['sid'] in sids_set]
        return np.nanmean(arr, axis=0) if arr else np.zeros(64)

    all_mean_left  = np.nanmean([r['left_mi']  for r in records], axis=0)
    all_mean_right = np.nanmean([r['right_mi'] for r in records], axis=0)

    group_means = {
        'High performers': {
            'left_mi':  group_mean('left_mi',  high_sids),
            'right_mi': group_mean('right_mi', high_sids),
        },
        'All subjects': {
            'left_mi':  all_mean_left,
            'right_mi': all_mean_right,
        },
        'Low performers': {
            'left_mi':  group_mean('left_mi',  low_sids),
            'right_mi': group_mean('right_mi', low_sids),
        },
    }

    print(f"\n  그룹 평균 Grad-CAM:")
    for grp, v in group_means.items():
        print(f"    {grp}: Left top={CH_NAMES[v['left_mi'].argmax()]}  "
              f"Right top={CH_NAMES[v['right_mi'].argmax()]}")

    # Figures
    fig_gradcam_group(group_means, pos, fig_dir)
    fig_gradcam_laterality(records, fig_dir)

    # Aggregate NPZ for reference
    np.savez_compressed(
        out_dir / 'gradcam_group_means.npz',
        **{f'{g}_{k}': v for g, d in group_means.items() for k, v in d.items()},
        ch_names=np.array(CH_NAMES),
        positions=pos,
    )

    print(f"\n{'='*60}")
    print(f"  Grad-CAM 완료.")
    print(f"  NPZ (figure6용): {cam_dir}")
    print(f"  figure6 실행 명령:")
    print(f"    python src/figure6_gradcam_topomaps.py \\")
    print(f"      --gradcam-dir {cam_dir} \\")
    print(f"      --positions-csv {pos_csv} \\")
    print(f"      --accuracy-csv {acc_csv} \\")
    print(f"      --output-dir {out_dir / 'figure6'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
