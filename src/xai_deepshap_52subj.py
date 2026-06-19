#!/usr/bin/env python3
"""
xai_deepshap_52subj.py
======================
DeepSHAP feature importance for HybridBCIModel — 52 LOSO subjects.

Colab A100 usage:
  !pip install shap -q
  !python src/xai_deepshap_52subj.py --root /content/drive/MyDrive/BCI_Research

Local CPU usage (subset for testing):
  python src/xai_deepshap_52subj.py --root BCI_Research --n_bg 20 --n_test 50

Outputs in <root>/results/xai_shap/:
  shap_channel_importance.npz   — mean |SHAP| per class × channel (52-subj aggregate)
  shap_per_subject.csv          — top-5 EEG + top-2 EMG channels per subject
  figures/
    fig_eeg_shap_topclass.png   — Left/Right MI EEG channel importance (bar)
    fig_emg_shap.png            — EMG channel importance by class
    fig_class_laterality.png    — Contralateral C3/C4 SHAP comparison
    fig_summary_table.png       — Top-10 features table (JNE ready)
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════
#  Cho2017 64-channel names (BrainProducts standard layout)
#  Index matches motor_protect_chs: C3=18, Cz=20, C4=22
# ═══════════════════════════════════════════════════════════

CH_NAMES = [
    'Fp1', 'AF7', 'AF3', 'F1',  'F3',  'F5',  'F7',   # 0-6
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',  # 7-13
    'FC6', 'FT8', 'T7',  'C5',  'C3',  'C1',  'Cz',   # 14-20
    'C2',  'C4',  'C6',  'T8',  'TP7', 'CP5', 'CP3',  # 21-27
    'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',          # 28-33
    'P7',  'P5',  'P3',  'P1',  'Pz',  'P2',           # 34-39
    'P4',  'P6',  'P8',  'PO7', 'PO3', 'POz',          # 40-45
    'PO4', 'PO8', 'O1',  'Oz',  'O2',  'Iz',           # 46-51
    'Fp2', 'AF8', 'AF4', 'F2',  'F4',  'F6',  'F8',   # 52-58
    'FT9', 'FT10','TP9', 'TP10','Fpz',                  # 59-63
]
assert len(CH_NAMES) == 64

EMG_NAMES = ['EMG1', 'EMG2', 'EMG3', 'EMG4']

MOTOR_CH = {ch: i for i, ch in enumerate(CH_NAMES)
             if ch in {'C3','C1','Cz','C2','C4','FC3','FC1','FCz','FC2','FC4',
                       'CP3','CP1','CPz','CP2','CP4'}}


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


class _LogitWrapper(nn.Module):
    """Strip attention weights; output logits only (required for SHAP)."""
    def __init__(self, model: HybridBCIModel):
        super().__init__()
        self.model = model

    def forward(self, eeg, emg):
        logits, _ = self.model(eeg, emg)
        return logits


# ═══════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════

def load_subject(sid: int, data_dir: str, ckpt_dir: str, cfg: dict, device: torch.device):
    h5_path   = os.path.join(data_dir, f"sub-{sid:02d}_member_A.h5")
    ckpt_path = os.path.join(ckpt_dir, f"best_s{sid:02d}.pt")
    for p in (h5_path, ckpt_path):
        if not os.path.exists(p):
            print(f"  [s{sid:02d}] 파일 없음: {os.path.basename(p)} — 건너뜀")
            return None

    ds = cfg["emg_ds_factor"]
    with h5py.File(h5_path, "r") as f:
        eeg = f["eeg/epochs"][:].astype(np.float32)        # (N, 64, 2304)
        emg = f["emg/epochs"][:, :, ::ds].astype(np.float32)  # (N, 4, 288)
        lbl = f["labels"][:].astype(np.int64) - 1             # 0/1

    model = HybridBCIModel(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    wrapper = _LogitWrapper(model).to(device)
    wrapper.eval()

    return eeg, emg, lbl, wrapper


# ═══════════════════════════════════════════════════════════
#  SHAP computation for one subject
# ═══════════════════════════════════════════════════════════

def compute_shap_subject(sid: int, data_dir: str, ckpt_dir: str,
                          cfg: dict, device: torch.device,
                          n_bg: int, n_test: int) -> dict | None:
    import shap

    result = load_subject(sid, data_dir, ckpt_dir, cfg, device)
    if result is None:
        return None
    eeg, emg, lbl, wrapper = result
    n = len(lbl)

    # Stratified background: n_bg/2 per class
    rng = np.random.RandomState(42 + sid)
    idx0 = np.where(lbl == 0)[0]
    idx1 = np.where(lbl == 1)[0]
    n_each = n_bg // 2
    bg_idx = np.concatenate([
        rng.choice(idx0, min(n_each, len(idx0)), replace=False),
        rng.choice(idx1, min(n_each, len(idx1)), replace=False),
    ])
    test_idx = rng.choice(n, min(n_test, n), replace=False)
    test_idx = np.setdiff1d(test_idx, bg_idx)[:n_test]

    bg_eeg  = torch.tensor(eeg[bg_idx],  dtype=torch.float32, device=device)
    bg_emg  = torch.tensor(emg[bg_idx],  dtype=torch.float32, device=device)
    te_eeg  = torch.tensor(eeg[test_idx], dtype=torch.float32, device=device)
    te_emg  = torch.tensor(emg[test_idx], dtype=torch.float32, device=device)
    te_lbl  = lbl[test_idx]

    try:
        explainer = shap.DeepExplainer(wrapper, [bg_eeg, bg_emg])
        sv = explainer.shap_values([te_eeg, te_emg])
    except Exception as e:
        print(f"  [s{sid:02d}] SHAP 오류: {e}")
        return None

    # sv: list(n_classes) of list(n_inputs)
    # sv[c][0]: (N_test, 64, 2304)  EEG SHAP for class c
    # sv[c][1]: (N_test, 4, 288)    EMG SHAP for class c
    if isinstance(sv, list) and isinstance(sv[0], list):
        sv_eeg = [np.array(sv[c][0]) for c in range(2)]  # 2 × (N, 64, 2304)
        sv_emg = [np.array(sv[c][1]) for c in range(2)]  # 2 × (N, 4, 288)
    else:
        # Older SHAP may return differently; handle gracefully
        print(f"  [s{sid:02d}] SHAP 출력 형식 예상과 다름 — 건너뜀")
        return None

    # Aggregate over time → per-channel mean |SHAP|  (n_classes, n_ch)
    eeg_importance = np.stack([
        np.abs(sv_eeg[c]).mean(axis=(0, 2)) for c in range(2)
    ])  # (2, 64)
    emg_importance = np.stack([
        np.abs(sv_emg[c]).mean(axis=(0, 2)) for c in range(2)
    ])  # (2, 4)

    # Per-class accuracy sanity check
    with torch.no_grad():
        logits = wrapper(te_eeg, te_emg).cpu().numpy()
    preds = logits.argmax(1)
    acc   = float((preds == te_lbl).mean())

    print(f"  [s{sid:02d}] SHAP 완료  acc={acc:.3f}  "
          f"top-EEG: {CH_NAMES[eeg_importance.mean(0).argmax()]}  "
          f"n_test={len(test_idx)}")

    return {
        "sid":             sid,
        "eeg_importance":  eeg_importance,   # (2, 64)
        "emg_importance":  emg_importance,   # (2, 4)
        "test_accuracy":   acc,
        "n_test":          len(test_idx),
    }


# ═══════════════════════════════════════════════════════════
#  Aggregate across 52 subjects
# ═══════════════════════════════════════════════════════════

def aggregate(records: list[dict]) -> dict:
    eeg_stack = np.stack([r["eeg_importance"] for r in records])  # (S, 2, 64)
    emg_stack = np.stack([r["emg_importance"] for r in records])  # (S, 2, 4)
    return {
        "eeg_mean":  eeg_stack.mean(0),   # (2, 64)
        "eeg_std":   eeg_stack.std(0),
        "emg_mean":  emg_stack.mean(0),
        "emg_std":   emg_stack.std(0),
        "n_subjects": len(records),
    }


# ═══════════════════════════════════════════════════════════
#  Figures
# ═══════════════════════════════════════════════════════════

COLORS = {'left': '#2196F3', 'right': '#F44336', 'motor': '#4CAF50'}
CLASS_NAMES = ['Left MI', 'Right MI']


def _top_k(values: np.ndarray, k: int = 20) -> tuple[np.ndarray, list[str]]:
    idx = np.argsort(values)[::-1][:k]
    return idx, [CH_NAMES[i] for i in idx]


def fig_eeg_bar(agg: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for c, (ax, cls) in enumerate(zip(axes, CLASS_NAMES)):
        mean  = agg["eeg_mean"][c]
        std   = agg["eeg_std"][c]
        idx, names = _top_k(mean, k=20)
        color = [COLORS['motor'] if CH_NAMES[i] in MOTOR_CH else '#9E9E9E' for i in idx]
        ax.barh(range(20), mean[idx][::-1], xerr=std[idx][::-1],
                color=color[::-1], alpha=0.85, ecolor='black', capsize=3)
        ax.set_yticks(range(20))
        ax.set_yticklabels(names[::-1], fontsize=9)
        ax.set_xlabel('Mean |SHAP| value', fontsize=10)
        ax.set_title(f'{cls} — Top-20 EEG Channels\n(green = motor cortex)',
                     fontsize=11, fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.5)
    fig.suptitle(f'EEG Feature Importance — DeepSHAP (n={agg["n_subjects"]} subjects)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'fig_eeg_shap_topclass.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'fig_eeg_shap_topclass.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  저장: {path}")


def fig_emg_bar(agg: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(4)
    w = 0.35
    for c, (cls, color) in enumerate(zip(CLASS_NAMES, [COLORS['left'], COLORS['right']])):
        ax.bar(x + c * w, agg['emg_mean'][c], w,
               yerr=agg['emg_std'][c], label=cls,
               color=color, alpha=0.8, ecolor='black', capsize=4)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(EMG_NAMES)
    ax.set_ylabel('Mean |SHAP| value')
    ax.set_title(f'sEMG Feature Importance — DeepSHAP\n(n={agg["n_subjects"]} subjects)',
                 fontweight='bold')
    ax.legend()
    plt.tight_layout()
    path = out_dir / 'fig_emg_shap.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'fig_emg_shap.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  저장: {path}")


def fig_laterality(agg: dict, out_dir: Path) -> None:
    """Contralateral laterality: C3 vs C4 for each class."""
    c3_idx, c4_idx = MOTOR_CH['C3'], MOTOR_CH['C4']
    fig, ax = plt.subplots(figsize=(6, 4))

    for c, (cls, color) in enumerate(zip(CLASS_NAMES, [COLORS['left'], COLORS['right']])):
        vals = [agg['eeg_mean'][c][c3_idx], agg['eeg_mean'][c][c4_idx]]
        errs = [agg['eeg_std'][c][c3_idx],  agg['eeg_std'][c][c4_idx]]
        ax.bar([c * 3, c * 3 + 1], vals, yerr=errs,
               color=[color, color], alpha=[0.9, 0.55], ecolor='black', capsize=4,
               label=f'{cls} (C3, C4)')

    ax.set_xticks([0, 1, 3, 4])
    ax.set_xticklabels(['C3\n(Left)', 'C4\n(Right)', 'C3\n(Left)', 'C4\n(Right)'])
    ax.set_ylabel('Mean |SHAP|')
    ax.set_title('Hemispheric Laterality of EEG Importance\n'
                 'Left MI: C4↑ | Right MI: C3↑ (contralateral pattern)',
                 fontweight='bold')

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=COLORS['left'], label='Left MI'),
                        Patch(facecolor=COLORS['right'], label='Right MI')])
    # Expected contralateral annotation
    ax.annotate('Expected\ncontralateral', xy=(1, agg['eeg_mean'][0][c4_idx]),
                xytext=(1.5, agg['eeg_mean'][0][c4_idx] * 1.1),
                fontsize=8, color='grey',
                arrowprops=dict(arrowstyle='->', color='grey'))

    plt.tight_layout()
    path = out_dir / 'fig_class_laterality.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'fig_class_laterality.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  저장: {path}")


def fig_summary_table(agg: dict, out_dir: Path) -> None:
    """Publication-ready top-10 features table figure."""
    rows = []
    for c, cls in enumerate(CLASS_NAMES):
        mean = agg['eeg_mean'][c]
        std  = agg['eeg_std'][c]
        idx, _ = _top_k(mean, k=10)
        for rank, i in enumerate(idx, 1):
            rows.append({
                'Class':   cls,
                'Rank':    rank,
                'Channel': CH_NAMES[i],
                'Region':  'Motor' if CH_NAMES[i] in MOTOR_CH else 'Non-motor',
                'Mean |SHAP|':  f'{mean[i]:.4f}',
                'Std':          f'±{std[i]:.4f}',
            })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.5)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white', fontweight='bold')
        elif df.iloc[row - 1]['Region'] == 'Motor' if row > 0 else False:
            cell.set_facecolor('#E8F5E9')

    ax.set_title(f'Top-10 EEG Features by DeepSHAP Importance\n'
                 f'n={agg["n_subjects"]} subjects (LOSO validation)',
                 fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    path = out_dir / 'fig_summary_table.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'fig_summary_table.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  저장: {path}")
    df.to_csv(out_dir / 'top10_features.csv', index=False)


# ═══════════════════════════════════════════════════════════
#  Per-subject CSV
# ═══════════════════════════════════════════════════════════

def build_subject_csv(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        sid = r['sid']
        for c, cls in enumerate(CLASS_NAMES):
            mean = r['eeg_importance'][c]
            top5_idx = np.argsort(mean)[::-1][:5]
            rows.append({
                'sid':        sid,
                'class':      cls,
                'top1_ch':    CH_NAMES[top5_idx[0]],
                'top2_ch':    CH_NAMES[top5_idx[1]],
                'top3_ch':    CH_NAMES[top5_idx[2]],
                'top4_ch':    CH_NAMES[top5_idx[3]],
                'top5_ch':    CH_NAMES[top5_idx[4]],
                'top_emg_ch': EMG_NAMES[r['emg_importance'][c].argmax()],
                'c3_shap':    round(float(mean[MOTOR_CH['C3']]), 5),
                'c4_shap':    round(float(mean[MOTOR_CH['C4']]), 5),
                'cz_shap':    round(float(mean[MOTOR_CH['Cz']]), 5),
                'accuracy':   round(r['test_accuracy'], 4),
                'n_test':     r['n_test'],
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root',    default='BCI_Research',
                   help='BCI_Research 루트 경로')
    p.add_argument('--n_bg',   type=int, default=100,
                   help='SHAP background 샘플 수 (기본 100)')
    p.add_argument('--n_test', type=int, default=200,
                   help='SHAP 테스트 샘플 수 (기본 전체)')
    p.add_argument('--sids',   type=str, default='all',
                   help='피험자 범위 (예: 1-52, 1,3,5, all)')
    p.add_argument('--device', default='auto',
                   choices=['auto', 'cuda', 'cpu'])
    return p.parse_args()


def parse_sids(spec: str) -> list[int]:
    if spec == 'all':
        return list(range(1, 53))
    if '-' in spec:
        a, b = spec.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(',')]


def main():
    try:
        import shap
        print(f"  shap 버전: {shap.__version__}")
    except ImportError:
        print("shap 미설치. 설치: pip install shap")
        sys.exit(1)

    args  = parse_args()
    root  = Path(args.root)
    data_dir = str(root / 'preprocessed' / 'member_A')
    ckpt_dir = str(root / 'results' / 'checkpoints_A')
    out_dir  = root / 'results' / 'xai_shap'
    fig_dir  = out_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"  DeepSHAP  |  device={device}  |  bg={args.n_bg}  test={args.n_test}")
    print(f"{'='*60}\n")

    cfg  = DEFAULT_CFG.copy()
    sids = parse_sids(args.sids)

    records = []
    for sid in sids:
        r = compute_shap_subject(sid, data_dir, ckpt_dir, cfg, device,
                                  n_bg=args.n_bg, n_test=args.n_test)
        if r is not None:
            records.append(r)

    if not records:
        print("처리된 피험자 없음 — 경로를 확인하세요.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  완료: {len(records)}/{len(sids)} 피험자")

    # Aggregate
    agg = aggregate(records)
    np.savez_compressed(
        out_dir / 'shap_channel_importance.npz',
        eeg_mean=agg['eeg_mean'],  eeg_std=agg['eeg_std'],
        emg_mean=agg['emg_mean'],  emg_std=agg['emg_std'],
        ch_names=np.array(CH_NAMES), emg_names=np.array(EMG_NAMES),
    )
    print(f"  저장: {out_dir / 'shap_channel_importance.npz'}")

    # Per-subject CSV
    subj_df = build_subject_csv(records)
    subj_df.to_csv(out_dir / 'shap_per_subject.csv', index=False)
    print(f"  저장: {out_dir / 'shap_per_subject.csv'}")

    # Summary stats
    for c, cls in enumerate(CLASS_NAMES):
        top_idx = agg['eeg_mean'][c].argmax()
        print(f"\n  [{cls}] Top EEG: {CH_NAMES[top_idx]} "
              f"({agg['eeg_mean'][c][top_idx]:.4f}±{agg['eeg_std'][c][top_idx]:.4f})")
        print(f"    C3={agg['eeg_mean'][c][MOTOR_CH['C3']]:.4f}  "
              f"Cz={agg['eeg_mean'][c][MOTOR_CH['Cz']]:.4f}  "
              f"C4={agg['eeg_mean'][c][MOTOR_CH['C4']]:.4f}")

    # Figures
    print(f"\n  피규어 생성 중...")
    fig_eeg_bar(agg, fig_dir)
    fig_emg_bar(agg, fig_dir)
    fig_laterality(agg, fig_dir)
    fig_summary_table(agg, fig_dir)

    print(f"\n{'='*60}")
    print(f"  완료. 출력: {out_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
