"""
attention_analysis.py — Softmax Attention Weight vs EMG SNR 연계 분석
=====================================================================
각 피험자의 best_s*.pt 체크포인트에서 SoftmaxAttentionFusion 가중치를
추출하고 EMG SNR / EMG Activation 과의 Spearman 상관관계를 분석합니다.

Colab 실행 (PyTorch 필요):
  !python /content/attention_analysis.py \\
      --drive_root /content/drive/MyDrive/BCI_Research

출력:
  results/attention/
  ├── attention_weights_per_subject.csv   ← 피험자별 w_eeg / w_emg 평균
  └── attention_snr_correlation.json      ← Spearman ρ, p-value
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr

SEED = 42


# ════════════════════════════════════════════════════════════════
#  CONFIG (inference.py 와 동일)
# ════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "n_eeg_ch":        64,
    "n_emg_ch":        4,
    "n_times":         2304,
    "n_classes":       2,
    "emg_ds_factor":   8,
    "eegnet_F1":       8,
    "eegnet_D":        2,
    "eegnet_kern_len": 256,
    "eegnet_dropout":  0.5,
    "lstm_hidden":     128,
    "lstm_layers":     2,
    "lstm_dropout":    0.3,
    "clf_dropout":     0.3,
    "feat_dim":        256,
}
DEFAULT_CONFIG["n_times_emg"] = (DEFAULT_CONFIG["n_times"]
                                  // DEFAULT_CONFIG["emg_ds_factor"])


# ════════════════════════════════════════════════════════════════
#  모델 (inference.py 와 동일 구조 — 체크포인트 호환 필수)
# ════════════════════════════════════════════════════════════════

class EEGNetEncoder(nn.Module):
    def __init__(self, n_ch, n_times, F1=8, D=2,
                 kern_len=256, dropout=0.5, feat_dim=256):
        super().__init__()
        F2 = F1 * D
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len),
                      padding=(0, kern_len // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F2, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        flat = self._flat(n_ch, n_times)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(flat, feat_dim), nn.ELU())

    def _flat(self, n_ch, n_times):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_ch, n_times)
            return self.block3(self.block2(self.block1(x))).numel()

    def forward(self, x):
        return self.fc(self.block3(self.block2(self.block1(x.unsqueeze(1)))))


class EMGBiLSTMEncoder(nn.Module):
    def __init__(self, n_ch=4, hidden=128, n_layers=2,
                 dropout=0.3, feat_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(
            n_ch, hidden, n_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
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
        w = F.softmax(
            self.attn(torch.cat([h_eeg, h_emg], dim=-1)), dim=-1
        )
        return (w[:, 0:1] * self.W_eeg(h_eeg)
                + w[:, 1:2] * self.W_emg(h_emg)), w


class HybridBCIModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        fd = cfg["feat_dim"]
        self.eeg_enc = EEGNetEncoder(
            cfg["n_eeg_ch"], cfg["n_times"],
            cfg["eegnet_F1"], cfg["eegnet_D"],
            cfg["eegnet_kern_len"], cfg["eegnet_dropout"], fd,
        )
        self.emg_enc = EMGBiLSTMEncoder(
            cfg["n_emg_ch"], cfg["lstm_hidden"],
            cfg["lstm_layers"], cfg["lstm_dropout"], fd,
        )
        self.fusion = SoftmaxAttentionFusion(fd)
        self.clf = nn.Sequential(
            nn.Linear(fd, 128), nn.ELU(),
            nn.Dropout(cfg["clf_dropout"]),
            nn.Linear(128, cfg["n_classes"]),
        )

    def forward(self, eeg, emg):
        h_eeg = self.eeg_enc(eeg)
        h_emg = self.emg_enc(emg)
        fused, w = self.fusion(h_eeg, h_emg)
        return self.clf(fused), w


# ════════════════════════════════════════════════════════════════
#  피험자 단위 가중치 추출
# ════════════════════════════════════════════════════════════════

def extract_weights(sid: int, ckpt_dir: str, data_dir: str,
                    cfg: dict, device: torch.device):
    ckpt_path = os.path.join(ckpt_dir, f"best_s{sid:02d}.pt")
    data_path = os.path.join(data_dir, f"sub-{sid:02d}_member_A.h5")

    for p in (ckpt_path, data_path):
        if not os.path.exists(p):
            print(f"  [s{sid:02d}] 없음: {os.path.basename(p)} — 건너뜀")
            return None

    model = HybridBCIModel(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    ds = cfg["emg_ds_factor"]
    with h5py.File(data_path, "r") as f:
        eeg = f["eeg/epochs"][:].astype(np.float32)
        lbl = (f["labels"][:].astype(np.int64) - 1)
        if "emg" in f and "epochs" in f["emg"]:
            emg = f["emg/epochs"][:, :, ::ds].astype(np.float32)
        else:
            emg = np.zeros(
                (eeg.shape[0], cfg["n_emg_ch"],
                 cfg["n_times"] // ds), dtype=np.float32,
            )

    n       = min(eeg.shape[0], emg.shape[0])
    dataset = TensorDataset(
        torch.from_numpy(eeg[:n]),
        torch.from_numpy(emg[:n]),
        torch.from_numpy(lbl[:n]),
    )
    loader  = DataLoader(dataset, batch_size=64, shuffle=False)

    all_w, all_pred, all_true = [], [], []

    with torch.no_grad():
        for b_eeg, b_emg, b_lbl in loader:
            logits, w = model(b_eeg.to(device), b_emg.to(device))
            all_w.append(w.cpu().numpy())
            all_pred.extend(logits.argmax(1).cpu().tolist())
            all_true.extend(b_lbl.tolist())

    w_arr = np.concatenate(all_w, axis=0)   # (N, 2)
    acc   = float(np.mean(np.array(all_pred) == np.array(all_true)))

    # 클래스별 w_emg 평균
    preds = np.array(all_pred)
    w_emg_left  = float(w_arr[preds == 0, 1].mean()) if (preds == 0).any() else float("nan")
    w_emg_right = float(w_arr[preds == 1, 1].mean()) if (preds == 1).any() else float("nan")

    result = {
        "sid":           sid,
        "w_eeg_mean":    round(float(w_arr[:, 0].mean()), 5),
        "w_emg_mean":    round(float(w_arr[:, 1].mean()), 5),
        "w_eeg_std":     round(float(w_arr[:, 0].std()),  5),
        "w_emg_std":     round(float(w_arr[:, 1].std()),  5),
        "w_emg_left":    round(w_emg_left,  5),
        "w_emg_right":   round(w_emg_right, 5),
        "accuracy":      round(acc, 5),
        "n_trials":      n,
    }
    print(f"  [s{sid:02d}] w_eeg={result['w_eeg_mean']:.4f}  "
          f"w_emg={result['w_emg_mean']:.4f}  acc={acc:.4f}")
    return result


# ════════════════════════════════════════════════════════════════
#  상관 분석
# ════════════════════════════════════════════════════════════════

def _spearman(df: pd.DataFrame, x_col: str, y_col: str, label: str):
    sub  = df[[x_col, y_col]].dropna()
    rho, p = spearmanr(sub[x_col], sub[y_col])
    sig  = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
    print(f"  {label:<45}: ρ={rho:+.4f}  p={p:.4f}  {sig}  (n={len(sub)})")
    return {
        "label":  label,
        "rho":    round(float(rho), 4),
        "p":      round(float(p),   6),
        "sig":    sig,
        "n":      int(len(sub)),
    }


# ════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════

def run(drive_root: str):
    data_dir    = os.path.join(drive_root, "preprocessed",  "member_A")
    ckpt_dir    = os.path.join(drive_root, "results", "checkpoints_A")
    out_dir     = os.path.join(drive_root, "results", "attention")
    summary_csv = os.path.join(data_dir, "summary_member_A_v4.csv")

    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"summary CSV 없음: {summary_csv}")

    summary_df = pd.read_csv(summary_csv)[["sid", "emg_snr_db", "emg_activation"]]
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = DEFAULT_CONFIG.copy()

    print(f"\n{'='*60}")
    print(f"  Attention Weight Extraction  |  device={device}")
    print(f"{'='*60}\n")

    records = []
    for sid in range(1, 53):
        r = extract_weights(sid, ckpt_dir, data_dir, cfg, device)
        if r is not None:
            records.append(r)

    if not records:
        print("  추출된 데이터 없음 — 체크포인트 경로를 확인하세요.")
        return

    attn_df = pd.DataFrame(records)
    merged  = attn_df.merge(summary_df, on="sid", how="left")

    csv_path = os.path.join(out_dir, "attention_weights_per_subject.csv")
    merged.to_csv(csv_path, index=False)
    print(f"\n  저장: {csv_path}")

    # ── 피험자 평균 요약
    print(f"\n  w_eeg 전체 평균: "
          f"{merged['w_eeg_mean'].mean():.4f} ± {merged['w_eeg_mean'].std():.4f}")
    print(f"  w_emg 전체 평균: "
          f"{merged['w_emg_mean'].mean():.4f} ± {merged['w_emg_mean'].std():.4f}")

    # ── Spearman 상관 ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Spearman Correlation")
    print(f"{'='*60}")

    corr_pairs = [
        ("w_emg_mean", "emg_snr_db",     "w_emg vs EMG SNR (dB)"),
        ("w_emg_mean", "emg_activation",  "w_emg vs EMG Activation"),
        ("w_emg_mean", "accuracy",        "w_emg vs Classification Accuracy"),
        ("w_eeg_mean", "accuracy",        "w_eeg vs Classification Accuracy"),
        ("w_eeg_mean", "emg_snr_db",      "w_eeg vs EMG SNR (dB)"),
    ]

    corr_out = {}
    for x_col, y_col, label in corr_pairs:
        if x_col not in merged.columns or y_col not in merged.columns:
            continue
        key = f"{x_col}_vs_{y_col}"
        corr_out[key] = _spearman(merged, x_col, y_col, label)

    corr_path = os.path.join(out_dir, "attention_snr_correlation.json")
    with open(corr_path, "w") as f:
        json.dump(corr_out, f, indent=2)
    print(f"\n  저장: {corr_path}")

    return corr_out


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Attention Weight vs EMG SNR Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--drive_root", type=str,
        default=str(Path(__file__).resolve().parents[1] / "BCI_Research"),
        help="BCI_Research 루트 경로",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.drive_root)
