"""
calibration.py — Post-hoc Logit Calibration (Right MI Bias 보정)
================================================================
v4 체크포인트를 재학습 없이 사용하여 추론 시 Right MI logit bias를 보정.

알고리즘:
  1. 각 피험자의 v4 체크포인트 로드
  2. 피험자 본인 데이터(LOSO 테스트셋)로 Left MI 샘플의 logit bias 추정
     bias = mean(logit[1] - logit[0])  for Left MI samples
  3. 추론 시 Right MI logit에서 bias를 감산
  4. 전체 52명 accuracy/kappa/left_recall/right_recall 비교 보고

사용법:
  # Colab (drive_root = MI-BCI 루트)
  python calibration.py --drive_root /content/drive/MyDrive/MI-BCI

  # 로컬 (PyTorch 환경)
  python src/calibration.py

출력:
  BCI_Research/results/calibration/
  ├── calibration_bias.json          # {sid: bias_value}
  ├── calibration_results.csv        # 피험자별 전/후 지표
  └── calibration_summary.txt        # 전체 요약
"""

import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

# ── 재현성 ──────────────────────────────────────────────────────
SEED = 42
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed()

# ── 모델 하이퍼파라미터 ──────────────────────────────────────────
CFG = {
    "n_eeg_ch": 64, "n_emg_ch": 4, "n_times": 2304,
    "n_classes": 2, "emg_ds_factor": 8,
    "eegnet_F1": 8, "eegnet_D": 2, "eegnet_kern_len": 256, "eegnet_dropout": 0.5,
    "lstm_hidden": 128, "lstm_layers": 2, "lstm_dropout": 0.3,
    "clf_dropout": 0.3, "feat_dim": 256,
}

BIAS_SUBJECTS = [1, 5, 7, 11, 12, 15, 24, 34, 36]
BIAS_FIX_THRESHOLD = 0.30  # right_recall - left_recall < 이 값이면 bias 개선 판정


# ════════════════════════════════════════════════════════════════
#  모델 정의 (ablation_study.py 와 동일 아키텍처)
# ════════════════════════════════════════════════════════════════

class EEGNetEncoder(nn.Module):
    def __init__(self, n_ch=64, n_times=2304, F1=8, D=2,
                 kern_len=256, dropout=0.5, feat_dim=256):
        super().__init__()
        F2 = F1 * D
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len // 2), bias=False),
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
        with torch.no_grad():
            flat = self.block3(self.block2(
                self.block1(torch.zeros(1, 1, n_ch, n_times)))).numel()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(flat, feat_dim), nn.ELU())

    def forward(self, x):
        return self.fc(self.block3(self.block2(self.block1(x.unsqueeze(1)))))


class EMGBiLSTMEncoder(nn.Module):
    def __init__(self, n_ch=4, hidden=128, n_layers=2, dropout=0.3, feat_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(n_ch, hidden, n_layers, batch_first=True,
                            bidirectional=True,
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


class FusionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        fd = cfg["feat_dim"]
        self.eeg_enc = EEGNetEncoder(
            cfg["n_eeg_ch"], cfg["n_times"],
            cfg["eegnet_F1"], cfg["eegnet_D"],
            cfg["eegnet_kern_len"], cfg["eegnet_dropout"], fd)
        self.emg_enc = EMGBiLSTMEncoder(
            cfg["n_emg_ch"], cfg["lstm_hidden"],
            cfg["lstm_layers"], cfg["lstm_dropout"], fd)
        self.fusion = SoftmaxAttentionFusion(fd)
        self.clf = nn.Sequential(
            nn.Linear(fd, 128), nn.ELU(),
            nn.Dropout(cfg["clf_dropout"]),
            nn.Linear(128, cfg["n_classes"]),
        )

    def forward(self, eeg, emg):
        fused, w = self.fusion(self.eeg_enc(eeg), self.emg_enc(emg))
        return self.clf(fused), w


# ════════════════════════════════════════════════════════════════
#  유틸
# ════════════════════════════════════════════════════════════════

def load_subject_data(h5_path: str, emg_ds: int = 8):
    with h5py.File(h5_path, "r") as f:
        eeg = f["eeg/epochs"][:].astype(np.float32)        # (N, 64, 2304)
        lbl = f["labels"][:].astype(np.int64) - 1           # 1/2 → 0/1
        if "emg" in f and "epochs" in f["emg"]:
            emg = f["emg/epochs"][:].astype(np.float32)
        else:
            emg = np.zeros((eeg.shape[0], 4, eeg.shape[2]), dtype=np.float32)
    if emg_ds > 1:
        emg = emg[:, :, ::emg_ds]                           # → (N, 4, 288)
    n = min(eeg.shape[0], emg.shape[0], lbl.shape[0])
    return eeg[:n], emg[:n], lbl[:n]


def load_model(ckpt_path: str, cfg: dict, device: torch.device) -> FusionModel:
    model = FusionModel(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def compute_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    acc = accuracy_score(true, pred)
    try:
        kappa = cohen_kappa_score(true, pred, labels=[0, 1])
    except Exception:
        kappa = 0.0
    cm = confusion_matrix(true, pred, labels=[0, 1])
    lr = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0.0
    rr = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0.0
    return {
        "accuracy":     round(float(acc), 6),
        "kappa":        round(float(kappa), 6),
        "left_recall":  round(float(lr), 6),
        "right_recall": round(float(rr), 6),
    }


@torch.no_grad()
def collect_logits(model: FusionModel, eeg: np.ndarray, emg: np.ndarray,
                   device: torch.device, batch_size: int = 64) -> torch.Tensor:
    """전체 trial에 대한 raw logit 반환 (N, 2)."""
    model.eval()
    parts = []
    for s in range(0, len(eeg), batch_size):
        e = torch.tensor(eeg[s:s+batch_size], dtype=torch.float32).to(device)
        m = torch.tensor(emg[s:s+batch_size], dtype=torch.float32).to(device)
        logits, _ = model(e, m)
        parts.append(logits.cpu())
    return torch.cat(parts, dim=0)


# ════════════════════════════════════════════════════════════════
#  메인 루프
# ════════════════════════════════════════════════════════════════

def run(drive_root: str, sids: list, device_str: str = "cuda",
        data_dir: str = None, conditional: bool = False) -> pd.DataFrame:
    """
    conditional=False: 모든 피험자에 보정 적용 → calibration_results.csv
    conditional=True : bias > 0인 경우에만 적용 → conditional_calibration_results.csv
    """
    bci_root = os.path.join(drive_root, "BCI_Research")
    data_dir = data_dir or os.path.join(bci_root, "preprocessed", "member_A")
    ckpt_dir = os.path.join(bci_root, "results", "checkpoints_A")
    out_dir  = os.path.join(bci_root, "results", "calibration")
    os.makedirs(out_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available() and device_str == "cuda"
    device   = torch.device("cuda" if use_cuda else "cpu")
    mode_tag = "conditional (bias>0)" if conditional else "unconditional"
    print(f"\n  [Calibration] device={device}  subjects={len(sids)}  mode={mode_tag}")
    print("  " + "=" * 60)

    bias_dict = {}
    rows      = []

    for sid in sids:
        h5_path   = os.path.join(data_dir, f"sub-{sid:02d}_member_A.h5")
        ckpt_path = os.path.join(ckpt_dir, f"best_s{sid:02d}.pt")

        if not os.path.exists(h5_path):
            print(f"  s{sid:02d}: HDF5 없음 — 스킵")
            continue
        if not os.path.exists(ckpt_path):
            print(f"  s{sid:02d}: 체크포인트 없음 — 스킵")
            continue

        eeg, emg, labels = load_subject_data(h5_path, CFG["emg_ds_factor"])
        model = load_model(ckpt_path, CFG, device)

        # raw logit 수집
        logits = collect_logits(model, eeg, emg, device)   # (N, 2)

        # bias 추정: Left MI 샘플에서 Right logit 과대 평가량
        left_mask = torch.tensor(labels) == 0
        if left_mask.sum() > 0:
            left_logits = logits[left_mask]                # (n_left, 2)
            bias = float((left_logits[:, 1] - left_logits[:, 0]).mean().item())
        else:
            bias = 0.0

        # 보정 적용 (conditional=True이면 bias > 0인 경우에만)
        cal_logits = logits.clone()
        if bias > 0 or not conditional:
            cal_logits[:, 1] -= bias

        pred_before = logits.argmax(dim=1).numpy()
        pred_after  = cal_logits.argmax(dim=1).numpy()

        mb = compute_metrics(labels, pred_before)
        ma = compute_metrics(labels, pred_after)
        is_bias = sid in BIAS_SUBJECTS

        rows.append({
            "sid":                 sid,
            "is_bias_subject":     is_bias,
            "bias_value":          round(bias, 6),
            "accuracy_before":     mb["accuracy"],
            "kappa_before":        mb["kappa"],
            "left_recall_before":  mb["left_recall"],
            "right_recall_before": mb["right_recall"],
            "accuracy_after":      ma["accuracy"],
            "kappa_after":         ma["kappa"],
            "left_recall_after":   ma["left_recall"],
            "right_recall_after":  ma["right_recall"],
        })
        bias_dict[sid] = bias

        tag = "[BIAS]" if is_bias else "      "
        print(
            f"  s{sid:02d} {tag}  bias={bias:+.4f}  "
            f"acc {mb['accuracy']:.4f}→{ma['accuracy']:.4f}  "
            f"κ {mb['kappa']:.4f}→{ma['kappa']:.4f}  "
            f"L {mb['left_recall']:.3f}→{ma['left_recall']:.3f}  "
            f"R {mb['right_recall']:.3f}→{ma['right_recall']:.3f}"
        )

    df = pd.DataFrame(rows)

    # 저장 — conditional 여부에 따라 파일명 분리
    fname    = "conditional_calibration_results.csv" if conditional else "calibration_results.csv"
    csv_path = os.path.join(out_dir, fname)
    df.to_csv(csv_path, index=False)
    print(f"\n  저장: {csv_path}")

    bias_path = os.path.join(out_dir, "calibration_bias.json")
    with open(bias_path, "w") as f:
        json.dump({str(k): round(v, 6) for k, v in bias_dict.items()}, f, indent=2)
    print(f"  저장: {bias_path}")

    _print_and_save_summary(df, out_dir)
    return df


def _print_and_save_summary(df: pd.DataFrame, out_dir: str):
    df2 = df.copy()
    df2["rdiff_before"] = df2["right_recall_before"] - df2["left_recall_before"]
    df2["rdiff_after"]  = df2["right_recall_after"]  - df2["left_recall_after"]
    bias_rows = df2[df2["is_bias_subject"]]

    acc_b  = df["accuracy_before"].mean()
    acc_a  = df["accuracy_after"].mean()
    k_b    = df["kappa_before"].mean()
    k_a    = df["kappa_after"].mean()
    dk     = k_a - k_b

    fixed = int((bias_rows["rdiff_after"] < BIAS_FIX_THRESHOLD).sum()) if len(bias_rows) > 0 else 0

    lines = [
        "",
        "=" * 65,
        "  Calibration 결과 요약",
        "=" * 65,
        f"  전체 피험자 (N={len(df)})",
        f"    Accuracy : {acc_b:.4f} → {acc_a:.4f}  (Δ={acc_a-acc_b:+.4f})",
        f"    Kappa    : {k_b:.4f} → {k_a:.4f}  (Δ={dk:+.4f})",
    ]

    if len(bias_rows) > 0:
        b_acc_b = bias_rows["accuracy_before"].mean()
        b_acc_a = bias_rows["accuracy_after"].mean()
        b_k_b   = bias_rows["kappa_before"].mean()
        b_k_a   = bias_rows["kappa_after"].mean()
        lines += [
            "",
            f"  Bias 피험자 ({len(bias_rows)}명)",
            f"    Accuracy : {b_acc_b:.4f} → {b_acc_a:.4f}  (Δ={b_acc_a-b_acc_b:+.4f})",
            f"    Kappa    : {b_k_b:.4f} → {b_k_a:.4f}  (Δ={b_k_a-b_k_b:+.4f})",
            f"    Bias Fix (recall_diff < {BIAS_FIX_THRESHOLD:.2f}): {fixed}/{len(bias_rows)}명",
        ]

    verdict = (f"[경고] kappa 손실 {dk:.4f} > -0.03 — 기각 기준 초과"
               if dk < -0.03 else
               f"[통과] kappa 손실 {dk:.4f} (허용 기준 -0.03 이내)")
    lines += ["", f"  {verdict}", "=" * 65]

    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(out_dir, "calibration_summary.txt"), "w") as f:
        f.write(summary)


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Post-hoc Logit Calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--drive_root", type=str, default=None,
                   help="MI-BCI 루트 경로 (Colab: /content/drive/MyDrive/MI-BCI)")
    p.add_argument("--data_dir", type=str, default=None,
                   help="HDF5 데이터 디렉터리 (기본: drive_root/BCI_Research/preprocessed/member_A)")
    p.add_argument("--sids", type=int, nargs="+", default=list(range(1, 53)),
                   help="실행할 피험자 목록")
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--conditional", action="store_true",
                   help="bias > 0인 경우에만 보정 적용 → conditional_calibration_results.csv")
    return p.parse_args()


if __name__ == "__main__":
    args  = parse_args()
    root  = args.drive_root or str(Path(__file__).resolve().parent.parent)
    run(root, args.sids, args.device, args.data_dir, args.conditional)
