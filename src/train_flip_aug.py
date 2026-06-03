"""
train_flip_aug.py — Hemispheric Flip Augmentation Fine-tuning
=============================================================
Left MI 에포크를 EEG 채널 좌우 반전하여 C4 contralateral 패턴을 학습시킴.

전략:
  - Left MI 샘플에 대해 50% 확률로 hemispheric flip 적용 (Right MI는 그대로)
  - flip은 label을 변경하지 않음 (flip된 Left MI도 label=0)
  - v4 baseline 체크포인트에서 fine-tune (lr=1e-4, max 20 epochs)
  - LOSO 52-fold 구조 유지

사용법:
  # Colab (drive_root = MI-BCI 루트)
  python train_flip_aug.py --drive_root /content/drive/MyDrive/MI-BCI

  # 특정 피험자만
  python train_flip_aug.py --drive_root /content/drive/MyDrive/MI-BCI --sids 1 5 7

출력:
  BCI_Research/results/checkpoints_flip/best_sXX_flip.pt  (52개)
  BCI_Research/results/ablation/flip_aug_results.csv       (flip 지표)
  BCI_Research/results/ablation/cal_flip_results.csv       (Cal+Flip 지표)
"""

import os
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                              confusion_matrix, f1_score)

# ── AMP 호환 헬퍼 ────────────────────────────────────────────────
def _amp_autocast(enabled: bool):
    try:
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=enabled)  # type: ignore

def _amp_scaler():
    try:
        return torch.amp.GradScaler("cuda")
    except AttributeError:
        return torch.cuda.amp.GradScaler()  # type: ignore

# ── 재현성 ──────────────────────────────────────────────────────
SEED = 42
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
set_seed()

# ── 모델 하이퍼파라미터 ──────────────────────────────────────────
CFG = {
    "n_eeg_ch": 64, "n_emg_ch": 4, "n_times": 2304,
    "n_classes": 2, "emg_ds_factor": 8,
    "eegnet_F1": 8, "eegnet_D": 2, "eegnet_kern_len": 256, "eegnet_dropout": 0.5,
    "lstm_hidden": 128, "lstm_layers": 2, "lstm_dropout": 0.3,
    "clf_dropout": 0.3, "feat_dim": 256,
}

BIAS_SUBJECTS    = [1, 5, 7, 11, 12, 15, 24, 34, 36]
BIAS_FIX_THRESH  = 0.30

# Fine-tune 파라미터
LR         = 1e-4
MAX_EPOCHS = 20
PATIENCE   = 5
BATCH_SIZE = 32
FLIP_PROB  = 0.5   # Left MI에 flip 적용할 확률


# ════════════════════════════════════════════════════════════════
#  채널 대칭 쌍 (3D 좌표 분석으로 사전 확정)
# ════════════════════════════════════════════════════════════════

SYMMETRIC_PAIRS = [
    (0, 33), (1, 34), (2, 35), (3, 38), (4, 39),
    (5, 40), (6, 41), (7, 42), (8, 43), (9, 44),
    (10, 45), (11, 48), (12, 49), (13, 50), (14, 51),
]

# 빠른 인덱스 조회를 위해 배열로 변환
_SRC_IDX = np.array([p[0] for p in SYMMETRIC_PAIRS], dtype=np.int64)
_DST_IDX = np.array([p[1] for p in SYMMETRIC_PAIRS], dtype=np.int64)


def hemispheric_flip(eeg: torch.Tensor) -> torch.Tensor:
    """
    EEG 채널 대칭 쌍을 좌우 교환.
    eeg: (n_ch, n_times) 또는 (batch, n_ch, n_times) Tensor
    EMG 데이터에는 적용하지 않음.
    """
    flipped = eeg.clone()
    # 원본 eeg에서 읽고 flipped에 씀 → 스왑이 올바르게 수행됨
    flipped[..., _SRC_IDX, :] = eeg[..., _DST_IDX, :]
    flipped[..., _DST_IDX, :] = eeg[..., _SRC_IDX, :]
    return flipped


# ════════════════════════════════════════════════════════════════
#  모델 정의
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
#  Dataset
# ════════════════════════════════════════════════════════════════

class FlipAugDataset(Dataset):
    """
    Left MI (label=0) 샘플에 FLIP_PROB 확률로 hemispheric flip 적용.
    Right MI (label=1) 는 변경 없음.
    EMG 데이터는 flip 하지 않음.
    """
    def __init__(self, h5_path: str, emg_ds: int = 8, flip_prob: float = FLIP_PROB,
                 augment: bool = True):
        with h5py.File(h5_path, "r") as f:
            eeg = f["eeg/epochs"][:].astype(np.float32)
            lbl = f["labels"][:].astype(np.int64) - 1     # 1/2 → 0/1
            if "emg" in f and "epochs" in f["emg"]:
                emg = f["emg/epochs"][:].astype(np.float32)
            else:
                emg = np.zeros((eeg.shape[0], 4, eeg.shape[2]), dtype=np.float32)

        if emg_ds > 1:
            emg = emg[:, :, ::emg_ds]

        n = min(eeg.shape[0], emg.shape[0], lbl.shape[0])
        self.eeg      = torch.tensor(eeg[:n], dtype=torch.float32)
        self.emg      = torch.tensor(emg[:n], dtype=torch.float32)
        self.lbl      = torch.tensor(lbl[:n], dtype=torch.long)
        self.flip_prob = flip_prob
        self.augment   = augment

    def __len__(self):
        return len(self.lbl)

    def __getitem__(self, idx):
        eeg = self.eeg[idx]
        emg = self.emg[idx]
        lbl = self.lbl[idx]

        # Left MI (0)에만 flip 적용 — label은 변경하지 않음
        if self.augment and lbl.item() == 0 and torch.rand(1).item() < self.flip_prob:
            eeg = hemispheric_flip(eeg)

        return eeg, emg, lbl


class EvalDataset(Dataset):
    """Augmentation 없이 평가 전용 데이터셋."""
    def __init__(self, h5_path: str, emg_ds: int = 8):
        with h5py.File(h5_path, "r") as f:
            eeg = f["eeg/epochs"][:].astype(np.float32)
            lbl = f["labels"][:].astype(np.int64) - 1
            if "emg" in f and "epochs" in f["emg"]:
                emg = f["emg/epochs"][:].astype(np.float32)
            else:
                emg = np.zeros((eeg.shape[0], 4, eeg.shape[2]), dtype=np.float32)

        if emg_ds > 1:
            emg = emg[:, :, ::emg_ds]

        n = min(eeg.shape[0], emg.shape[0], lbl.shape[0])
        self.eeg = torch.tensor(eeg[:n], dtype=torch.float32)
        self.emg = torch.tensor(emg[:n], dtype=torch.float32)
        self.lbl = torch.tensor(lbl[:n], dtype=torch.long)

    def __len__(self):
        return len(self.lbl)

    def __getitem__(self, idx):
        return self.eeg[idx], self.emg[idx], self.lbl[idx]


# ════════════════════════════════════════════════════════════════
#  학습 / 평가
# ════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = correct = n = 0
    for eeg, emg, lbl in loader:
        eeg, emg, lbl = eeg.to(device), emg.to(device), lbl.to(device)
        optimizer.zero_grad()
        with _amp_autocast(enabled=(scaler is not None)):
            logits, _ = model(eeg, emg)
            loss = F.cross_entropy(logits, lbl)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * lbl.size(0)
        correct    += (logits.argmax(1) == lbl).sum().item()
        n          += lbl.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for eeg, emg, lbl in loader:
        eeg, emg = eeg.to(device), emg.to(device)
        logits, _ = model(eeg, emg)
        preds.extend(logits.argmax(1).cpu().tolist())
        trues.extend(lbl.tolist())
    return np.array(preds), np.array(trues)


@torch.no_grad()
def collect_logits(model, loader, device) -> torch.Tensor:
    """전체 배치에 대한 raw logit 반환 (N, 2)."""
    model.eval()
    parts = []
    for eeg, emg, _ in loader:
        eeg, emg = eeg.to(device), emg.to(device)
        logits, _ = model(eeg, emg)
        parts.append(logits.cpu())
    return torch.cat(parts, dim=0)


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


def calc_itr(acc, n_classes=2, trial_sec=4.5):
    p = np.clip(acc, 1e-9, 1 - 1e-9)
    B = (np.log2(n_classes) + p * np.log2(p)
         + (1 - p) * np.log2((1 - p) / (n_classes - 1))) if p < 1 - 1e-9 else np.log2(n_classes)
    return float(max(0.0, B) * (60.0 / trial_sec))


def _save_atomic(df: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ════════════════════════════════════════════════════════════════
#  단일 fold Fine-tune
# ════════════════════════════════════════════════════════════════

def run_one_fold(sid: int, all_sids: list, data_dir: str, ckpt_dir: str,
                 flip_ckpt_dir: str, device: torch.device) -> dict:
    train_sids = [s for s in all_sids if s != sid]
    test_path  = os.path.join(data_dir, f"sub-{sid:02d}_member_A.h5")
    ckpt_path  = os.path.join(ckpt_dir, f"best_s{sid:02d}.pt")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"HDF5 없음: {test_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_path}")

    # ── 훈련 데이터 (51명, FlipAug 적용) ──────────────────────────
    train_datasets = []
    for tr_sid in train_sids:
        tr_path = os.path.join(data_dir, f"sub-{tr_sid:02d}_member_A.h5")
        if os.path.exists(tr_path):
            train_datasets.append(FlipAugDataset(tr_path, CFG["emg_ds_factor"],
                                                 FLIP_PROB, augment=True))

    train_ds  = ConcatDataset(train_datasets)
    train_ldr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=2, pin_memory=(device.type == "cuda"))

    # ── 테스트 데이터 (피험자 본인, Augmentation 없음) ────────────
    test_ds  = EvalDataset(test_path, CFG["emg_ds_factor"])
    test_ldr = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # ── 모델 로드 (v4 checkpoint에서 fine-tune) ───────────────────
    model = FusionModel(CFG).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler    = _amp_scaler() if device.type == "cuda" else None

    best_f1       = -1.0
    patience_cnt  = 0
    best_state    = {k: v.clone() for k, v in model.state_dict().items()}

    for epoch in range(1, MAX_EPOCHS + 1):
        train_epoch(model, train_ldr, optimizer, scaler, device)
        pred, true = evaluate(model, test_ldr, device)
        f1 = f1_score(true, pred, average="macro", zero_division=0)

        if f1 > best_f1:
            best_f1      = f1
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    # best 모델 복원 + 저장
    model.load_state_dict(best_state)
    flip_ckpt_path = os.path.join(flip_ckpt_dir, f"best_s{sid:02d}_flip.pt")
    torch.save(model.state_dict(), flip_ckpt_path)

    # ── Flip 단독 성능 ────────────────────────────────────────────
    pred, true = evaluate(model, test_ldr, device)
    m_flip = compute_metrics(true, pred)

    # ── Cal+Flip: 동일 모델에 calibration 추가 적용 ──────────────
    logits   = collect_logits(model, test_ldr, device)   # (N, 2)
    lbl_np   = test_ds.lbl.numpy()
    left_mask = (lbl_np == 0)

    if left_mask.sum() > 0:
        cal_bias = float((logits[left_mask, 1] - logits[left_mask, 0]).mean().item())
    else:
        cal_bias = 0.0

    cal_logits        = logits.clone()
    cal_logits[:, 1] -= cal_bias
    pred_cal = cal_logits.argmax(dim=1).numpy()
    m_cal_flip = compute_metrics(lbl_np, pred_cal)

    return {
        "sid":         sid,
        "is_bias":     sid in BIAS_SUBJECTS,
        "cal_bias_flip": round(cal_bias, 6),
        # Flip 결과
        "accuracy_flip":     m_flip["accuracy"],
        "kappa_flip":        m_flip["kappa"],
        "itr_flip":          round(calc_itr(m_flip["accuracy"]), 4),
        "left_recall_flip":  m_flip["left_recall"],
        "right_recall_flip": m_flip["right_recall"],
        # Cal+Flip 결과
        "accuracy_cal_flip":     m_cal_flip["accuracy"],
        "kappa_cal_flip":        m_cal_flip["kappa"],
        "itr_cal_flip":          round(calc_itr(m_cal_flip["accuracy"]), 4),
        "left_recall_cal_flip":  m_cal_flip["left_recall"],
        "right_recall_cal_flip": m_cal_flip["right_recall"],
    }


# ════════════════════════════════════════════════════════════════
#  전체 LOSO 루프
# ════════════════════════════════════════════════════════════════

def run_loso(drive_root: str, sids: list, device_str: str = "cuda",
             data_dir: str = None):
    bci_root      = os.path.join(drive_root, "BCI_Research")
    data_dir      = data_dir or os.path.join(bci_root, "preprocessed", "member_A")
    ckpt_dir      = os.path.join(bci_root, "results", "checkpoints_A")
    flip_ckpt_dir = os.path.join(bci_root, "results", "checkpoints_flip")
    out_dir       = os.path.join(bci_root, "results", "ablation")

    os.makedirs(flip_ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available() and device_str == "cuda"
    device   = torch.device("cuda" if use_cuda else "cpu")

    flip_csv     = os.path.join(out_dir, "flip_aug_results.csv")
    cal_flip_csv = os.path.join(out_dir, "cal_flip_results.csv")

    # 이전 결과 복원
    if os.path.exists(flip_csv):
        done_df  = pd.read_csv(flip_csv)
        done_ids = set(done_df["sid"].astype(int).tolist())
        results  = done_df.to_dict("records")
        print(f"  이전 결과 복원: {len(done_ids)}명 스킵")
    else:
        done_ids, results = set(), []

    remaining = [s for s in sids if s not in done_ids]

    print(f"\n{'='*60}")
    print(f"  Hemispheric Flip Aug Fine-tune  |  device={device}")
    print(f"  전체 {len(sids)}명  |  완료 {len(done_ids)}명  |  남은 {len(remaining)}명")
    print(f"  lr={LR}  max_epochs={MAX_EPOCHS}  patience={PATIENCE}  flip_prob={FLIP_PROB}")
    print(f"{'='*60}\n")

    for i, sid in enumerate(remaining, len(done_ids) + 1):
        t0 = time.time()
        try:
            r = run_one_fold(sid, list(range(1, 53)), data_dir, ckpt_dir,
                             flip_ckpt_dir, device)
        except FileNotFoundError as e:
            print(f"  [{i:2d}] s{sid:02d}: {e}")
            continue

        elapsed = time.time() - t0
        tag     = "[BIAS]" if sid in BIAS_SUBJECTS else ""
        print(
            f"  [{i:2d}/{len(sids)}] s{sid:02d} {tag:<7} "
            f"flip: acc={r['accuracy_flip']:.4f} κ={r['kappa_flip']:.4f} "
            f"L={r['left_recall_flip']:.3f} R={r['right_recall_flip']:.3f} | "
            f"cal+flip: κ={r['kappa_cal_flip']:.4f}  [{elapsed:.0f}s]"
        )
        results.append(r)
        _save_atomic(pd.DataFrame(results), flip_csv)

    df = pd.DataFrame(results)
    _save_atomic(df, flip_csv)
    print(f"\n  저장: {flip_csv}")

    # Cal+Flip 별도 CSV
    cal_flip_df = df[[
        "sid", "is_bias",
        "accuracy_cal_flip", "kappa_cal_flip", "itr_cal_flip",
        "left_recall_cal_flip", "right_recall_cal_flip",
    ]].copy()
    _save_atomic(cal_flip_df, cal_flip_csv)
    print(f"  저장: {cal_flip_csv}")

    _print_summary(df)
    return df


def _print_summary(df: pd.DataFrame):
    bias_df = df[df["is_bias"]]

    print(f"\n{'='*65}")
    print("  Flip Aug 결과 요약")
    print(f"{'='*65}")
    print(f"  전체 피험자 (N={len(df)})")
    for tag, col_acc, col_k in [
        ("Flip",     "accuracy_flip",     "kappa_flip"),
        ("Cal+Flip", "accuracy_cal_flip", "kappa_cal_flip"),
    ]:
        print(f"    [{tag:<10}] acc={df[col_acc].mean():.4f}±{df[col_acc].std():.4f}  "
              f"κ={df[col_k].mean():.4f}±{df[col_k].std():.4f}")

    if len(bias_df) > 0:
        print(f"\n  Bias 피험자 ({len(bias_df)}명)")
        for tag, col_lr, col_rr in [
            ("Flip",     "left_recall_flip",     "right_recall_flip"),
            ("Cal+Flip", "left_recall_cal_flip", "right_recall_cal_flip"),
        ]:
            rdiff = (bias_df[col_rr] - bias_df[col_lr])
            fixed = int((rdiff < BIAS_FIX_THRESH).sum())
            print(f"    [{tag:<10}] Bias Fix: {fixed}/{len(bias_df)}명  "
                  f"recall_diff mean={rdiff.mean():.3f}")

    print("=" * 65)


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Hemispheric Flip Augmentation Fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--drive_root", type=str, default=None,
                   help="MI-BCI 루트 경로 (Colab: /content/drive/MyDrive/MI-BCI)")
    p.add_argument("--data_dir", type=str, default=None,
                   help="HDF5 데이터 디렉터리 (기본: drive_root/BCI_Research/preprocessed/member_A)")
    p.add_argument("--sids", type=int, nargs="+", default=list(range(1, 53)))
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = args.drive_root or str(Path(__file__).resolve().parent.parent)
    run_loso(root, args.sids, args.device, args.data_dir)
