"""
train_flip_full.py — Full LOSO Retraining with Hemispheric Flip Augmentation
=============================================================================
fine-tune 방식의 실패(v4 편향 강화) 원인 분석 후, 랜덤 초기화부터 전체 재학습.

v4와 동일한 하이퍼파라미터를 사용하되, train_epoch 내에서
Left MI 샘플의 50%에 hemispheric flip을 적용합니다.

사용법:
  # 전체 52명 (~6시간 A100)
  python train_flip_full.py --drive_root /content/drive/MyDrive/MI-BCI

  # 빠른 검증 (bias 피험자 3명만)
  python train_flip_full.py --drive_root /content/drive/MyDrive/MI-BCI --sids 1 7 36

출력:
  BCI_Research/results/checkpoints_flip_full/best_sXX_flip_full.pt  (52개)
  BCI_Research/results/ablation/flip_full_results.csv
"""

import os
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

# ── v4와 동일한 하이퍼파라미터 ──────────────────────────────────
CFG = {
    "n_eeg_ch": 64, "n_emg_ch": 4, "n_times": 2304,
    "n_classes": 2, "emg_ds_factor": 8,
    "eegnet_F1": 8, "eegnet_D": 2, "eegnet_kern_len": 256, "eegnet_dropout": 0.5,
    "lstm_hidden": 128, "lstm_layers": 2, "lstm_dropout": 0.3,
    "clf_dropout": 0.3, "feat_dim": 256,
}

LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 32
MAX_EPOCHS   = 200
PATIENCE     = 20
NUM_WORKERS  = 4
FLIP_PROB    = 0.5

BIAS_SUBJECTS   = [1, 5, 7, 11, 12, 15, 24, 34, 36]
FIX_THRESHOLD   = 0.30


# ════════════════════════════════════════════════════════════════
#  채널 대칭 쌍 (train_flip_aug.py와 동일)
# ════════════════════════════════════════════════════════════════

SYMMETRIC_PAIRS = [
    (0, 33), (1, 34), (2, 35), (3, 38), (4, 39),
    (5, 40), (6, 41), (7, 42), (8, 43), (9, 44),
    (10, 45), (11, 48), (12, 49), (13, 50), (14, 51),
]
_SRC_IDX = np.array([p[0] for p in SYMMETRIC_PAIRS], dtype=np.int64)
_DST_IDX = np.array([p[1] for p in SYMMETRIC_PAIRS], dtype=np.int64)


def hemispheric_flip(eeg: torch.Tensor) -> torch.Tensor:
    """(n_ch, n_times) 또는 (batch, n_ch, n_times) Tensor 채널 좌우 교환."""
    flipped = eeg.clone()
    flipped[..., _SRC_IDX, :] = eeg[..., _DST_IDX, :]
    flipped[..., _DST_IDX, :] = eeg[..., _SRC_IDX, :]
    return flipped


# ════════════════════════════════════════════════════════════════
#  모델 정의 (S3 HybridBCIModel과 동일)
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

class BCIDataset(Dataset):
    """S3 BCIEpochDataset와 동일 — augmentation 없음."""
    def __init__(self, h5_path: str, emg_ds: int = 8):
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

def train_epoch_flip(model, loader, optimizer, criterion, device, scaler):
    """train_epoch + Left MI 샘플에 FLIP_PROB 확률로 hemispheric flip 적용."""
    model.train()
    total_loss = correct = n = 0
    use_amp = scaler is not None

    for eeg, emg, lbl in loader:
        # ── flip augmentation (CPU에서 수행 후 GPU로 이동) ───────
        left_mask = (lbl == 0)
        if left_mask.any():
            left_idx = left_mask.nonzero(as_tuple=True)[0]
            flip_sel = torch.rand(len(left_idx)) < FLIP_PROB
            if flip_sel.any():
                flip_idx       = left_idx[flip_sel]
                eeg[flip_idx]  = hemispheric_flip(eeg[flip_idx])

        eeg, emg, lbl = eeg.to(device), emg.to(device), lbl.to(device)
        optimizer.zero_grad()

        with _amp_autocast(enabled=use_amp):
            logits, _ = model(eeg, emg)
            loss = criterion(logits, lbl)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(lbl)
        correct    += (logits.argmax(1) == lbl).sum().item()
        n          += len(lbl)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, scaler):
    """평가 전용 — augmentation 없음."""
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    use_amp = scaler is not None

    for eeg, emg, lbl in loader:
        eeg, emg, lbl = eeg.to(device), emg.to(device), lbl.to(device)
        with _amp_autocast(enabled=use_amp):
            logits, _ = model(eeg, emg)
            total_loss += criterion(logits, lbl).item() * len(lbl)
        preds.extend(logits.argmax(1).cpu().tolist())
        trues.extend(lbl.cpu().tolist())

    n  = len(trues)
    f1 = f1_score(trues, preds, average="macro", zero_division=0)
    return total_loss / n, f1, np.array(preds), np.array(trues)


@torch.no_grad()
def collect_logits_loader(model, loader, device) -> tuple[torch.Tensor, np.ndarray]:
    """(logits (N,2), labels (N,)) 반환."""
    model.eval()
    all_logits, all_lbl = [], []
    for eeg, emg, lbl in loader:
        eeg, emg = eeg.to(device), emg.to(device)
        logits, _ = model(eeg, emg)
        all_logits.append(logits.cpu())
        all_lbl.extend(lbl.tolist())
    return torch.cat(all_logits, dim=0), np.array(all_lbl)


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


def calc_logit_bias(logits: torch.Tensor, labels: np.ndarray) -> float:
    """Left MI 샘플에서 Right logit 과대 평가량 (양수 = Right MI 편향)."""
    mask = (labels == 0)
    if mask.sum() == 0:
        return 0.0
    left_logits = logits[mask]
    return float((left_logits[:, 1] - left_logits[:, 0]).mean().item())


def _save_atomic(df: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ════════════════════════════════════════════════════════════════
#  단일 fold: v4 평가 + flip_full 학습 + 평가
# ════════════════════════════════════════════════════════════════

def run_one_fold(sid: int, all_sids: list, data_dir: str,
                 v4_ckpt_dir: str, flip_ckpt_dir: str,
                 device: torch.device) -> dict:
    train_sids = [s for s in all_sids if s != sid]
    test_path  = os.path.join(data_dir, f"sub-{sid:02d}_member_A.h5")
    v4_ckpt    = os.path.join(v4_ckpt_dir, f"best_s{sid:02d}.pt")
    flip_ckpt  = os.path.join(flip_ckpt_dir, f"best_s{sid:02d}_flip_full.pt")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"HDF5 없음: {test_path}")
    if not os.path.exists(v4_ckpt):
        raise FileNotFoundError(f"v4 체크포인트 없음: {v4_ckpt}")

    # ── 테스트 DataLoader ────────────────────────────────────────
    test_ds  = BCIDataset(test_path, CFG["emg_ds_factor"])
    test_ldr = DataLoader(test_ds, batch_size=64, shuffle=False,
                          num_workers=0, pin_memory=(device.type == "cuda"))

    use_amp = device.type == "cuda"
    scaler  = _amp_scaler() if use_amp else None
    crit    = nn.CrossEntropyLoss()

    # ── v4 모델 평가 (재학습 없음) ───────────────────────────────
    v4_model = FusionModel(CFG).to(device)
    state    = torch.load(v4_ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    v4_model.load_state_dict(state)

    v4_logits, v4_lbl = collect_logits_loader(v4_model, test_ldr, device)
    v4_pred   = v4_logits.argmax(dim=1).numpy()
    v4_m      = compute_metrics(v4_lbl, v4_pred)
    v4_bias   = calc_logit_bias(v4_logits, v4_lbl)
    del v4_model

    # ── flip_full 학습 (체크포인트 이미 있으면 스킵) ─────────────
    if os.path.exists(flip_ckpt):
        flip_model = FusionModel(CFG).to(device)
        state      = torch.load(flip_ckpt, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        flip_model.load_state_dict(state)
    else:
        # 훈련 데이터 로드 (51명)
        train_datasets = []
        for tr_sid in train_sids:
            tr_path = os.path.join(data_dir, f"sub-{tr_sid:02d}_member_A.h5")
            if os.path.exists(tr_path):
                train_datasets.append(BCIDataset(tr_path, CFG["emg_ds_factor"]))

        train_ds  = ConcatDataset(train_datasets)
        train_ldr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS,
                               pin_memory=(device.type == "cuda"),
                               drop_last=True)

        # 랜덤 초기화에서 시작 (v4 체크포인트 로드 없음)
        torch.manual_seed(SEED)
        flip_model = FusionModel(CFG).to(device)
        optimizer  = torch.optim.Adam(flip_model.parameters(),
                                      lr=LR, weight_decay=WEIGHT_DECAY)
        if scaler is not None:
            scaler = _amp_scaler()

        best_f1      = -1.0
        best_state   = {k: v.clone() for k, v in flip_model.state_dict().items()}
        patience_cnt = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            train_epoch_flip(flip_model, train_ldr, optimizer, crit, device, scaler)
            _, val_f1, _, _ = eval_epoch(flip_model, test_ldr, crit, device, scaler)

            if val_f1 > best_f1:
                best_f1      = val_f1
                best_state   = {k: v.clone() for k, v in flip_model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    break

        flip_model.load_state_dict(best_state)
        torch.save(flip_model.state_dict(), flip_ckpt)

    # ── flip_full 평가 ────────────────────────────────────────────
    flip_logits, flip_lbl = collect_logits_loader(flip_model, test_ldr, device)
    flip_pred = flip_logits.argmax(dim=1).numpy()
    flip_m    = compute_metrics(flip_lbl, flip_pred)
    flip_bias = calc_logit_bias(flip_logits, flip_lbl)
    del flip_model

    return {
        "sid":     sid,
        "is_bias": sid in BIAS_SUBJECTS,
        # v4 기준
        "acc_v4":          v4_m["accuracy"],
        "kappa_v4":        v4_m["kappa"],
        "left_recall_v4":  v4_m["left_recall"],
        "right_recall_v4": v4_m["right_recall"],
        "recall_diff_v4":  round(v4_m["right_recall"] - v4_m["left_recall"], 6),
        "bias_v4":         round(v4_bias, 6),
        # flip_full
        "acc_flip":          flip_m["accuracy"],
        "kappa_flip":        flip_m["kappa"],
        "left_recall_flip":  flip_m["left_recall"],
        "right_recall_flip": flip_m["right_recall"],
        "recall_diff_flip":  round(flip_m["right_recall"] - flip_m["left_recall"], 6),
        "bias_flip":         round(flip_bias, 6),
    }


# ════════════════════════════════════════════════════════════════
#  전체 LOSO 루프
# ════════════════════════════════════════════════════════════════

def run_loso(drive_root: str, sids: list, device_str: str = "cuda",
             data_dir: str = None):
    bci_root      = os.path.join(drive_root, "BCI_Research")
    data_dir      = data_dir or os.path.join(bci_root, "preprocessed", "member_A")
    v4_ckpt_dir   = os.path.join(bci_root, "results", "checkpoints_A")
    flip_ckpt_dir = os.path.join(bci_root, "results", "checkpoints_flip_full")
    out_dir       = os.path.join(bci_root, "results", "ablation")

    os.makedirs(flip_ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available() and device_str == "cuda"
    device   = torch.device("cuda" if use_cuda else "cpu")
    csv_path = os.path.join(out_dir, "flip_full_results.csv")

    # 이전 결과 복원
    if os.path.exists(csv_path):
        done_df  = pd.read_csv(csv_path)
        done_ids = set(done_df["sid"].astype(int).tolist())
        results  = done_df.to_dict("records")
        print(f"  이전 결과 복원: {len(done_ids)}명 스킵")
    else:
        done_ids, results = set(), []

    remaining = [s for s in sids if s not in done_ids]

    print(f"\n{'='*65}")
    print(f"  Flip Full Retraining  |  device={device}")
    print(f"  lr={LR}  epochs={MAX_EPOCHS}  patience={PATIENCE}  flip_prob={FLIP_PROB}")
    print(f"  전체 {len(sids)}명  |  완료 {len(done_ids)}명  |  남은 {len(remaining)}명")
    print(f"{'='*65}\n")

    for i, sid in enumerate(remaining, len(done_ids) + 1):
        t0 = time.time()
        try:
            r = run_one_fold(sid, list(range(1, 53)), data_dir,
                             v4_ckpt_dir, flip_ckpt_dir, device)
        except FileNotFoundError as e:
            print(f"  [{i:2d}] s{sid:02d}: {e}")
            continue

        elapsed = time.time() - t0
        tag     = "[BIAS]" if sid in BIAS_SUBJECTS else ""
        dk      = r["kappa_flip"] - r["kappa_v4"]
        print(
            f"  [{i:2d}/{len(sids)}] s{sid:02d} {tag:<7} "
            f"v4: κ={r['kappa_v4']:.4f} L={r['left_recall_v4']:.3f} R={r['right_recall_v4']:.3f} | "
            f"flip: κ={r['kappa_flip']:.4f} L={r['left_recall_flip']:.3f} R={r['right_recall_flip']:.3f} "
            f"Δκ={dk:+.4f}  [{elapsed:.0f}s]"
        )
        results.append(r)
        _save_atomic(pd.DataFrame(results), csv_path)

    df = pd.DataFrame(results)
    _save_atomic(df, csv_path)
    print(f"\n  저장: {csv_path}")
    _print_summary(df)
    return df


def _print_summary(df: pd.DataFrame):
    if df.empty:
        return
    bias_df = df[df["is_bias"]]

    print(f"\n{'='*65}")
    print("  Flip Full Retraining 결과 요약")
    print(f"{'='*65}")

    for tag, col_acc, col_k in [
        ("v4 baseline", "acc_v4",   "kappa_v4"),
        ("Flip Full  ", "acc_flip", "kappa_flip"),
    ]:
        print(f"  [{tag}] acc={df[col_acc].mean():.4f}±{df[col_acc].std():.4f}  "
              f"κ={df[col_k].mean():.4f}±{df[col_k].std():.4f}")

    dk = df["kappa_flip"].mean() - df["kappa_v4"].mean()
    verdict = f"[기각] Δκ={dk:+.4f}" if dk < -0.03 else f"[통과] Δκ={dk:+.4f}"
    print(f"  전체 kappa Δ: {verdict}")

    if len(bias_df) > 0:
        print(f"\n  Bias 피험자 ({len(bias_df)}명)")
        for tag, col_lr, col_rr in [
            ("v4 baseline", "left_recall_v4",   "right_recall_v4"),
            ("Flip Full  ", "left_recall_flip", "right_recall_flip"),
        ]:
            rdiff = (bias_df[col_rr] - bias_df[col_lr])
            fixed = int((rdiff < FIX_THRESHOLD).sum())
            print(f"  [{tag}] Bias Fix: {fixed}/{len(bias_df)}명  "
                  f"recall_diff mean={rdiff.mean():.3f}")

    print("=" * 65)


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Full LOSO Retraining with Hemispheric Flip Aug",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--drive_root", type=str, default=None,
                   help="MI-BCI 루트 경로 (Colab: /content/drive/MyDrive/MI-BCI)")
    p.add_argument("--data_dir", type=str, default=None,
                   help="HDF5 데이터 디렉터리 (기본: drive_root/BCI_Research/preprocessed/member_A)")
    p.add_argument("--sids", type=int, nargs="+", default=list(range(1, 53)),
                   help="실행할 피험자 목록")
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = args.drive_root or str(Path(__file__).resolve().parent.parent)
    run_loso(root, args.sids, args.device, args.data_dir)
