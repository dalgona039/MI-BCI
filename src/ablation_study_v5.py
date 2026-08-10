"""
ablation_study_v5.py — v5-clean: 움직임 오염 trial 배제 후 LOSO 재학습
=========================================================================
`ablation_study.py` (v4) 의 복사본. 모델 정의·하이퍼파라미터·LOSO 분할·
early stopping 은 v4와 **완전히 동일**하다. 유일한 차이는 입력 데이터에서
`bad_trial_idx_mi ∪ bad_trial_idx_voltage` 에 해당하는 에폭을 제거한 것.

v4 대비 변경점 (의도된 것만):
  1. BCIDataset(exclude_idx=...) — 로드 직후 오염 에폭 제거
  2. use_existing_ckpt 경로 제거 — fusion 도 처음부터 재학습
  3. 출력 경로 → results/ablation_v5_clean/
  4. --seed 인자 (D4: 42 / 1337 / 2024)
  5. per-trial 예측·attention 로그 저장 (T3)
  6. MPS(Apple Silicon) 디바이스 지원 — 수치 연산은 동일

⚠ v4에서 이어받은 주의사항 (의도적으로 유지):
  early stopping 과 best-epoch 선택이 **테스트 fold** 의 macro-F1 으로
  이루어진다 (v4 `ablation_study.py` 및 S3 노트북과 동일). 이는 절대
  정확도를 낙관적으로 편향시키지만, v4와 v5가 동일 프로토콜이어야
  두 결과의 차이를 데이터 배제 효과로 해석할 수 있으므로 그대로 둔다.

실행:
  python src/ablation_study_v5.py --model_type fusion --seed 42 \
      --data_dir BCI_Research/preprocessed/member_A \
      --exclude_json BCI_Research/results/ablation_v5_clean/excluded_epochs.json

출력:
  results/ablation_v5_clean/
  ├── ablation_{cond}_seed{S}_results.csv
  ├── per_trial/{cond}_seed{S}_s{NN}.csv
  └── progress.log
"""

import os
import sys
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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                              confusion_matrix, classification_report)

# ── AMP 호환 헬퍼 (PyTorch 2.5+ 신 API / 구 API 자동 선택) ───────
def _amp_autocast(enabled: bool):
    try:
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=enabled)  # type: ignore[attr-defined]

def _amp_scaler():
    try:
        return torch.amp.GradScaler("cuda")
    except AttributeError:
        return torch.cuda.amp.GradScaler()  # type: ignore[attr-defined]

# ── 재현성 ──────────────────────────────────────────────────────
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed()


def pick_device() -> torch.device:
    """CUDA > MPS > CPU. 연산 자체는 동일하며 속도만 다르다."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ── ITR ─────────────────────────────────────────────────────────
def calc_itr(accuracy, n_classes=2, trial_duration_sec=4.5):
    p = np.clip(accuracy, 1e-9, 1 - 1e-9)
    if p >= 1.0 - 1e-9:
        B = np.log2(n_classes)
    else:
        B = (np.log2(n_classes)
             + p * np.log2(p)
             + (1 - p) * np.log2((1 - p) / (n_classes - 1)))
    return float(max(0.0, B) * (60.0 / trial_duration_sec))


# ════════════════════════════════════════════════════════════════
#  Dataset
# ════════════════════════════════════════════════════════════════

class BCIDataset(Dataset):
    """v4와 동일. `exclude_idx` 로 오염 에폭을 로드 직후 제거하는 것만 추가."""

    def __init__(self, h5_path: str, emg_ds_factor: int = 8,
                 exclude_idx=None):
        with h5py.File(h5_path, "r") as f:
            eeg = f["eeg/epochs"][:]          # (N, 64, 2304)
            lbl = f["labels"][:].astype(np.int64) - 1  # 1/2 → 0/1
            if "emg" in f and "epochs" in f["emg"]:
                emg = f["emg/epochs"][:]      # (N, 4, 2304)
            else:
                emg = np.zeros((eeg.shape[0], 4, eeg.shape[2]), dtype=np.float32)

        if emg_ds_factor > 1:
            emg = emg[:, :, ::emg_ds_factor]  # → (N, 4, 288)

        n = min(eeg.shape[0], emg.shape[0], lbl.shape[0])
        eeg, emg, lbl = eeg[:n], emg[:n], lbl[:n]

        # ── v5: 오염 에폭 제거 ──────────────────────────────────
        keep = np.ones(n, dtype=bool)
        if exclude_idx:
            ex = np.asarray(sorted(set(int(i) for i in exclude_idx)), dtype=int)
            assert ex.min() >= 0 and ex.max() < n, (
                f"{os.path.basename(h5_path)}: 배제 인덱스가 [0,{n-1}] 범위를 "
                f"벗어남 (min={ex.min()}, max={ex.max()})"
            )
            keep[ex] = False
        self.epoch_idx = np.nonzero(keep)[0]          # 원본 h5 에폭 인덱스 보존

        self.eeg = torch.tensor(eeg[keep], dtype=torch.float32)
        self.emg = torch.tensor(emg[keep], dtype=torch.float32)
        self.lbl = torch.tensor(lbl[keep], dtype=torch.long)

    def __len__(self):
        return len(self.lbl)

    def __getitem__(self, idx):
        return self.eeg[idx], self.emg[idx], self.lbl[idx]


# ════════════════════════════════════════════════════════════════
#  Model 정의
# ════════════════════════════════════════════════════════════════

CFG = {
    "n_eeg_ch": 64, "n_emg_ch": 4, "n_times": 2304,
    "n_classes": 2, "emg_ds_factor": 8,
    "eegnet_F1": 8, "eegnet_D": 2, "eegnet_kern_len": 256, "eegnet_dropout": 0.5,
    "lstm_hidden": 128, "lstm_layers": 2, "lstm_dropout": 0.3,
    "clf_dropout": 0.3, "feat_dim": 256,
}
CFG["n_times_emg"] = CFG["n_times"] // CFG["emg_ds_factor"]


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
    def __init__(self, n_ch=4, hidden=128, n_layers=2,
                 dropout=0.3, feat_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(
            n_ch, hidden, n_layers, batch_first=True, bidirectional=True,
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
        w = F.softmax(self.attn(torch.cat([h_eeg, h_emg], dim=-1)), dim=-1)
        return w[:, 0:1] * self.W_eeg(h_eeg) + w[:, 1:2] * self.W_emg(h_emg), w


# ── 세 가지 모델 ─────────────────────────────────────────────────

class EEGOnlyModel(nn.Module):
    """EEG 스트림만 사용. EMG 입력을 받아도 무시."""
    def __init__(self, cfg):
        super().__init__()
        fd = cfg["feat_dim"]
        self.enc = EEGNetEncoder(
            cfg["n_eeg_ch"], cfg["n_times"],
            cfg["eegnet_F1"], cfg["eegnet_D"],
            cfg["eegnet_kern_len"], cfg["eegnet_dropout"], fd,
        )
        self.clf = nn.Sequential(
            nn.Linear(fd, 128), nn.ELU(),
            nn.Dropout(cfg["clf_dropout"]),
            nn.Linear(128, cfg["n_classes"]),
        )

    def forward(self, eeg, emg=None):
        return self.clf(self.enc(eeg)), None


class EMGOnlyModel(nn.Module):
    """sEMG 스트림만 사용. EEG 입력을 받아도 무시."""
    def __init__(self, cfg):
        super().__init__()
        fd = cfg["feat_dim"]
        self.enc = EMGBiLSTMEncoder(
            cfg["n_emg_ch"], cfg["lstm_hidden"],
            cfg["lstm_layers"], cfg["lstm_dropout"], fd,
        )
        self.clf = nn.Sequential(
            nn.Linear(fd, 128), nn.ELU(),
            nn.Dropout(cfg["clf_dropout"]),
            nn.Linear(128, cfg["n_classes"]),
        )

    def forward(self, eeg=None, emg=None):
        return self.clf(self.enc(emg)), None


class FusionModel(nn.Module):
    """EEG + sEMG Fusion (기존 HybridBCIModel)."""
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
        fused, w = self.fusion(self.eeg_enc(eeg), self.emg_enc(emg))
        return self.clf(fused), w


MODEL_CLASSES = {
    "eeg_only": EEGOnlyModel,
    "emg_only": EMGOnlyModel,
    "fusion":   FusionModel,
}


# ════════════════════════════════════════════════════════════════
#  학습 / 평가 함수
# ════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
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
def evaluate(model, loader, device, return_detail: bool = False):
    """v4와 동일한 예측 로직. return_detail=True 면 per-trial 상세도 반환."""
    model.eval()
    all_pred, all_true, all_prob, all_w = [], [], [], []
    for eeg, emg, lbl in loader:
        eeg, emg = eeg.to(device), emg.to(device)
        logits, w = model(eeg, emg)
        all_pred.extend(logits.argmax(1).cpu().tolist())
        all_true.extend(lbl.tolist())
        if return_detail:
            all_prob.append(F.softmax(logits.float(), dim=1).cpu().numpy())
            all_w.append(w.float().cpu().numpy() if w is not None else None)

    pred, true = np.array(all_pred), np.array(all_true)
    if not return_detail:
        return pred, true

    prob = np.concatenate(all_prob, axis=0)
    w_arr = (np.concatenate(all_w, axis=0)
             if all_w and all_w[0] is not None else None)
    return pred, true, prob, w_arr


def _save_per_trial(out_dir: str, model_type: str, seed: int, sid: int,
                    epoch_idx, true, pred, prob, w_arr) -> None:
    """T3: fold 평가 결과를 trial 단위로 저장."""
    d = {
        "epoch_idx":  np.asarray(epoch_idx, dtype=int),
        "true_label": true + 1,               # 0/1 → 1(left)/2(right)
        "pred_label": pred + 1,
        "correct":    (pred == true).astype(int),
        "prob_left":  prob[:, 0],
        "prob_right": prob[:, 1],
        "confidence": prob.max(axis=1),
    }
    if w_arr is not None:
        d["w_eeg"] = w_arr[:, 0]
        d["w_emg"] = w_arr[:, 1]

    pt_dir = os.path.join(out_dir, "per_trial")
    os.makedirs(pt_dir, exist_ok=True)
    path = os.path.join(pt_dir, f"{model_type}_seed{seed}_s{sid:02d}.csv")
    _save_atomic(pd.DataFrame(d), path)


def run_one_fold(sid: int, all_sids: list, model_type: str,
                 data_dir: str, out_dir: str, cfg: dict,
                 device: torch.device, excl: dict, seed: int) -> dict:
    """단일 LOSO fold 실행 (v5: 오염 에폭 배제 후 전 조건 재학습)."""
    # fold 단위 재시작에도 결과가 재현되도록 fold별로 시드를 고정
    set_seed(seed * 100000 + sid)

    train_sids = [s for s in all_sids if s != sid]
    test_path  = os.path.join(data_dir, f"sub-{sid:02d}_member_A.h5")

    # ── 데이터 로드 ─────────────────────────────────────────────
    test_ds  = BCIDataset(test_path, cfg["emg_ds_factor"],
                          exclude_idx=excl.get(str(sid), {}).get("excl_idx"))
    test_ldr = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # ── 전 조건 재학습 (v4의 fusion 체크포인트 재사용 경로는 제거) ──
    train_datasets = []
    for tr_sid in train_sids:
        tr_path = os.path.join(data_dir, f"sub-{tr_sid:02d}_member_A.h5")
        if os.path.exists(tr_path):
            train_datasets.append(BCIDataset(
                tr_path, cfg["emg_ds_factor"],
                exclude_idx=excl.get(str(tr_sid), {}).get("excl_idx")))

    from torch.utils.data import ConcatDataset
    train_ds  = ConcatDataset(train_datasets)
    train_ldr = DataLoader(train_ds, batch_size=32, shuffle=True,
                           num_workers=0, pin_memory=(device.type == "cuda"))

    model     = MODEL_CLASSES[model_type](cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler    = _amp_scaler() if device.type == "cuda" else None

    best_f1, patience_cnt = -1, 0
    PATIENCE, MAX_EPOCHS  = 20, 200
    best_state, best_epoch = None, -1

    for epoch in range(1, MAX_EPOCHS + 1):
        train_epoch(model, train_ldr, optimizer, scaler, device)

        # val: test set으로 조기종료 (v4와 동일 — 낙관 편향은 두 버전 공통)
        pred, true = evaluate(model, test_ldr, device)
        from sklearn.metrics import f1_score
        f1 = f1_score(true, pred, average="macro", zero_division=0)

        if f1 > best_f1:
            best_f1   = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt, best_epoch = 0, epoch
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    model.load_state_dict(best_state)
    pred, true, prob, w_arr = evaluate(model, test_ldr, device,
                                       return_detail=True)
    _save_per_trial(out_dir, model_type, seed, sid,
                    test_ds.epoch_idx, true, pred, prob, w_arr)

    m = _metrics(sid, pred, true)
    m["seed"]           = seed
    m["model_type"]     = model_type
    m["best_epoch"]     = best_epoch
    m["n_train_trials"] = len(train_ds)
    if w_arr is not None:
        m["w_eeg_mean"] = round(float(w_arr[:, 0].mean()), 6)
        m["w_emg_mean"] = round(float(w_arr[:, 1].mean()), 6)
        m["w_emg_std"]  = round(float(w_arr[:, 1].std()),  6)
    return m


def _metrics(sid: int, pred: np.ndarray, true: np.ndarray) -> dict:
    acc = accuracy_score(true, pred)
    # labels=[0,1] 지정: 한 클래스만 예측해도 안전
    try:
        kappa = cohen_kappa_score(true, pred, labels=[0, 1])
    except Exception:
        kappa = 0.0
    cm = confusion_matrix(true, pred, labels=[0, 1])
    left_recall  = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0.0
    right_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0.0

    return {
        "sid":          sid,
        "accuracy":     round(float(acc), 6),
        "kappa":        round(float(kappa), 6),
        "itr":          round(calc_itr(acc), 4),
        "left_recall":  round(float(left_recall), 6),
        "right_recall": round(float(right_recall), 6),
        "n_trials":     len(pred),
        "confusion":    cm.tolist(),
    }


# ════════════════════════════════════════════════════════════════
#  메인 LOSO 루프
# ════════════════════════════════════════════════════════════════

def _save_atomic(df: pd.DataFrame, path: str) -> None:
    """tmp 파일에 먼저 쓴 뒤 원자적으로 교체 — 쓰기 도중 파일 깨짐 방지."""
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _log(out_dir: str, msg: str) -> None:
    """진행 상황을 progress.log 에 append (T4)."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(out_dir, "progress.log"), "a") as f:
        f.write(f"[{stamp}] {msg}\n")


def run_loso(model_type: str, data_dir: str, out_dir: str,
             cfg: dict, sids: list, excl: dict, seed: int,
             pool_sids: list = None) -> list:
    """sids = 평가할 fold 목록, pool_sids = 학습에 쓸 전체 피험자 풀.

    D3 로 평가에서 빠진 피험자도 그의 clean trial 은 학습에 계속 쓴다.
    """
    pool_sids = pool_sids or sids
    device = pick_device()
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir,
                            f"ablation_{model_type}_seed{seed}_results.csv")

    # ── 이전 결과 복원 (재개) ────────────────────────────────────
    if os.path.exists(csv_path):
        done_df  = pd.read_csv(csv_path)
        done_ids = set(done_df["sid"].astype(int).tolist())
        results  = done_df.to_dict("records")
        print(f"  📂 이전 결과 복원: {len(done_ids)}명 스킵")
    else:
        done_ids, results = set(), []

    remaining = [s for s in sids if s not in done_ids]

    print(f"\n{'='*60}")
    print(f"  Ablation v5-clean: {model_type.upper()}  |  seed={seed}  |  device={device}")
    print(f"  평가 대상 {len(sids)}명  |  완료 {len(done_ids)}명  |  남은 {len(remaining)}명")
    print(f"{'='*60}\n")
    _log(out_dir, f"START {model_type} seed={seed} device={device} "
                  f"remaining={len(remaining)}")

    for i, sid in enumerate(remaining, len(done_ids) + 1):
        t0 = time.time()
        r  = run_one_fold(sid, pool_sids, model_type, data_dir, out_dir,
                          cfg, device, excl, seed)
        elapsed = time.time() - t0
        line = (f"[{i:2d}/{len(sids)}] s{sid:02d} | "
                f"acc={r['accuracy']:.4f}  κ={r['kappa']:.4f}  "
                f"ITR={r['itr']:.2f}  n={r['n_trials']}  "
                f"ep={r['best_epoch']}  [{elapsed:.0f}s]")
        print("  " + line)
        results.append(r)
        # fold 완료 즉시 저장 → 중단 시 이 지점부터 재개
        _save_atomic(pd.DataFrame(results), csv_path)
        _log(out_dir, f"{model_type} seed={seed} " + line)

    print(f"\n  저장: {csv_path}")

    accs   = [r["accuracy"] for r in results]
    kappas = [r["kappa"]    for r in results]
    itrs   = [r["itr"]      for r in results]
    print(f"\n  {model_type.upper()} 요약:")
    print(f"    Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"    Kappa    : {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")
    print(f"    ITR      : {np.mean(itrs):.4f} ± {np.std(itrs):.4f}")

    return results




def merge_and_summarize(out_dir: str, seeds: list, conds: list):
    """조건 × seed CSV 를 병합 → ablation_v5_results.csv / _summary.json."""
    frames = []
    for cond in conds:
        for sd in seeds:
            p = os.path.join(out_dir, f"ablation_{cond}_seed{sd}_results.csv")
            if os.path.exists(p):
                df = pd.read_csv(p)
                df["model_type"] = cond
                df["seed"] = sd
                frames.append(df)

    if not frames:
        print("  병합할 결과 없음")
        return

    allr = pd.concat(frames, ignore_index=True)
    _save_atomic(allr, os.path.join(out_dir, "ablation_v5_results.csv"))
    print(f"  병합 저장: ablation_v5_results.csv  ({len(allr)} rows)")

    summary = {}
    for cond in conds:
        sub = allr[allr.model_type == cond]
        if sub.empty:
            continue
        # seed 별 피험자 평균 → seed 간 SD
        per_seed = sub.groupby("seed")[["accuracy", "kappa", "itr"]].mean()
        # 전 seed 를 합친 피험자 단위 평균 (피험자 간 SD)
        per_sid = sub.groupby("sid")[["accuracy", "kappa", "itr"]].mean()
        summary[cond] = {
            "n_subjects":       int(sub["sid"].nunique()),
            "seeds":            sorted(int(s) for s in sub["seed"].unique()),
            "accuracy_mean":    round(float(per_sid["accuracy"].mean()), 4),
            "accuracy_std_subj": round(float(per_sid["accuracy"].std()), 4),
            "accuracy_std_seed": round(float(per_seed["accuracy"].std()), 4)
                                 if len(per_seed) > 1 else None,
            "kappa_mean":       round(float(per_sid["kappa"].mean()), 4),
            "kappa_std_subj":   round(float(per_sid["kappa"].std()), 4),
            "kappa_std_seed":   round(float(per_seed["kappa"].std()), 4)
                                 if len(per_seed) > 1 else None,
            "itr_mean":         round(float(per_sid["itr"].mean()), 4),
            "itr_std_subj":     round(float(per_sid["itr"].std()), 4),
        }

    with open(os.path.join(out_dir, "ablation_v5_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("  요약 저장: ablation_v5_summary.json")
    for c, v in summary.items():
        print(f"    {c:9s} acc={v['accuracy_mean']:.4f}±{v['accuracy_std_subj']:.4f}"
              f"  κ={v['kappa_mean']:.4f}  (n={v['n_subjects']}, "
              f"seeds={v['seeds']})")


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def load_exclusions(path: str):
    with open(path) as f:
        excl = json.load(f)
    n_ex = sum(len(v["excl_idx"]) for v in excl.values())
    n_tot = sum(v["n_total"] for v in excl.values())
    print(f"  배제 로드: {len(excl)}명 / {n_ex}개 에폭 제거 "
          f"({100*n_ex/n_tot:.2f}% of {n_tot})")
    return excl


def parse_args():
    p = argparse.ArgumentParser(
        description="BCI Ablation v5-clean — 움직임 오염 trial 배제 후 재학습",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_type", type=str,
                   choices=["eeg_only", "emg_only", "fusion"],
                   help="실행할 조건 (--merge 시 생략 가능)")
    p.add_argument("--seed", type=int, default=42, help="난수 시드 (D4)")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--exclude_json", type=str, default=None,
                   help="excluded_epochs.json 경로")
    p.add_argument("--sids", type=int, nargs="+", default=None,
                   help="평가할 피험자 (기본: excluded_epochs.json 의 eligible)")
    p.add_argument("--include_ineligible", action="store_true",
                   help="D3 미달(클래스당 <30 trial) 피험자도 평가에 포함")
    p.add_argument("--merge", action="store_true",
                   help="결과 병합만 수행")
    p.add_argument("--merge_seeds", type=int, nargs="+",
                   default=[42, 1337, 2024])
    return p.parse_args()


def main():
    args = parse_args()
    local = Path(__file__).resolve().parent.parent
    data_dir = args.data_dir or str(local / "BCI_Research" / "preprocessed" / "member_A")
    out_dir = args.out_dir or str(local / "BCI_Research" / "results" / "ablation_v5_clean")
    excl_json = args.exclude_json or os.path.join(out_dir, "excluded_epochs.json")
    os.makedirs(out_dir, exist_ok=True)

    if args.merge:
        merge_and_summarize(out_dir, args.merge_seeds,
                            ["eeg_only", "emg_only", "fusion"])
        return

    if not args.model_type:
        raise SystemExit("--model_type 이 필요합니다 (또는 --merge)")

    excl = load_exclusions(excl_json)

    # 학습 풀 = 플래그를 확보한 전 피험자 (D1: s06 은 .mat 부재로 제외)
    pool_sids = sorted(int(k) for k in excl.keys())
    # 평가 대상 = D3 기준 통과 피험자
    if args.sids:
        eval_sids = args.sids
    elif args.include_ineligible:
        eval_sids = pool_sids
    else:
        eval_sids = [s for s in pool_sids if excl[str(s)]["eligible"]]
        dropped = [s for s in pool_sids if not excl[str(s)]["eligible"]]
        if dropped:
            print(f"  D3: 클래스당 <30 trial → 평가 제외 {dropped} "
                  f"(학습에는 계속 사용)")

    cfg = dict(CFG)
    run_loso(args.model_type, data_dir, out_dir, cfg,
             eval_sids, excl, args.seed, pool_sids=pool_sids)
    merge_and_summarize(out_dir, [args.seed], [args.model_type])


if __name__ == "__main__":
    main()
