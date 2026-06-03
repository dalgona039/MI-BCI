"""
transfer_bcic2a.py — Zero-shot Cross-Dataset Transfer: GigaDB → BCIC IV 2a
============================================================================
실험 설계:
  1. GigaDB 52명 전체 데이터로 EEGNet 22ch EEG-only 모델 학습 (all-subjects 학습)
  2. BCIC Competition IV Dataset 2a (9명, Left/Right 손 MI, A0xT.gdf + A0xE.gdf) 에
     동일 모델을 zero-shot으로 평가
  3. 채널 매핑: GigaDB 64ch 중 BCIC IV 2a와 겹치는 22개 채널만 사용
  4. 샘플링: GigaDB 512Hz 에포크의 MI 구간(0~4s, 2048 samples)만 사용;
            BCIC IV 2a(250Hz, 4s)를 scipy.signal.resample으로 512Hz 업샘플링

실행 방법 (Colab 권장):
  python transfer_bcic2a.py \\
      --data_dir       BCI_Research/preprocessed/member_A \\
      --bcic_dir       BCICIV_2a_gdf \\
      --out_dir        BCI_Research/results/transfer_bcic2a \\
      [--skip_train]   # 체크포인트가 이미 있으면 학습 건너뜀

출력:
  results/transfer_bcic2a/
  ├── transfer_22ch_model.pt          ← GigaDB all-52 학습 체크포인트
  ├── transfer_bcic2a_results.csv     ← 피험자별 accuracy, κ, ITR
  └── transfer_bcic2a_summary.json    ← 평균 ± SD
"""

import os, sys, json, time, random, argparse, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from scipy.signal import butter, filtfilt, resample
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import mne
mne.set_log_level('ERROR')

# ─── 재현성 ────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# ─── AMP 헬퍼 ──────────────────────────────────────────────────
def _amp_autocast(enabled):
    try:    return torch.amp.autocast(device_type="cuda", enabled=enabled)
    except: return torch.cuda.amp.autocast(enabled=enabled)

def _amp_scaler():
    try:    return torch.amp.GradScaler("cuda")
    except: return torch.cuda.amp.GradScaler()

# ══════════════════════════════════════════════════════════════
#  채널 매핑 상수
# ══════════════════════════════════════════════════════════════

# GigaDB 64ch 순서 (전처리 노트북 EEG_CH_NAMES 동일)
GIGADB_64 = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'FC5','FC3','FC1','FCz','FC2','FC4','FC6',
    'T7','C5','C3','C1','Cz','C2','C4','C6','T8',
    'TP9','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP10',
    'P7','P5','P3','P1','Pz','P2','P4','P6','P8',
    'PO9','PO7','PO3','POz','PO4','PO8','PO10',
    'O1','Oz','O2',
    'Iz','Fpz',
    'AF7','AF3','AFz','AF4','AF8',
    'FT9','FT7','FT8','FT10','TP7','TP8',
]

# BCIC IV 2a 22ch 표준 이름 (채널 순서는 GDF 파일 기준)
BCIC_22 = [
    'Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1',
    'Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4',
    'P1','Pz','P2','Oz',
]

# GigaDB 64ch 중 BCIC 22ch에 해당하는 인덱스 (사전 계산)
GIGADB_22_IDX = [GIGADB_64.index(ch) for ch in BCIC_22]
# = [4,8,9,10,11,12,15,16,17,18,19,20,21,25,26,27,28,29,35,36,37,49]

# GDF 파일의 채널 이름 → 표준 이름 매핑
# MNE로 읽으면 EEG-C3, EEG-0 등으로 표시됨 → 위치 기반으로 처리
BCIC_GDF_22_IDX = list(range(22))  # GDF의 첫 22ch가 순서대로 BCIC_22에 매핑

# BCIC IV 2a 이벤트 코드
BCIC_LEFT_CODE  = 769   # class 1 = Left hand MI
BCIC_RIGHT_CODE = 770   # class 2 = Right hand MI

# 샘플링/에포크 파라미터
FS_GIGADB   = 512    # GigaDB 원본
FS_BCIC     = 250    # BCIC IV 2a 원본
FS_TARGET   = 512    # 모델 학습/추론 기준
N_TIMES     = 2048   # 4s × 512 Hz (MI 구간만, baseline 제외)
# GigaDB HDF5 에포크: -0.5s~4.0s = 4.5s = 2304 samples at 512Hz
# MI 구간 = 0s~4.0s → samples[256:2304]
GIGADB_MI_START = 256   # 0.5s × 512Hz
GIGADB_MI_END   = 2304  # 4.5s × 512Hz
# BCIC: 0s~4s @250Hz → 1000 samples → upsample to 512Hz → 2048 samples
BCIC_EPOCH_SEC  = 4.0
BCIC_EPOCH_SAMP = int(BCIC_EPOCH_SEC * FS_BCIC)   # 1000
BCIC_TMIN       = 0.0   # cue onset

# ══════════════════════════════════════════════════════════════
#  ITR 계산
# ══════════════════════════════════════════════════════════════
def calc_itr(acc, n_classes=2, trial_sec=4.0):
    p = np.clip(acc, 1e-9, 1.0 - 1e-9)
    B = (np.log2(n_classes)
         + p * np.log2(p)
         + (1 - p) * np.log2((1 - p) / (n_classes - 1)))
    return float(max(0.0, B) * 60.0 / trial_sec)

# ══════════════════════════════════════════════════════════════
#  GigaDB 데이터셋 (22ch, MI 구간만)
# ══════════════════════════════════════════════════════════════
class GigaDB22Dataset(Dataset):
    """GigaDB HDF5에서 22채널 서브스페이스 + MI 구간(0~4s)만 로드."""
    def __init__(self, h5_path: str):
        with h5py.File(h5_path, 'r') as f:
            eeg = f['eeg/epochs'][:]   # (N, 64, 2304)
            lbl = f['labels'][:].astype(np.int64) - 1  # 1/2 → 0/1
        # 22ch 서브스페이스 + MI 구간
        eeg22 = eeg[:, GIGADB_22_IDX, GIGADB_MI_START:GIGADB_MI_END]  # (N,22,2048)
        self.eeg = torch.tensor(eeg22, dtype=torch.float32)
        self.lbl = torch.tensor(lbl,   dtype=torch.long)

    def __len__(self):  return len(self.lbl)
    def __getitem__(self, i): return self.eeg[i], self.lbl[i]

# ══════════════════════════════════════════════════════════════
#  BCIC IV 2a 전처리 및 데이터셋
# ══════════════════════════════════════════════════════════════
def bandpass_filter(data, low=4.0, high=40.0, fs=512, order=4):
    """Zero-phase Butterworth bandpass."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)

def load_bcic_subject(gdf_train: str, gdf_eval: str) -> tuple:
    """
    BCIC IV 2a 피험자 GDF 파일 로드 → Left/Right 손 MI 에포크 추출.
    반환: (eeg_array, labels)
      eeg_array: (N, 22, 2048) float32  — 512Hz 업샘플된 4s 에포크
      labels:    (N,)          int64    — 0=Left, 1=Right
    """
    all_eeg, all_lbl = [], []

    for gdf_path in [gdf_train, gdf_eval]:
        if not os.path.exists(gdf_path):
            print(f'  [SKIP] {gdf_path} not found')
            continue

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)

        # EEG 22채널만 선택 (EOG 제외)
        eeg_picks = [i for i, ch in enumerate(raw.ch_names)
                     if not ch.startswith('EOG')][:22]
        data = raw.get_data(picks=eeg_picks)  # (22, T)

        # 4–40 Hz 밴드패스 (GigaDB 전처리와 동일)
        data = bandpass_filter(data, low=4.0, high=40.0, fs=FS_BCIC, order=4)

        # 이벤트 추출
        events, eid = mne.events_from_annotations(raw, verbose=False)
        # Left MI = 769, Right MI = 770
        left_code  = eid.get(str(BCIC_LEFT_CODE),  None)
        right_code = eid.get(str(BCIC_RIGHT_CODE), None)
        if left_code is None or right_code is None:
            # 일부 eval 파일은 cue 없이 연속 기록 → 건너뜀
            print(f'  [SKIP] {gdf_path}: cue 이벤트 없음')
            continue

        target_codes = {left_code: 0, right_code: 1}

        for evt in events:
            onset, _, code = evt
            if code not in target_codes:
                continue
            start = onset                           # cue onset
            end   = start + BCIC_EPOCH_SAMP         # +4s @250Hz
            if end > data.shape[1]:
                continue
            epoch = data[:, start:end]              # (22, 1000) @250Hz

            # 업샘플: 250 → 512 Hz (resampling ratio = 512/250)
            # scipy.signal.resample_poly or resample
            # resample: (22, 1000) → (22, 2048)  [512/250 * 1000 = 2048.0 exactly]
            epoch_512 = resample(epoch, N_TIMES, axis=-1)  # (22, 2048)

            # z-score 정규화 (per-channel, per-trial — GigaDB와 동일)
            mu  = epoch_512.mean(axis=-1, keepdims=True)
            std = epoch_512.std(axis=-1, keepdims=True) + 1e-8
            epoch_512 = (epoch_512 - mu) / std

            all_eeg.append(epoch_512.astype(np.float32))
            all_lbl.append(target_codes[code])

    if not all_eeg:
        return None, None

    return np.stack(all_eeg), np.array(all_lbl, dtype=np.int64)


class BCICDataset(Dataset):
    def __init__(self, eeg: np.ndarray, labels: np.ndarray):
        self.eeg = torch.tensor(eeg,    dtype=torch.float32)
        self.lbl = torch.tensor(labels, dtype=torch.long)
    def __len__(self):  return len(self.lbl)
    def __getitem__(self, i): return self.eeg[i], self.lbl[i]

# ══════════════════════════════════════════════════════════════
#  모델 (EEGNet 22ch EEG-only)
# ══════════════════════════════════════════════════════════════
class EEGNet22(nn.Module):
    """EEGNet for 22-channel 2048-sample input."""
    def __init__(self, n_ch=22, n_times=2048, F1=8, D=2,
                 kern_len=256, dropout=0.5, feat_dim=256, n_classes=2):
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
        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, feat_dim), nn.ELU(),
            nn.Linear(feat_dim, 128), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.clf(self.block3(self.block2(self.block1(x.unsqueeze(1)))))

# ══════════════════════════════════════════════════════════════
#  학습
# ══════════════════════════════════════════════════════════════
def train_all_subjects(data_dir: str, ckpt_path: str,
                       device: torch.device,
                       n_subjects: int = 52,
                       max_epochs: int = 100,
                       patience: int = 15) -> nn.Module:
    """GigaDB 전체 52명으로 EEGNet22 학습."""
    print(f'\n{"="*60}')
    print(f'  GigaDB 22ch EEG-only 학습 (all {n_subjects} subjects)')
    print(f'  Device: {device}')
    print(f'{"="*60}\n')

    # 데이터 로드
    all_ds = []
    for sid in range(1, n_subjects + 1):
        h5 = os.path.join(data_dir, f'sub-{sid:02d}_member_A.h5')
        if os.path.exists(h5):
            all_ds.append(GigaDB22Dataset(h5))
        else:
            print(f'  [SKIP] sub-{sid:02d} not found')

    full_ds = ConcatDataset(all_ds)
    n_total = len(full_ds)
    n_val   = max(int(n_total * 0.1), 64)
    n_train = n_total - n_val

    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED))

    train_ldr = DataLoader(train_ds, batch_size=64, shuffle=True,
                           num_workers=2, pin_memory=True)
    val_ldr   = DataLoader(val_ds,   batch_size=64, shuffle=False,
                           num_workers=0)

    model  = EEGNet22().to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epochs)
    scaler = _amp_scaler() if device.type == 'cuda' else None

    best_val_acc, patience_cnt, best_state = 0.0, 0, None

    for epoch in range(1, max_epochs + 1):
        # ── 학습 ──
        model.train()
        for eeg, lbl in train_ldr:
            eeg, lbl = eeg.to(device), lbl.to(device)
            optim.zero_grad()
            with _amp_autocast(scaler is not None):
                loss = F.cross_entropy(model(eeg), lbl)
            if scaler:
                scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
            else:
                loss.backward(); optim.step()
        sched.step()

        # ── 검증 ──
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for eeg, lbl in val_ldr:
                preds.extend(model(eeg.to(device)).argmax(1).cpu().tolist())
                trues.extend(lbl.tolist())
        val_acc = accuracy_score(trues, preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f'  Early stop @ epoch {epoch}, best val acc = {best_val_acc:.4f}')
                break

        if epoch % 10 == 0:
            print(f'  Epoch {epoch:3d} | val_acc={val_acc:.4f} '
                  f'(best={best_val_acc:.4f}, patience={patience_cnt}/{patience})')

    model.load_state_dict(best_state)
    torch.save(best_state, ckpt_path)
    print(f'  모델 저장: {ckpt_path}')
    return model

# ══════════════════════════════════════════════════════════════
#  BCIC IV 2a 평가
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_bcic(model: nn.Module, bcic_dir: str,
                  out_dir: str, device: torch.device) -> list:
    """9명 BCIC IV 2a 피험자 zero-shot 평가."""
    print(f'\n{"="*60}')
    print('  BCIC IV 2a Zero-shot 평가')
    print(f'{"="*60}\n')

    model.eval()
    results = []

    for sid in range(1, 10):
        letter = chr(ord('A') + sid - 1)   # A01→A, A02→B, ...
        t_gdf  = os.path.join(bcic_dir, f'A0{sid}T.gdf')
        e_gdf  = os.path.join(bcic_dir, f'A0{sid}E.gdf')

        eeg, lbl = load_bcic_subject(t_gdf, e_gdf)
        if eeg is None:
            print(f'  [SKIP] A0{sid}: 데이터 로드 실패')
            continue

        ds  = BCICDataset(eeg, lbl)
        ldr = DataLoader(ds, batch_size=64, shuffle=False)

        preds, trues = [], []
        for x, y in ldr:
            preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
            trues.extend(y.tolist())

        pred_arr = np.array(preds)
        true_arr = np.array(trues)
        acc   = accuracy_score(true_arr, pred_arr)
        try:
            kappa = cohen_kappa_score(true_arr, pred_arr, labels=[0, 1])
        except:
            kappa = 0.0
        itr   = calc_itr(acc, trial_sec=4.0)
        cm    = confusion_matrix(true_arr, pred_arr, labels=[0, 1])
        lr    = cm[0,0]/cm[0].sum() if cm[0].sum() > 0 else 0.0
        rr    = cm[1,1]/cm[1].sum() if cm[1].sum() > 0 else 0.0

        r = dict(subject=f'A0{sid}', n_trials=len(trues),
                 accuracy=round(float(acc),6), kappa=round(float(kappa),6),
                 itr=round(itr,4),
                 left_recall=round(float(lr),6), right_recall=round(float(rr),6))
        results.append(r)

        print(f'  A0{sid} | n={len(trues):3d} | '
              f'acc={acc:.4f}  κ={kappa:.4f}  ITR={itr:.2f}')

    # 저장
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'transfer_bcic2a_results.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f'\n  결과 저장: {csv_path}')

    # 요약
    accs   = [r['accuracy'] for r in results]
    kappas = [r['kappa']    for r in results]
    itrs   = [r['itr']      for r in results]
    summary = {
        'n_subjects':    len(results),
        'accuracy_mean': round(float(np.mean(accs)),  4),
        'accuracy_std':  round(float(np.std(accs)),   4),
        'kappa_mean':    round(float(np.mean(kappas)),4),
        'kappa_std':     round(float(np.std(kappas)), 4),
        'itr_mean':      round(float(np.mean(itrs)),  4),
        'itr_std':       round(float(np.std(itrs)),   4),
        'transfer_config': {
            'source_dataset': 'GigaDB (Cho et al. 2017), 52 subjects',
            'target_dataset': 'BCI Competition IV 2a, 9 subjects',
            'n_channels':     22,
            'channel_names':  BCIC_22,
            'source_fs':      FS_GIGADB,
            'target_fs_orig': FS_BCIC,
            'target_fs_resampled': FS_TARGET,
            'n_times':        N_TIMES,
            'epoch_sec':      4.0,
            'model':          'EEGNet22 (EEG-only, trained on all 52 GigaDB subjects)',
            'transfer_type':  'zero-shot (no fine-tuning on target data)',
        }
    }
    sum_path = os.path.join(out_dir, 'transfer_bcic2a_summary.json')
    with open(sum_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'  요약 저장: {sum_path}')

    print(f'\n  ── 요약 ──')
    print(f'  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    print(f'  Kappa    : {np.mean(kappas):.4f} ± {np.std(kappas):.4f}')
    print(f'  ITR      : {np.mean(itrs):.4f} ± {np.std(itrs):.4f}')

    return results

# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description='Zero-shot Transfer: GigaDB(22ch) → BCIC IV 2a',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data_dir',   default='BCI_Research/preprocessed/member_A',
                   help='GigaDB HDF5 디렉터리')
    p.add_argument('--bcic_dir',   default='BCICIV_2a_gdf',
                   help='BCIC IV 2a GDF 파일 디렉터리')
    p.add_argument('--out_dir',    default='BCI_Research/results/transfer_bcic2a',
                   help='결과 저장 디렉터리')
    p.add_argument('--ckpt',       default=None,
                   help='체크포인트 저장/로드 경로 (기본: out_dir/transfer_22ch_model.pt)')
    p.add_argument('--skip_train', action='store_true',
                   help='체크포인트가 있으면 학습 건너뜀')
    p.add_argument('--drive_root', default=None,
                   help='Colab Drive 경로 (예: /content/drive/MyDrive/BCI_Research)')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 경로 결정
    if args.drive_root:
        root     = Path(args.drive_root)
        data_dir = args.data_dir or str(root / 'preprocessed' / 'member_A')
        bcic_dir = args.bcic_dir or str(root.parent / 'BCICIV_2a_gdf')
        out_dir  = args.out_dir  or str(root / 'results' / 'transfer_bcic2a')
    else:
        base     = Path(__file__).resolve().parent.parent
        data_dir = str(base / args.data_dir)
        bcic_dir = str(base / args.bcic_dir)
        out_dir  = str(base / args.out_dir)

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = args.ckpt or os.path.join(out_dir, 'transfer_22ch_model.pt')

    # 1. 학습
    if args.skip_train and os.path.exists(ckpt_path):
        print(f'  체크포인트 로드: {ckpt_path}')
        model = EEGNet22().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    else:
        model = train_all_subjects(data_dir, ckpt_path, device)

    # 2. 평가
    evaluate_bcic(model, bcic_dir, out_dir, device)


if __name__ == '__main__':
    main()
