"""
generate_s5_notebook.py
S5_Bias_Fix_Colab.ipynb 를 생성합니다.
세 개의 .py 파일을 읽어 %%writefile 셀로 내장합니다.

실행:
  python notebooks/generate_s5_notebook.py
"""
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
OUT  = ROOT / "notebooks" / "S5_Bias_Fix_Colab.ipynb"


def cell_id():
    return uuid.uuid4().hex[:16]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id(),
        "metadata": {},
        "source": [text],
    }


def code(source: str, outputs=None) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id(),
        "metadata": {},
        "outputs": outputs or [],
        "source": [source],
    }


def writefile_cell(colab_path: str, py_path: Path) -> dict:
    """%%writefile 매직으로 .py 파일 내용을 Colab에 씁니다."""
    content = py_path.read_text(encoding="utf-8")
    source  = f"%%writefile {colab_path}\n" + content
    return code(source)


# ════════════════════════════════════════════════════════════════
#  셀 정의
# ════════════════════════════════════════════════════════════════

cells = []

# ── 셀 0: 제목 ───────────────────────────────────────────────────
cells.append(md("""\
# S5 Bias Fix — Right MI Bias 수정 실험

**목표**: HybridBCIModel의 Right MI 과분류 문제(9/52명, 17%) 수정

**실패한 시도 (재실행 금지)**:
- v5 Label Smoothing → kappa 손실 12~13% → 기각
- v6 Class Weighting → kappa 손실 7.5~11% → 기각

**이번 실험**:
1. **셀 1~3**: 스크립트 작성 (%%writefile)
2. **셀 4**: Drive 마운트 & 패키지 설치
3. **셀 5**: HDF5 데이터 로컬 복사 (Drive 끊김 방지)
4. **셀 6**: Calibration 실행 (~5분, 추론만)
5. **셀 7**: Flip Aug Fine-tune (~25분, A100 기준)
6. **셀 8**: 최종 비교 리포트 출력
"""))

# ── 셀 1: calibration.py writefile ───────────────────────────────
cells.append(md("## 셀 1: calibration.py 작성"))
cells.append(writefile_cell("/content/calibration.py", SRC / "calibration.py"))

# ── 셀 2: train_flip_aug.py writefile ────────────────────────────
cells.append(md("## 셀 2: train_flip_aug.py 작성"))
cells.append(writefile_cell("/content/train_flip_aug.py", SRC / "train_flip_aug.py"))

# ── 셀 3: train_flip_full.py writefile ───────────────────────────
cells.append(md("## 셀 3: train_flip_full.py 작성"))
cells.append(writefile_cell("/content/train_flip_full.py", SRC / "train_flip_full.py"))

# ── 셀 4: bias_fix_report.py writefile ───────────────────────────
cells.append(md("## 셀 4: bias_fix_report.py 작성"))
cells.append(writefile_cell("/content/bias_fix_report.py", SRC / "bias_fix_report.py"))

# ── 셀 5: Drive 마운트 & 설치 ─────────────────────────────────────
cells.append(md("## 셀 5: Drive 마운트 & 환경 설정"))
cells.append(code("""\
import os, subprocess, sys
from google.colab import drive

if os.path.isdir('/content/drive/MyDrive'):
    print('Drive 이미 마운트됨 ✅')
else:
    drive.mount('/content/drive')

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'h5py', 'scikit-learn'], check=False)

# BCI_Research 폴더 위치로 DRIVE_ROOT 결정
# DRIVE_ROOT = BCI_Research의 부모 폴더
_mydrive = '/content/drive/MyDrive'
if os.path.isdir(f'{_mydrive}/BCI_Research'):
    DRIVE_ROOT = _mydrive                          # BCI_Research가 MyDrive 바로 아래
elif os.path.isdir(f'{_mydrive}/MI-BCI/BCI_Research'):
    DRIVE_ROOT = f'{_mydrive}/MI-BCI'
else:
    DRIVE_ROOT = _mydrive   # ← BCI_Research 위치가 다르면 직접 수정

BCI_ROOT = os.path.join(DRIVE_ROOT, 'BCI_Research')
assert os.path.isdir(BCI_ROOT), f'BCI_Research 없음: {BCI_ROOT}'

gpu = subprocess.getoutput('nvidia-smi --query-gpu=name --format=csv,noheader')
print(f'DRIVE_ROOT : {DRIVE_ROOT}')
print(f'BCI_ROOT   : {BCI_ROOT}')
print(f'GPU        : {gpu}')
print('완료 ✅')
"""))

# ── 셀 6: HDF5 데이터 로컬 복사 ──────────────────────────────────
cells.append(md("## 셀 6: HDF5 데이터 로컬 복사 (Drive 끊김 방지, ~3분)"))
cells.append(code("""\
import shutil, os

LOCAL_DATA = '/content/bci_data'
os.makedirs(LOCAL_DATA, exist_ok=True)

drive_data = os.path.join(BCI_ROOT, 'preprocessed', 'member_A')
h5_files   = sorted(f for f in os.listdir(drive_data) if f.endswith('.h5'))
print(f'복사 대상: {len(h5_files)}개 HDF5 파일')

for i, fname in enumerate(h5_files, 1):
    dst = f'{LOCAL_DATA}/{fname}'
    if os.path.exists(dst):
        print(f'  [{i:2d}/{len(h5_files)}] 이미 있음: {fname}')
    else:
        shutil.copy2(f'{drive_data}/{fname}', dst)
        if i % 10 == 0 or i == len(h5_files):
            print(f'  [{i:2d}/{len(h5_files)}] 복사 중...')

print(f'\\n완료: {LOCAL_DATA}')
"""))

# ── 셀 7: Calibration (전체) 실행 ─────────────────────────────────
cells.append(md("""\
## 셀 7: Post-hoc Logit Calibration (전체)

재학습 없이 v4 체크포인트의 logit bias를 추정하고 보정합니다. (전체 피험자에 적용)
- 예상 시간: **~5분** (순수 추론)
- 출력: `BCI_Research/results/calibration/calibration_results.csv`
"""))
cells.append(code("""\
!python /content/calibration.py \\
    --drive_root {DRIVE_ROOT} \\
    --data_dir /content/bci_data \\
    --device cuda
"""))

# ── 셀 8: Calibration (조건부) 실행 ──────────────────────────────
cells.append(md("""\
## 셀 8: Post-hoc Logit Calibration (조건부)

bias > 0 인 피험자에만 보정을 적용합니다. (음수 bias 피험자 보호)
- 예상 시간: **~5분** (순수 추론)
- 출력: `BCI_Research/results/calibration/conditional_calibration_results.csv`
"""))
cells.append(code("""\
!python /content/calibration.py \\
    --drive_root {DRIVE_ROOT} \\
    --data_dir /content/bci_data \\
    --device cuda \\
    --conditional
"""))

# ── 셀 9: Flip Aug Fine-tune ──────────────────────────────────────
cells.append(md("""\
## 셀 9: Hemispheric Flip Aug — Fine-tune (v4 checkpoint에서)

Left MI 에포크를 채널 좌우 반전, v4 체크포인트에서 fine-tune합니다.
- 예상 시간: **~20~30분** (A100 기준)
- 출력: `BCI_Research/results/checkpoints_flip/`, `flip_aug_results.csv`
- 중단 후 재실행해도 이전 fold 이어서 실행됩니다.
"""))
cells.append(code("""\
!python /content/train_flip_aug.py \\
    --drive_root {DRIVE_ROOT} \\
    --data_dir /content/bci_data \\
    --device cuda
"""))

# ── 셀 10: Flip Full Retrain ──────────────────────────────────────
cells.append(md("""\
## 셀 10: Hemispheric Flip Aug — Full LOSO Retraining (random init)

Left MI 에포크를 채널 좌우 반전, 랜덤 초기화에서 전체 LOSO 재학습합니다.
- 예상 시간: **~3~4시간** (A100 기준, 52 fold × 200 epochs)
- 출력: `BCI_Research/results/checkpoints_flip_full/`, `flip_full_results.csv`
- 중단 후 재실행해도 이전 fold 이어서 실행됩니다.
"""))
cells.append(code("""\
!python /content/train_flip_full.py \\
    --drive_root {DRIVE_ROOT} \\
    --data_dir /content/bci_data \\
    --device cuda
"""))

# ── 셀 11: 최종 비교 리포트 ───────────────────────────────────────
cells.append(md("""\
## 셀 11: 최종 비교 리포트

v4 baseline / Calibration (전체/조건부) / Flip Aug (fine-tune/full) / Cal+Flip 여섯 전략을 비교합니다.
"""))
cells.append(code("""\
!python /content/bias_fix_report.py \\
    --drive_root {DRIVE_ROOT}
"""))

# ── 셀 12: 결과 시각화 ────────────────────────────────────────────
cells.append(md("## 셀 12: 결과 시각화"))
cells.append(code("""\
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

BCI_ROOT  = os.path.join(DRIVE_ROOT, 'BCI_Research')
cal_dir   = os.path.join(BCI_ROOT, 'results', 'calibration')
abl_dir   = os.path.join(BCI_ROOT, 'results', 'ablation')
BIAS_SIDS = [1, 5, 7, 11, 12, 15, 24, 34, 36]

def safe_read(path):
    return pd.read_csv(path) if os.path.exists(path) else None

cal_df      = safe_read(os.path.join(cal_dir, 'calibration_results.csv'))
cond_cal_df = safe_read(os.path.join(cal_dir, 'conditional_calibration_results.csv'))
flip_df     = safe_read(os.path.join(abl_dir, 'flip_aug_results.csv'))
flip_full_df = safe_read(os.path.join(abl_dir, 'flip_full_results.csv'))

# ── 전략별 recall_diff / kappa 수집 ──────────────────────────────
recall_strategies = {}
kappa_strategies  = {}

if cal_df is not None:
    b = cal_df[cal_df['sid'].isin(BIAS_SIDS)]
    recall_strategies['v4 baseline']  = b['right_recall_before'] - b['left_recall_before']
    recall_strategies['Cal (전체)']   = b['right_recall_after']  - b['left_recall_after']
    kappa_strategies['v4 baseline']   = cal_df['kappa_before']
    kappa_strategies['Cal (전체)']    = cal_df['kappa_after']

if cond_cal_df is not None:
    b = cond_cal_df[cond_cal_df['sid'].isin(BIAS_SIDS)]
    recall_strategies['Cal (조건부)'] = b['right_recall_after']  - b['left_recall_after']
    kappa_strategies['Cal (조건부)']  = cond_cal_df['kappa_after']

if flip_df is not None:
    b = flip_df[flip_df['sid'].isin(BIAS_SIDS)]
    recall_strategies['Flip (fine)'] = b['right_recall_flip'] - b['left_recall_flip']
    kappa_strategies['Flip (fine)']  = flip_df['kappa_flip']

if flip_full_df is not None:
    b = flip_full_df[flip_full_df['sid'].isin(BIAS_SIDS)]
    recall_strategies['Flip (full)'] = b['right_recall_flip'] - b['left_recall_flip']
    kappa_strategies['Flip (full)']  = flip_full_df['kappa_flip']

colors = ['#4C72B0', '#DD8452', '#8172B2', '#55A868', '#937860', '#C44E52']
n = max(len(recall_strategies), len(kappa_strategies))
pal = colors[:n]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 왼쪽: recall_diff boxplot (bias 피험자 9명)
ax = axes[0]
if recall_strategies:
    bp = ax.boxplot(list(recall_strategies.values()), patch_artist=True)
    for patch, color in zip(bp['boxes'], pal):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.axhline(0.30, color='red', linestyle='--', linewidth=1.5, label='Bias threshold (0.30)')
    ax.axhline(0.0,  color='gray', linestyle=':', linewidth=1)
    ax.set_xticks(range(1, len(recall_strategies)+1))
    ax.set_xticklabels(list(recall_strategies.keys()), rotation=15, ha='right')
    ax.set_ylabel('right_recall - left_recall')
    ax.set_title('Bias 피험자 9명: recall_diff 분포')
    ax.legend(); ax.grid(axis='y', alpha=0.4)

# 오른쪽: kappa 비교 (전체 52명)
ax2 = axes[1]
if kappa_strategies:
    means = [v.mean() for v in kappa_strategies.values()]
    stds  = [v.std()  for v in kappa_strategies.values()]
    x     = np.arange(len(kappa_strategies))
    bars  = ax2.bar(x, means, yerr=stds, capsize=5, color=pal[:len(kappa_strategies)], alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(kappa_strategies.keys()), rotation=15, ha='right')
    ax2.set_ylabel("Cohen's κ")
    ax2.set_title('전체 52명: Kappa 비교')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.4)
    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, mean + 0.02,
                 f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

fig.suptitle('Right MI Bias Fix 전략 비교', fontsize=14, fontweight='bold')
plt.tight_layout()
out_path = os.path.join(cal_dir, 'bias_fix_comparison.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'저장: {out_path}')
"""))

# ════════════════════════════════════════════════════════════════
#  노트북 JSON 생성
# ════════════════════════════════════════════════════════════════

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
        "colab": {
            "provenance": [],
            "gpuType": "A100",
        },
        "accelerator": "GPU",
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"생성 완료: {OUT}")
print(f"셀 수: {len(cells)}")
