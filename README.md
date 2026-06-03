# Multimodal EEG-sEMG Motor Imagery BCI with Real-time VR Avatar Control

> **EEG + sEMG 융합 운동 상상 BCI → Meta Quest 3 VR 아바타 양팔 실시간 제어**

---

## Overview

본 연구는 EEG(뇌파)와 sEMG(표면 근전도)를 동시에 활용하는 **하이브리드 BCI(Brain-Computer Interface)** 시스템을 개발합니다. 운동 상상(Motor Imagery, MI) 신호만으로 Meta Quest 3 VR 환경에서 아바타의 양팔을 실시간으로 제어하는 것을 최종 목표로 합니다.

### Key Contributions

- **듀얼 스트림 딥러닝 모델**: EEGNet 기반 CNN(EEG) + 2층 BiLSTM(sEMG) 을 Softmax Attention Fusion으로 결합
- **전처리 절제 연구 (Ablation)**: 4가지 전처리 전략(baseline / wideband / narrowband / gamma)의 성능 비교
- **편향 분석 및 교정**: Right MI 과예측 편향을 Hemispheric Flip Augmentation 및 Logit Calibration으로 해소
- **실시간 BCI-VR 통합**: ONNX 모델 + WebSocket 서버 → Unity 아바타 양팔 IK 제어 (목표 지연 < 100 ms)
- **크로스 데이터셋 전이 학습**: GigaDB(52명)로 학습한 모델을 BCI Competition IV 2a(9명)에 zero-shot 평가

---

## Datasets

### GigaDB 100295 (주 데이터셋)
- Cho et al. 2017 | [gigadb.org/dataset/100295](http://gigadb.org/dataset/100295) | CC BY 4.0
- 피험자 52명 (여성 19명, 평균 24.8세)
- EEG 64ch + sEMG 4ch, 512 Hz, Biosemi ActiveTwo
- 파일: `s01.mat` ~ `s52.mat` (MATLAB v5)
- 레이블: 0 = Rest, 1 = Left MI, 2 = Right MI

### BCI Competition IV Dataset 2a (벤치마크)
- 피험자 9명, EEG 22ch, 250 Hz, `.gdf` 포맷
- GigaDB 모델의 cross-dataset 일반화 평가에만 사용

> **데이터 파일은 용량 문제로 저장소에 포함되지 않습니다.**  
> 위 링크에서 직접 다운로드 후 프로젝트 루트의 `GigaDB_100295/`, `BCICIV_2a_gdf/` 폴더에 배치하세요.

---

## Project Structure

```
MI-BCI/
├── src/                          # Python 소스 코드
│   ├── train_flip_full.py        # LOSO 전체 재학습 (Hemispheric Flip Augmentation)
│   ├── train_flip_aug.py         # Flip Augmentation 학습 (부분 피험자용)
│   ├── inference.py              # 슬라이딩 윈도우 실시간 추론 엔진
│   ├── export_onnx.py            # PyTorch → ONNX 변환 (opset 17)
│   ├── server_onnx.py            # ONNX 기반 WebSocket 서버 (Unity 연동)
│   ├── websocket_server.py       # PyTorch 기반 WebSocket 서버
│   ├── ablation_study.py         # 4가지 전처리 전략 성능 비교
│   ├── statistical_tests.py      # Wilcoxon signed-rank + ITR Bootstrap CI
│   ├── wilcoxon_analysis.py      # 추가 Wilcoxon 분석
│   ├── attention_analysis.py     # Softmax Attention 가중치 vs EMG SNR 분석
│   ├── bias_analysis.py          # Right MI 편향 분석
│   ├── bias_fix_report.py        # 편향 수정 결과 보고서 생성
│   ├── calibration.py            # Post-hoc Logit Calibration
│   ├── subgroup_analysis.py      # 피험자 서브그룹 분석
│   ├── transfer_bcic2a.py        # GigaDB → BCI IV 2a 전이 학습/평가
│   ├── latency_bench.py          # 추론 지연 시간 벤치마크
│   └── test_ws_client.py         # WebSocket 클라이언트 테스트
│
├── notebooks/                    # Colab 실행용 Jupyter 노트북
│   ├── S1_S2_Preprocessing_MemberA.ipynb  # 데이터 로딩 & 전처리 (Member A)
│   ├── S3_Model_Training_MemberA.ipynb    # 모델 학습 & LOSO CV
│   ├── S4_Ablation_Study_Colab.ipynb      # 전처리 절제 연구
│   ├── S5_Attention_Analysis_Colab.ipynb  # Attention 분석 & XAI
│   ├── S5_Bias_Fix_Colab.ipynb            # 편향 수정 실험
│   └── S6_Transfer_BCIC2a.ipynb           # 크로스 데이터셋 전이 평가
│
├── unity/                        # Unity C# 스크립트
│   ├── AvatarController.cs       # WebSocket 수신 → 아바타 IK 제어
│   ├── BCIExperimentManager.cs   # 실험 관리 (Trial 시작/종료, 이벤트)
│   └── BCISessionLogger.cs       # 세션 데이터 로깅
│
├── BCI-VR/                       # Unity 프로젝트 (Meta Quest 3)
│   ├── Assets/
│   ├── Packages/
│   └── ProjectSettings/
│
├── BCI_Research/
│   ├── preprocessed/             # 전처리된 HDF5 파일 (git 제외 — 대용량)
│   └── results/
│       ├── ablation/             # 절제 연구 결과 CSV/JSON/PNG ✓ (git 포함)
│       ├── attention/            # Attention 분석 결과 ✓ (git 포함)
│       ├── calibration/          # Calibration 결과 ✓ (git 포함)
│       ├── checkpoints_A/        # 모델 체크포인트 .pt (git 제외 — 212 MB)
│       ├── onnx/                 # ONNX 변환 파일 (git 제외 — 212 MB)
│       └── vr_sessions/          # VR 세션 로그 (git 제외)
│
└── generate_progress_report.py   # 진행 상황 보고서 자동 생성
```

---

## Model Architecture

```
EEG Input (64ch × 2304)          sEMG Input (4ch × 288)
       │                                  │
  ┌────▼──────┐                    ┌──────▼──────┐
  │  EEGNet   │                    │  BiLSTM ×2  │
  │  (CNN)    │                    │  hidden=128  │
  └────┬──────┘                    └──────┬──────┘
       │ h_EEG (256-dim)                  │ h_EMG (256-dim)
       └──────────────┬───────────────────┘
                      │
              ┌───────▼────────┐
              │ Softmax Attention│
              │  Fusion Layer   │
              │ w_EEG + w_EMG=1 │
              └───────┬────────┘
                      │ F_fused (256-dim)
              ┌───────▼────────┐
              │  Classifier    │
              │ 256→128→2      │
              │ ELU, Dropout   │
              └───────┬────────┘
                      │
               Left MI / Right MI
```

| 하이퍼파라미터 | 값 |
|---|---|
| EEGNet F1 / D | 8 / 2 |
| BiLSTM hidden | 128, 2 layers, bidirectional |
| Fusion | Softmax Attention (가중합, 합=1) |
| Classifier Dropout | 0.3 |
| Loss | Cross-Entropy + L2 (λ=1e-4) |
| Optimizer | Adam, lr=1e-3 |
| Batch size / Epochs | 32 / 200 (early stop patience=20) |
| CV | LOSO (Leave-One-Subject-Out, 52 subjects) |
| Monitor | val F1-macro |

---

## Preprocessing Strategies (Ablation)

4명의 팀원이 각각 다른 전처리 파라미터를 사용합니다.

| 파라미터 | A — baseline_v4 | B — wideband | C — narrowband | D — gamma |
|---|---|---|---|---|
| BPF | 4–40 Hz | 1–45 Hz | 8–30 Hz | 4–50 Hz |
| ICA components | 25 | 20 | 30 | 15 |
| epoch_tmin | −0.5 s | 0.0 s | −1.0 s | −0.5 s |
| epoch_tmax | 4.0 s | 4.0 s | 4.0 s | 3.0 s |
| baseline | (−0.5, 0.0) | None | (−1.0, 0.0) | (−0.5, 0.0) |
| EMG window | 50 ms | 100 ms | 25 ms | 200 ms |
| Normalization | z-score | min-max | robust | z-score |
| EOG threshold | r = 0.7 | r = 0.7 | r = 0.6 | r = 0.6 |

**공통**: random_seed=42, 동일 모델 아키텍처, LOSO CV

### EEG 전처리 순서
1. CAR (공통 평균 재참조)
2. Butterworth BPF 4차, zero-phase (`filtfilt`)
3. FastICA / MNE — EOG 아티팩트 제거 (C3/C4/Cz 보호)
4. 에포킹 (트리거 기반, 기준선 보정)
5. 적응형 PTP 기각: `threshold = median(ch_ptp) × 10`
6. 정규화 (z-score / min-max / robust)
7. μ대역(8–12 Hz), β대역(13–30 Hz) 추출

### sEMG 전처리 순서
1. 전파정류 (full-wave rectification)
2. 이동 RMS 엔벨로프
3. BPF 20–124 Hz, 노치 60/120 Hz
4. EEG와 동일 트리거로 에포킹
5. SNR 계산: $\text{SNR} = 20 \log_{10}(\text{RMS}_{MI} / \text{RMS}_{baseline})$ [dB]

---

## Pipeline Stages

```
S1  Data Loading & Sync
     └─ GigaDB .mat 로드, EEG/sEMG 분리, HDF5 캐싱

S2  Preprocessing Ablation  (4 members × 다른 파라미터)
     └─ BCI_Research/preprocessed/member_{A~D}/sub-XX.h5

S3  Model Training & LOSO CV
     └─ BCI_Research/results/checkpoints_A/best_sXX.pt

S4  XAI Analysis
     └─ DeepSHAP, Grad-CAM, ERD(%) 검증
     └─ BCI_Research/results/attention/

S5  Bias Analysis & Fix
     └─ Right MI 과예측 편향 분석
     └─ Hemispheric Flip Augmentation 전체 재학습
     └─ Post-hoc Logit Calibration

S6  Cross-dataset Transfer
     └─ GigaDB → BCI IV 2a zero-shot 평가
     └─ BCI_Research/results/transfer_bcic2a/

S7  Real-time BCI-VR Integration
     └─ ONNX 변환 → WebSocket 서버 → Unity
```

---

## Real-time BCI-VR System

```
Python Inference Server                Unity (Meta Quest 3)
┌─────────────────────────┐            ┌──────────────────────────┐
│  ONNX Model (bci_sXX)   │            │  AvatarController.cs     │
│  Sliding Window          │            │  ├─ NativeWebSocket      │
│  (2048 samples, s=128)   │  WebSocket │  ├─ Lerp + EMA (α=0.3)  │
│  EMA smoothing (α=0.3)   │ ─────────▶│  └─ IK Target Control    │
│  min_confidence=0.6      │   JSON     │                          │
└─────────────────────────┘            └──────────────────────────┘
  Latency target: < 100 ms
```

**실행 방법 (실시간 서버)**:
```bash
# ONNX 기반 서버 (권장)
python src/server_onnx.py --sid 3 --wait_client --min_confidence 0.6

# PyTorch 기반 서버
python src/websocket_server.py --sid 3 --port 8765

# Unity: WebSocket URL = ws://<PC-IP>:8765
```

**ONNX 입출력 스펙**:
| 텐서 | shape | dtype |
|---|---|---|
| `eeg` (입력) | (1, 64, 2304) | float32 |
| `emg` (입력) | (1, 4, 288) | float32 |
| `logits` (출력) | (1, 2) | float32 |
| `probs` (출력) | (1, 2) | float32 |
| `label` (출력) | (1,) | int64 |

---

## Evaluation Metrics

| 지표 | 설명 |
|---|---|
| F1-macro | Left / Right 클래스 균형 평가 |
| Cohen's κ | 우연 수준 보정 일치도 |
| ITR | 정보 전달률 (bits/min) |
| Wilcoxon | 조건 간 비모수 쌍별 검정 |
| Bonferroni | 다중 비교 보정 (6쌍, α=0.05/6) |

---

## Installation

### Python 환경

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# 의존성 설치
pip install torch torchvision torchaudio
pip install numpy scipy pandas h5py scikit-learn
pip install mne onnx onnxruntime websockets
```

### Unity 설정 (BCI-VR)

1. Unity 2022.3 LTS 이상 + Meta XR SDK 설치
2. Package Manager → **Add package from git URL**:  
   `https://github.com/endel/NativeWebSocket.git#upm`
3. `unity/AvatarController.cs`, `unity/BCIExperimentManager.cs`, `unity/BCISessionLogger.cs` 를 `BCI-VR/Assets/Scripts/` 에 복사
4. Scene의 Avatar GameObject에 `AvatarController` 컴포넌트 추가
5. `leftArmTarget` / `rightArmTarget` IK Target Transform 연결
6. `serverUrl` = `ws://<Python서버IP>:8765` 설정

---

## Quick Start

### 1. 데이터 전처리 (Colab 권장)
```python
# notebooks/S1_S2_Preprocessing_MemberA.ipynb 실행
# 출력: BCI_Research/preprocessed/member_A/sub-XX_member_A.h5
```

### 2. 모델 학습
```python
# notebooks/S3_Model_Training_MemberA.ipynb 실행
# 또는
python src/train_flip_full.py --drive_root /content/drive/MyDrive/MI-BCI
# 출력: BCI_Research/results/checkpoints_A/best_sXX.pt
```

### 3. ONNX 변환
```bash
python src/export_onnx.py --all --out_dir BCI_Research/results/onnx
```

> **참고**: `.pt` 체크포인트(212 MB)와 `.onnx` 파일(212 MB)은 크기 문제로 저장소에 포함되지 않습니다.  
> 학습 완료 후 로컬에 생성되며, Google Drive 등 외부 스토리지로 공유하세요.

### 4. 통계 검정
```bash
python src/statistical_tests.py
# 출력: results/ablation/wilcoxon_results.json
#       results/ablation/itr_bootstrap.json
```

### 5. 실시간 VR 데모
```bash
# Python 서버 시작
python src/server_onnx.py --sid 3 --wait_client

# Unity에서 Play → 아바타가 BCI 신호로 팔 움직임
```

---

## HDF5 File Format

```
sub-01_member_A.h5
├── eeg/
│   ├── epochs        (n_epochs, 64, n_times)   # 원시 EEG 에포크
│   ├── mu_epochs     (n_epochs, 64, n_times)   # 8–12 Hz 필터링
│   └── beta_epochs   (n_epochs, 64, n_times)   # 13–30 Hz 필터링
├── emg/
│   └── epochs        (n_epochs, 4, n_times)    # sEMG 에포크
├── labels            (n_epochs,)               # 0=Left, 1=Right
└── metadata/                                   # attrs: CONFIG 전체 저장
```

---

## Team Roles

| 멤버 | 담당 |
|---|---|
| **A (본 저장소)** | 데이터 로딩, EEG 전처리, LOSO 학습 루프, 파이프라인 기준 코드 |
| B | sEMG 전처리, EEGNet CNN + BiLSTM + Attention Fusion 모델 설계 |
| C | XAI (DeepSHAP, Grad-CAM, ERD 검증), F1/κ/ITR 평가, Wilcoxon, 논문 Results |
| D | TorchScript/ONNX 변환, WebSocket 서버, Unity 아바타 제어, GitHub 관리 |

---

## Citation

```
[Dataset] Cho et al. (2017). EEG datasets for motor imagery brain–computer interface.
GigaScience, 6(7). https://doi.org/10.1093/gigascience/gix034
```

---

## License

본 연구 코드는 연구 목적으로만 사용 가능합니다.  
GigaDB 데이터셋은 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 라이선스를 따릅니다.
