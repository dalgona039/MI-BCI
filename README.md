# Multimodal EEG-sEMG Motor Imagery BCI with Real-time VR Avatar Control

> **EEG + sEMG Fusion Motor Imagery BCI → Real-time Bilateral VR Avatar Arm Control on Meta Quest 3**

---

## Overview

This study develops a **hybrid BCI (Brain-Computer Interface)** system that simultaneously utilizes EEG (electroencephalography) and sEMG (surface electromyography).  
Using only motor imagery (MI) signals, the system classifies left/right intent and controls both arms of a VR avatar in real time.

### Key Contributions

- **Dual-stream deep learning model**: EEGNet-based CNN (EEG) + 2-layer BiLSTM (sEMG), fused with Softmax Attention Fusion
- **Preprocessing ablation study**: Performance comparison across four preprocessing strategies (baseline / wideband / narrowband / gamma)
- **Bias analysis and correction**: Mitigation of Right MI overprediction bias using Hemispheric Flip Augmentation and Logit Calibration
- **Real-time BCI-VR integration**: ONNX model + WebSocket server → bilateral avatar IK arm control in Unity (target latency < 100 ms)
- **Cross-dataset transfer learning**: Zero-shot evaluation on BCI Competition IV 2a (9 subjects) using a model trained on GigaDB (52 subjects)

---

## Datasets

### GigaDB 100295 (Main dataset)
- Cho et al. 2017 | [gigadb.org/dataset/100295](http://gigadb.org/dataset/100295) | CC BY 4.0
- 52 subjects (19 female, mean age 24.8)
- EEG 64ch + sEMG 4ch, 512 Hz, Biosemi ActiveTwo
- Files: `s01.mat` ~ `s52.mat` (MATLAB v5)
- Labels: 0 = Rest, 1 = Left MI, 2 = Right MI

### BCI Competition IV Dataset 2a (Benchmark)
- 9 subjects, EEG 22ch, 250 Hz, `.gdf` format
- Used only for cross-dataset generalization evaluation of the GigaDB-trained model

> **Dataset files are not included in this repository due to file size limitations.**  
> Please download them directly from the links above and place them under `GigaDB_100295/` and `BCICIV_2a_gdf/` in the project root.

---

## Project Structure

```text
MI-BCI/
├── src/                          # Python source code
│   ├── train_flip_full.py        # Full LOSO retraining (Hemispheric Flip Augmentation)
│   ├── train_flip_aug.py         # Flip Augmentation training (subset subjects)
│   ├── inference.py              # Sliding-window real-time inference engine
│   ├── export_onnx.py            # PyTorch → ONNX conversion (opset 17)
│   ├── server_onnx.py            # ONNX-based WebSocket server (Unity integration)
│   ├── websocket_server.py       # PyTorch-based WebSocket server
│   ├── ablation_study.py         # Performance comparison of 4 preprocessing strategies
│   ├── statistical_tests.py      # Wilcoxon signed-rank + ITR bootstrap CI
│   ├── wilcoxon_analysis.py      # Additional Wilcoxon analysis
│   ├── attention_analysis.py     # Softmax Attention weights vs EMG SNR analysis
│   ├── bias_analysis.py          # Right MI bias analysis
│   ├── bias_fix_report.py        # Bias correction result report generator
│   ├── calibration.py            # Post-hoc Logit Calibration
│   ├── subgroup_analysis.py      # Subject subgroup analysis
│   ├── transfer_bcic2a.py        # GigaDB → BCI IV 2a transfer learning/evaluation
│   ├── latency_bench.py          # Inference latency benchmark
│   └── test_ws_client.py         # WebSocket client test
│
├── notebooks/                    # Jupyter notebooks for Colab execution
│   ├── S1_S2_Preprocessing_MemberA.ipynb  # Data loading & preprocessing (Member A)
│   ├── S3_Model_Training_MemberA.ipynb    # Model training & LOSO CV
│   ├── S4_Ablation_Study_Colab.ipynb      # Preprocessing ablation study
│   ├── S5_Attention_Analysis_Colab.ipynb  # Attention analysis & XAI
│   ├── S5_Bias_Fix_Colab.ipynb            # Bias correction experiment
│   └── S6_Transfer_BCIC2a.ipynb           # Cross-dataset transfer evaluation
│
├── unity/                        # Unity C# scripts
│   ├── AvatarController.cs       # WebSocket input → avatar IK control
│   ├── BCIExperimentManager.cs   # Experiment manager (trial start/end, events)
│   └── BCISessionLogger.cs       # Session data logging
│
├── BCI-VR/                       # Unity project (Meta Quest 3)
│   ├── Assets/
│   ├── Packages/
│   └── ProjectSettings/
│
├── BCI_Research/
│   ├── preprocessed/             # Preprocessed HDF5 files (excluded from git — large)
│   └── results/
│       ├── ablation/             # Ablation results CSV/JSON/PNG ✓ (tracked in git)
│       ├── attention/            # Attention analysis results ✓ (tracked in git)
│       ├── calibration/          # Calibration results ✓ (tracked in git)
│       ├── checkpoints_A/        # Model checkpoints .pt (excluded from git — 212 MB)
│       ├── onnx/                 # ONNX files (excluded from git — 212 MB)
│       └── vr_sessions/          # VR session logs (excluded from git)
│
└── generate_progress_report.py   # Auto-generates project progress report
```

---

## Model Architecture

```text
EEG Input (64ch × 2304)          sEMG Input (4ch × 288)
       │                                  │
  ┌────▼──────┐                    ┌──────▼──────┐
  │  EEGNet   │                    │  BiLSTM ×2  │
  │  (CNN)    │                    │  hidden=128 │
  └────┬──────┘                    └──────┬──────┘
       │ h_EEG (256-dim)                  │ h_EMG (256-dim)
       └──────────────┬───────────────────┘
                      │
              ┌───────▼─────────┐
              │ Softmax Attention│
              │  Fusion Layer    │
              │ w_EEG + w_EMG=1  │
              └───────┬─────────┘
                      │ F_fused (256-dim)
              ┌───────▼─────────┐
              │   Classifier    │
              │   256→128→2     │
              │ ELU, Dropout    │
              └───────┬─────────┘
                      │
               Left MI / Right MI
```

| Hyperparameter | Value |
|---|---|
| EEGNet F1 / D | 8 / 2 |
| BiLSTM hidden | 128, 2 layers, bidirectional |
| Fusion | Softmax Attention (weighted sum, sum=1) |
| Classifier Dropout | 0.3 |
| Loss | Cross-Entropy + L2 (λ=1e-4) |
| Optimizer | Adam, lr=1e-3 |
| Batch size / Epochs | 32 / 200 (early stop patience=20) |
| CV | LOSO (Leave-One-Subject-Out, 52 subjects) |
| Monitor | val F1-macro |

---

## Preprocessing Strategies (Ablation)

Four team members each use different preprocessing parameters.

| Parameter | A — baseline_v4 | B — wideband | C — narrowband | D — gamma |
|---|---|---|---|---|
| BPF | 4–40 Hz | 1–45 Hz | 8–30 Hz | 4–50 Hz |
| ICA components | 25 | 20 | 30 | 15 |
| epoch_tmin | −0.5 s | 0.0 s | −1.0 s | −0.5 s |
| epoch_tmax | 4.0 s | 4.0 s | 4.0 s | 3.0 s |
| baseline | (−0.5, 0.0) | None | (−1.0, 0.0) | (−0.5, 0.0) |
| EMG window | 50 ms | 100 ms | 25 ms | 200 ms |
| Normalization | z-score | min-max | robust | z-score |
| EOG threshold | r = 0.7 | r = 0.7 | r = 0.6 | r = 0.6 |

**Common settings**: random_seed=42, identical model architecture, LOSO CV

### EEG Preprocessing Pipeline
1. CAR (Common Average Re-reference)
2. 4th-order Butterworth BPF, zero-phase (`filtfilt`)
3. FastICA / MNE — EOG artifact removal (protect C3/C4/Cz)
4. Epoching (trigger-based, baseline correction)
5. Adaptive PTP rejection: `threshold = median(ch_ptp) × 10`
6. Normalization (z-score / min-max / robust)
7. μ-band (8–12 Hz), β-band (13–30 Hz) extraction

### sEMG Preprocessing Pipeline
1. Full-wave rectification
2. Moving RMS envelope
3. BPF 20–124 Hz, notch 60/120 Hz
4. Epoching with the same triggers as EEG
5. SNR calculation: $\text{SNR} = 20 \log_{10}(\text{RMS}_{MI} / \text{RMS}_{baseline})$ [dB]

---

## Pipeline Stages

```text
S1  Data Loading & Sync
     └─ Load GigaDB .mat files, split EEG/sEMG, cache as HDF5

S2  Preprocessing Ablation  (4 members × different parameters)
     └─ BCI_Research/preprocessed/member_{A~D}/sub-XX.h5

S3  Model Training & LOSO CV
     └─ BCI_Research/results/checkpoints_A/best_sXX.pt

S4  XAI Analysis
     └─ DeepSHAP, Grad-CAM, ERD(%) validation
     └─ BCI_Research/results/attention/

S5  Bias Analysis & Fix
     └─ Right MI overprediction bias analysis
     └─ Full retraining with Hemispheric Flip Augmentation
     └─ Post-hoc Logit Calibration

S6  Cross-dataset Transfer
     └─ GigaDB → BCI IV 2a zero-shot evaluation
     └─ BCI_Research/results/transfer_bcic2a/

S7  Real-time BCI-VR Integration
     └─ ONNX conversion → WebSocket server → Unity
```

---

## Real-time BCI-VR System

```text
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

**How to run (real-time server)**:
```bash
# ONNX-based server (recommended)
python src/server_onnx.py --sid 3 --wait_client --min_confidence 0.6

# PyTorch-based server
python src/websocket_server.py --sid 3 --port 8765

# Unity: WebSocket URL = ws://<PC-IP>:8765
```

**ONNX I/O spec**:

| Tensor | shape | dtype |
|---|---|---|
| `eeg` (input) | (1, 64, 2304) | float32 |
| `emg` (input) | (1, 4, 288) | float32 |
| `logits` (output) | (1, 2) | float32 |
| `probs` (output) | (1, 2) | float32 |
| `label` (output) | (1,) | int64 |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| F1-macro | Balanced evaluation across Left / Right classes |
| Cohen's κ | Chance-corrected agreement |
| ITR | Information Transfer Rate (bits/min) |
| Wilcoxon | Non-parametric paired test between conditions |
| Bonferroni | Multiple-comparison correction (6 pairs, α=0.05/6) |

---

## Installation

### Python Environment

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy scipy pandas h5py scikit-learn
pip install mne onnx onnxruntime websockets
```

### Unity Setup (BCI-VR)

1. Install Unity 2022.3 LTS or later + Meta XR SDK
2. Package Manager → **Add package from git URL**:  
   `https://github.com/endel/NativeWebSocket.git#upm`
3. Copy `unity/AvatarController.cs`, `unity/BCIExperimentManager.cs`, `unity/BCISessionLogger.cs` into `BCI-VR/Assets/Scripts/`
4. Add `AvatarController` component to the avatar GameObject in your scene
5. Assign IK target transforms for `leftArmTarget` / `rightArmTarget`
6. Set `serverUrl` = `ws://<Python-server-IP>:8765`

---

## Quick Start

### 1. Data preprocessing (Colab recommended)
```python
# Run notebooks/S1_S2_Preprocessing_MemberA.ipynb
# Output: BCI_Research/preprocessed/member_A/sub-XX_member_A.h5
```

### 2. Model training
```python
# Run notebooks/S3_Model_Training_MemberA.ipynb
# or
python src/train_flip_full.py --drive_root /content/drive/MyDrive/MI-BCI
# Output: BCI_Research/results/checkpoints_A/best_sXX.pt
```

### 3. ONNX export
```bash
python src/export_onnx.py --all --out_dir BCI_Research/results/onnx
```

> **Note**: `.pt` checkpoints (212 MB) and `.onnx` files (212 MB) are not included in the repository due to size limits.  
> They are generated locally after training; share them via external storage such as Google Drive.

### 4. Statistical testing
```bash
python src/statistical_tests.py
# Output: results/ablation/wilcoxon_results.json
#         results/ablation/itr_bootstrap.json
```

### 5. Real-time VR demo
```bash
# Start Python server
python src/server_onnx.py --sid 3 --wait_client

# Press Play in Unity → avatar arms move based on BCI signals
```

---

## HDF5 File Format

```text
sub-01_member_A.h5
├── eeg/
│   ├── epochs        (n_epochs, 64, n_times)   # Raw EEG epochs
│   ├── mu_epochs     (n_epochs, 64, n_times)   # 8–12 Hz filtered
│   └── beta_epochs   (n_epochs, 64, n_times)   # 13–30 Hz filtered
├── emg/
│   └── epochs        (n_epochs, 4, n_times)    # sEMG epochs
├── labels            (n_epochs,)               # 0=Left, 1=Right
└── metadata/                                   # attrs: full CONFIG stored
```

---


## Citation

```text
[Dataset] Cho et al. (2017). EEG datasets for motor imagery brain–computer interface.
GigaScience, 6(7). https://doi.org/10.1093/gigascience/gix034
```

---

## License

This research code is intended for research purposes only.  
The GigaDB dataset follows the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
