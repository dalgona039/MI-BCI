

import os
import sys
import time
import json
import argparse
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

try:
    from pythonosc import udp_client
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False

_SRC_DIR   = Path(__file__).resolve().parent          # src/
_ROOT_DIR  = _SRC_DIR.parent                          # MI-BCI/
_DATA_DIR  = _ROOT_DIR / "BCI_Research" / "preprocessed" / "member_A"
_CKPT_DIR  = _ROOT_DIR / "BCI_Research" / "results" / "checkpoints_A"

DEFAULT_CONFIG = {
    "member":        "A",
    "strategy":      "baseline_v4",
    "n_eeg_ch":      64,
    "n_emg_ch":      4,
    "n_times":       2304,
    "n_classes":     2,
    "emg_ds_factor": 8,
    "eegnet_F1":     8,
    "eegnet_D":      2,
    "eegnet_kern_len": 256,
    "eegnet_dropout":  0.5,
    "lstm_hidden":   128,
    "lstm_layers":   2,
    "lstm_dropout":  0.3,
    "clf_dropout":   0.3,
    "feat_dim":      256,
    "random_seed":   42,
}
DEFAULT_CONFIG["n_times_emg"] = DEFAULT_CONFIG["n_times"] // DEFAULT_CONFIG["emg_ds_factor"]

LABEL_NAMES = {0: "Left MI", 1: "Right MI"}


# ════════════════════════════════════════════════════════════════
#  모델 정의 (노트북의 STEP 3 과 동일)
# ════════════════════════════════════════════════════════════════

class EEGNetEncoder(nn.Module):
    def __init__(self, n_ch, n_times, F1=8, D=2,
                 kern_len=256, dropout=0.5, feat_dim=256):
        super().__init__()
        F2 = F1 * D
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F2, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        flat = self._get_flat_size(n_ch, n_times)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(flat, feat_dim), nn.ELU())

    def _get_flat_size(self, n_ch, n_times):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_ch, n_times)
            x = self.block3(self.block2(self.block1(x)))
            return x.numel()

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.fc(self.block3(self.block2(self.block1(x))))


class EMGBiLSTMEncoder(nn.Module):
    def __init__(self, n_ch=4, hidden=128, n_layers=2, dropout=0.3, feat_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_ch, hidden_size=hidden, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.fc   = nn.Sequential(nn.Linear(hidden * 2, feat_dim), nn.ELU())

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
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
        self.eeg_enc = EEGNetEncoder(
            n_ch=cfg["n_eeg_ch"], n_times=cfg["n_times"],
            F1=cfg["eegnet_F1"], D=cfg["eegnet_D"],
            kern_len=cfg["eegnet_kern_len"], dropout=cfg["eegnet_dropout"],
            feat_dim=fd,
        )
        self.emg_enc = EMGBiLSTMEncoder(
            n_ch=cfg["n_emg_ch"], hidden=cfg["lstm_hidden"],
            n_layers=cfg["lstm_layers"], dropout=cfg["lstm_dropout"],
            feat_dim=fd,
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
        fused, attn_w = self.fusion(h_eeg, h_emg)
        return self.clf(fused), attn_w


# ════════════════════════════════════════════════════════════════
#  SignalSimulator — HDF5 trial 재생기
# ════════════════════════════════════════════════════════════════

class SignalSimulator:
    """
    GigaDB 전처리 HDF5 파일에서 trial 을 순서대로(또는 랜덤으로) 재생.
    실제 전극 없이 시뮬레이션 신호를 공급.
    """

    def __init__(self, h5_path: str, cfg: dict, shuffle: bool = False):
        self.cfg     = cfg
        self.shuffle = shuffle
        ds = cfg.get("emg_ds_factor", 1)

        with h5py.File(h5_path, "r") as f:
            self.eeg = f["eeg/epochs"][:]                       # (N, 64, T)
            lbl      = f["labels"][:].astype(np.int64) - 1     # 1/2 → 0/1
            if "emg" in f and "epochs" in f["emg"]:
                emg = f["emg/epochs"][:]
            else:
                emg = np.zeros(
                    (self.eeg.shape[0], cfg["n_emg_ch"], cfg["n_times"]),
                    dtype=np.float32,
                )

        if ds > 1:
            emg = emg[:, :, ::ds]

        n = min(self.eeg.shape[0], emg.shape[0])
        self.eeg   = self.eeg[:n].astype(np.float32)
        self.emg   = emg[:n].astype(np.float32)
        self.lbl   = lbl[:n]
        self.n_trials = n

        self._order = np.arange(n)
        if shuffle:
            np.random.shuffle(self._order)
        self._idx = 0

        print(f"[Simulator] HDF5 로드 완료: {n}개 trial")
        print(f"  EEG shape: {self.eeg.shape}")
        print(f"  EMG shape: {self.emg.shape}")
        print(f"  Left MI: {(self.lbl==0).sum()}개  Right MI: {(self.lbl==1).sum()}개")

    def next_trial(self):
        """다음 trial (eeg, emg, true_label, trial_no) 반환. 마지막이면 None."""
        if self._idx >= self.n_trials:
            return None
        i = self._order[self._idx]
        self._idx += 1
        return (
            self.eeg[i],       # (64, 2304)
            self.emg[i],       # (4,  288)
            int(self.lbl[i]),  # 0 or 1
            self._idx,         # trial 번호 (1-based)
        )

    def reset(self):
        self._idx = 0
        if self.shuffle:
            np.random.shuffle(self._order)

    @property
    def remaining(self):
        return self.n_trials - self._idx


# ════════════════════════════════════════════════════════════════
#  BCIInferenceEngine — 모델 추론
# ════════════════════════════════════════════════════════════════

class BCIInferenceEngine:
    """학습된 HybridBCIModel checkpoint 를 로드하여 trial 단위 추론."""

    def __init__(self, ckpt_path: str, cfg: dict, device: str = "cpu"):
        self.cfg    = cfg
        self.device = device

        torch.manual_seed(cfg.get("random_seed", 42))
        self.model = HybridBCIModel(cfg).to(device)
        self.model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        self.model.eval()
        print(f"[Engine] 모델 로드: {ckpt_path}")
        print(f"  Device: {device}")

    @torch.no_grad()
    def predict(self, eeg: np.ndarray, emg: np.ndarray):
        """
        Parameters
        ----------
        eeg : (n_eeg_ch, n_times)
        emg : (n_emg_ch, n_times_emg)

        Returns
        -------
        pred_class  : int   0=Left MI, 1=Right MI
        confidence  : float softmax 최대값 (0~1)
        prob        : np.ndarray shape (n_classes,)
        latency_ms  : float 추론 소요시간 (ms)
        """
        t0 = time.perf_counter()

        eeg_t = torch.tensor(eeg, dtype=torch.float32).unsqueeze(0).to(self.device)
        emg_t = torch.tensor(emg, dtype=torch.float32).unsqueeze(0).to(self.device)

        logits, _ = self.model(eeg_t, emg_t)
        prob      = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred      = int(np.argmax(prob))
        conf      = float(prob[pred])
        latency   = (time.perf_counter() - t0) * 1000.0

        return pred, conf, prob, latency


# ════════════════════════════════════════════════════════════════
#  VRBridge — Unity OSC 전송
# ════════════════════════════════════════════════════════════════

class VRBridge:
    """
    Unity 측 OSC Receiver 로 BCI 명령을 전송.

    Unity에서 구독할 OSC 주소:
      /bci/prediction  (int)   0=Left MI, 1=Right MI
      /bci/confidence  (float) 0.0 ~ 1.0
      /bci/label       (str)   "Left MI" / "Right MI"
      /bci/trial_no    (int)   현재 trial 번호
      /bci/true_label  (int)   정답 (검증용)
    """

    def __init__(self, mode: str = "console",
                 osc_ip: str = "127.0.0.1", osc_port: int = 9000):
        self.mode = mode

        if mode == "osc":
            if not OSC_AVAILABLE:
                print("⚠️  python-osc 미설치 → console 모드로 전환")
                print("     pip install python-osc")
                self.mode = "console"
            else:
                self._client = udp_client.SimpleUDPClient(osc_ip, osc_port)
                print(f"[VRBridge] OSC 클라이언트 → {osc_ip}:{osc_port}")
        else:
            print("[VRBridge] Console 모드 (OSC 전송 없음)")

    def send(self, pred: int, conf: float, trial_no: int, true_label: int):
        label_str = LABEL_NAMES[pred]
        correct   = "✅" if pred == true_label else "❌"

        if self.mode == "osc":
            self._client.send_message("/bci/prediction", pred)
            self._client.send_message("/bci/confidence", conf)
            self._client.send_message("/bci/label",      label_str)
            self._client.send_message("/bci/trial_no",   trial_no)
            self._client.send_message("/bci/true_label", true_label)

        # 콘솔에는 항상 출력
        true_str = LABEL_NAMES[true_label]
        print(
            f"  Trial {trial_no:3d} | 예측: {label_str:<10} (conf={conf:.3f}) "
            f"| 정답: {true_str:<10} {correct}"
        )


# ════════════════════════════════════════════════════════════════
#  BCIVRDemo — 전체 파이프라인
# ════════════════════════════════════════════════════════════════

class BCIVRDemo:
    """
    SignalSimulator → BCIInferenceEngine → VRBridge 를 조합한 데모 루프.

    실행 흐름:
      1. simulator.next_trial() 로 (EEG, sEMG) 신호 획득
      2. engine.predict() 로 Left/Right MI 분류
      3. bridge.send() 로 Unity(OSC) 또는 콘솔에 결과 전송
      4. interval 초 대기 후 다음 trial
    """

    def __init__(
        self,
        sid: int,
        cfg: dict,
        ckpt_dir: str,
        data_dir: str,
        mode: str       = "console",
        osc_ip: str     = "127.0.0.1",
        osc_port: int   = 9000,
        interval: float = 4.0,
        shuffle: bool   = False,
        device: str     = "cpu",
        max_trials: int = None,
    ):
        self.interval   = interval
        self.max_trials = max_trials
        self._running   = False

        # ① 데이터 경로
        h5_path   = os.path.join(data_dir, f"sub-{sid:02d}_member_{cfg['member']}.h5")
        ckpt_path = os.path.join(ckpt_dir, f"best_s{sid:02d}.pt")

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 없음: {h5_path}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"체크포인트 없음: {ckpt_path}\n"
                f"  → STEP 6(run_loso) 완료 후 Drive에서 로컬로 복사 필요"
            )

        # ② 컴포넌트 초기화
        print("=" * 60)
        print(f"  BCI-VR Demo  |  Subject s{sid:02d}  |  Mode: {mode}")
        print("=" * 60)
        self.simulator = SignalSimulator(h5_path, cfg, shuffle=shuffle)
        self.engine    = BCIInferenceEngine(ckpt_path, cfg, device=device)
        self.bridge    = VRBridge(mode=mode, osc_ip=osc_ip, osc_port=osc_port)

    def run(self):
        """메인 루프: Ctrl+C 로 중단 가능."""
        self._running = True
        results = []

        print(f"\n▶ 시작 (trial 간격 {self.interval}s, Ctrl+C 로 중단)\n")
        print(f"  {'Trial':>5}  {'예측':<12} {'Conf':>6}  {'정답':<12}  {'지연':>8}")
        print("  " + "-" * 55)

        try:
            while self._running:
                item = self.simulator.next_trial()
                if item is None:
                    print("\n✅ 전체 trial 완료.")
                    break

                eeg, emg, true_lbl, trial_no = item

                # 추론
                pred, conf, prob, latency_ms = self.engine.predict(eeg, emg)

                # VR/콘솔 전송
                label_str = LABEL_NAMES[pred]
                true_str  = LABEL_NAMES[true_lbl]
                correct   = "✅" if pred == true_lbl else "❌"
                print(
                    f"  {trial_no:5d}  {label_str:<12} {conf:6.3f}  "
                    f"{true_str:<12}  {latency_ms:6.1f}ms  {correct}"
                )

                if self.bridge.mode == "osc":
                    self.bridge._client.send_message("/bci/prediction", pred)
                    self.bridge._client.send_message("/bci/confidence", conf)
                    self.bridge._client.send_message("/bci/label",      label_str)
                    self.bridge._client.send_message("/bci/trial_no",   trial_no)
                    self.bridge._client.send_message("/bci/true_label", true_lbl)

                results.append({
                    "trial":      trial_no,
                    "pred":       pred,
                    "true":       true_lbl,
                    "correct":    pred == true_lbl,
                    "confidence": conf,
                    "latency_ms": latency_ms,
                })

                if self.max_trials and trial_no >= self.max_trials:
                    print(f"\n✅ max_trials({self.max_trials}) 도달 — 종료")
                    break

                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\n⏹  사용자 중단")

        self._print_summary(results)
        return results

    def _print_summary(self, results: list):
        if not results:
            return
        n       = len(results)
        correct = sum(r["correct"] for r in results)
        acc     = correct / n
        avg_lat = np.mean([r["latency_ms"] for r in results])
        avg_conf = np.mean([r["confidence"] for r in results])

        print("\n" + "=" * 55)
        print("  데모 요약")
        print("=" * 55)
        print(f"  실행 trial  : {n}개")
        print(f"  정확도      : {correct}/{n}  ({acc:.1%})")
        print(f"  평균 신뢰도 : {avg_conf:.3f}")
        print(f"  평균 추론   : {avg_lat:.1f} ms")
        print("=" * 55)


# ════════════════════════════════════════════════════════════════
#  CLI 진입점
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="BCI-VR 시뮬레이션 데모")
    p.add_argument("--sid",       type=int,   default=3,
                   help="피험자 번호 (기본: 3, acc=0.97로 시각적으로 명확)")
    p.add_argument("--mode",      type=str,   default="console",
                   choices=["console", "osc"],
                   help="출력 모드 (console | osc)")
    p.add_argument("--osc_ip",    type=str,   default="127.0.0.1",
                   help="Unity OSC 수신 IP")
    p.add_argument("--osc_port",  type=int,   default=9000,
                   help="Unity OSC 수신 포트")
    p.add_argument("--interval",  type=float, default=4.0,
                   help="trial 간격 (초, 기본 4.0)")
    p.add_argument("--shuffle",   action="store_true",
                   help="trial 순서 랜덤화")
    p.add_argument("--max_trials",type=int,   default=None,
                   help="최대 trial 수 (기본: 전체)")
    p.add_argument("--ckpt_dir",  type=str,   default=str(_CKPT_DIR),
                   help="체크포인트 디렉터리")
    p.add_argument("--data_dir",  type=str,   default=str(_DATA_DIR),
                   help="HDF5 데이터 디렉터리")
    p.add_argument("--device",    type=str,   default="cpu",
                   choices=["cpu", "cuda"],
                   help="추론 장치")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = DEFAULT_CONFIG.copy()

    demo = BCIVRDemo(
        sid       = args.sid,
        cfg       = cfg,
        ckpt_dir  = args.ckpt_dir,
        data_dir  = args.data_dir,
        mode      = args.mode,
        osc_ip    = args.osc_ip,
        osc_port  = args.osc_port,
        interval  = args.interval,
        shuffle   = args.shuffle,
        device    = args.device,
        max_trials= args.max_trials,
    )
    demo.run()


if __name__ == "__main__":
    main()
