"""
export_onnx.py — HybridBCIModel ONNX 변환 스크립트
==========================================================
사용법:
  # 단일 피험자
  python export_onnx.py --sid 3

  # 전체 52명 일괄 변환
  python export_onnx.py --all

  # 출력 디렉터리 지정
  python export_onnx.py --all --out_dir ./onnx_models

출력:
  BCI_Research/results/onnx/
  ├── bci_s03.onnx           ← 피험자별 모델
  ├── bci_s03_metadata.json  ← 입출력 스펙 + 성능 수치
  └── ...

ONNX 입력:
  eeg  : float32 (1, 64, 2304)
  emg  : float32 (1,  4,  288)

ONNX 출력:
  logits : float32 (1, 2)   — CE loss 입력용 raw score
  probs  : float32 (1, 2)   — softmax 확률
  label  : int64   (1,)     — argmax 예측 클래스 (0=Left, 1=Right)
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 경로 기본값 ──────────────────────────────────────────────────
_SRC_DIR  = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
_CKPT_DIR = _ROOT_DIR / "BCI_Research" / "results" / "checkpoints_A"
_ONNX_DIR = _ROOT_DIR / "BCI_Research" / "results" / "onnx"

DEFAULT_CONFIG = {
    "member":          "A",
    "strategy":        "baseline_v4",
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
    "random_seed":     42,
}
DEFAULT_CONFIG["n_times_emg"] = DEFAULT_CONFIG["n_times"] // DEFAULT_CONFIG["emg_ds_factor"]

ALL_SIDS = list(range(1, 53))   # 1 ~ 52


# ════════════════════════════════════════════════════════════════
#  모델 정의 (inference.py 와 동일)
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
            return self.block3(self.block2(self.block1(x))).numel()

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
        return w[:, 0:1] * self.W_eeg(h_eeg) + w[:, 1:2] * self.W_emg(h_emg)


class HybridBCIModelONNX(nn.Module):

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

    def forward(self, eeg: torch.Tensor, emg: torch.Tensor):
        h_eeg  = self.eeg_enc(eeg)
        h_emg  = self.emg_enc(emg)
        fused  = self.fusion(h_eeg, h_emg)
        logits = self.clf(fused)
        probs  = F.softmax(logits, dim=-1)
        label  = torch.argmax(probs, dim=-1)
        return logits, probs, label


# ════════════════════════════════════════════════════════════════
#  ONNX 변환 함수
# ════════════════════════════════════════════════════════════════

def export_single(sid: int, cfg: dict, ckpt_dir: Path, out_dir: Path,
                  opset: int = 17, verify: bool = True) -> dict:
    """
    단일 피험자 모델을 ONNX 로 변환.

    Returns
    -------
    dict: 변환 결과 메타데이터
    """
    ckpt_path = ckpt_dir / f"best_s{sid:02d}.pt"
    if not ckpt_path.exists():
        print(f"  ⚠️  s{sid:02d}: 체크포인트 없음 → 스킵 ({ckpt_path})")
        return {"sid": sid, "status": "skipped", "reason": "no checkpoint"}

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"bci_s{sid:02d}.onnx"
    meta_path = out_dir / f"bci_s{sid:02d}_metadata.json"

    if onnx_path.exists():
        data_path = Path(str(onnx_path) + ".data")
        total_bytes = onnx_path.stat().st_size + (data_path.stat().st_size if data_path.exists() else 0)
        file_size_mb = total_bytes / (1024 ** 2)
        print(f"  ⏭️  s{sid:02d}: 이미 존재 ({file_size_mb:.1f} MB) → 스킵")
        return {"sid": sid, "status": "skipped", "reason": "already exists",
                "file_size_mb": file_size_mb, "latency_ms": None}

    model = HybridBCIModelONNX(cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)


    model.load_state_dict(state, strict=False)
    model.eval()

    eeg_dummy = torch.zeros(1, cfg["n_eeg_ch"], cfg["n_times"],     dtype=torch.float32)
    emg_dummy = torch.zeros(1, cfg["n_emg_ch"], cfg["n_times_emg"], dtype=torch.float32)

    torch.onnx.export(
        model,
        (eeg_dummy, emg_dummy),
        str(onnx_path),
        opset_version=opset,
        input_names=["eeg", "emg"],
        output_names=["logits", "probs", "label"],
        dynamic_axes={
            "eeg":    {0: "batch"},
            "emg":    {0: "batch"},
            "logits": {0: "batch"},
            "probs":  {0: "batch"},
            "label":  {0: "batch"},
        },
        do_constant_folding=True,
        export_params=True,
    )

    data_path = Path(str(onnx_path) + ".data")
    total_bytes = onnx_path.stat().st_size
    if data_path.exists():
        total_bytes += data_path.stat().st_size
    file_size_mb = total_bytes / (1024 ** 2)

    verify_status = "skipped"
    latency_ms    = None
    if verify:
        try:
            import onnxruntime as ort
            import time

            sess_opts = ort.SessionOptions()
            sess = ort.InferenceSession(
                str(onnx_path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )

            # warm-up
            feeds = {
                "eeg": eeg_dummy.numpy(),
                "emg": emg_dummy.numpy(),
            }
            for _ in range(3):
                sess.run(None, feeds)

            # latency 측정 (10회 평균)
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                out = sess.run(None, feeds)
                times.append((time.perf_counter() - t0) * 1000)

            latency_ms    = round(float(np.mean(times)), 2)
            pred_label    = int(out[2][0])
            verify_status = "ok"

            print(f"  ✅ s{sid:02d} → {onnx_path.name}  "
                  f"({file_size_mb:.1f} MB, latency={latency_ms:.1f}ms, "
                  f"pred={pred_label})")
        except ImportError:
            verify_status = "onnxruntime_not_installed"
            print(f"  ✅ s{sid:02d} → {onnx_path.name}  "
                  f"({file_size_mb:.1f} MB) [검증 스킵: pip install onnxruntime]")
        except Exception as e:
            verify_status = f"error: {e}"
            print(f"  ⚠️  s{sid:02d} 검증 실패: {e}")
    else:
        print(f"  ✅ s{sid:02d} → {onnx_path.name}  ({file_size_mb:.1f} MB)")

    meta = {
        "sid":          sid,
        "status":       "ok",
        "onnx_path":    str(onnx_path),
        "opset":        opset,
        "file_size_mb": round(file_size_mb, 2),
        "inputs": {
            "eeg":  [1, cfg["n_eeg_ch"], cfg["n_times"]],
            "emg":  [1, cfg["n_emg_ch"], cfg["n_times_emg"]],
        },
        "outputs": {
            "logits": [1, cfg["n_classes"]],
            "probs":  [1, cfg["n_classes"]],
            "label":  [1],
        },
        "label_map":    {0: "Left MI", 1: "Right MI"},
        "verify":       verify_status,
        "latency_ms":   latency_ms,
        "config":       cfg,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta



def parse_args():
    p = argparse.ArgumentParser(description="HybridBCIModel → ONNX 변환")
    p.add_argument("--sid",      type=int, default=None,
                   help="변환할 피험자 번호 (예: 3). --all 과 함께 사용 불가.")
    p.add_argument("--all",      action="store_true",
                   help="전체 52명 일괄 변환")
    p.add_argument("--ckpt_dir", type=str, default=str(_CKPT_DIR),
                   help="체크포인트 디렉터리")
    p.add_argument("--out_dir",  type=str, default=str(_ONNX_DIR),
                   help="ONNX 저장 디렉터리")
    p.add_argument("--opset",    type=int, default=17,
                   help="ONNX opset 버전 (기본: 17)")
    p.add_argument("--no_verify", action="store_true",
                   help="onnxruntime 검증 생략")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = DEFAULT_CONFIG.copy()

    ckpt_dir = Path(args.ckpt_dir)
    out_dir  = Path(args.out_dir)

    if not ckpt_dir.exists():
        print(f"❌ 체크포인트 디렉터리 없음: {ckpt_dir}")
        return

    if args.all:
        sids = ALL_SIDS
    elif args.sid is not None:
        sids = [args.sid]
    else:
        print("❌ --sid <번호> 또는 --all 을 지정하세요.")
        return

    print(f"ONNX 변환 시작 (opset={args.opset}, {len(sids)}명)")
    print(f"  체크포인트: {ckpt_dir}")
    print(f"  저장 경로:  {out_dir}\n")

    results = []
    for sid in sids:
        meta = export_single(
            sid=sid,
            cfg=cfg,
            ckpt_dir=ckpt_dir,
            out_dir=out_dir,
            opset=args.opset,
            verify=not args.no_verify,
        )
        results.append(meta)

    ok      = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    print(f"\n{'='*50}")
    print(f"  변환 완료: {len(ok)}/{len(sids)}명")
    if skipped:
        print(f"  스킵:     {[r['sid'] for r in skipped]}")
    if ok:
        sizes = [r["file_size_mb"] for r in ok]
        lats  = [r["latency_ms"] for r in ok if r["latency_ms"] is not None]
        print(f"  파일 크기: avg={sum(sizes)/len(sizes):.1f} MB")
        if lats:
            print(f"  추론 지연: avg={sum(lats)/len(lats):.1f} ms")
    print(f"  저장 위치: {out_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
