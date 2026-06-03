"""
server_onnx.py — ONNX 기반 BCI WebSocket 서버 (로컬 실행용)
=============================================================
PyTorch 없이 onnxruntime 으로 추론. 로컬 .venv에서 바로 실행 가능.

의존성 (모두 .venv에 설치됨):
  websockets, onnxruntime, numpy, h5py

실행:
  python src/server_onnx.py --sid 3 --wait_client

Meta Quest 3 연결:
  python src/server_onnx.py --sid 3 --wait_client --host 0.0.0.0
  → Unity serverUrl = "ws://<Mac LAN IP>:8765"
  (Mac LAN IP: 시스템 환경설정 → 네트워크, 예: 192.168.0.5)

변경점 vs websocket_server.py:
  - torch → onnxruntime (로컬 실행 가능)
  - trial_start 메시지에 true_label / true_label_str 포함 (VR 큐 표시용)
  - --cue_duration: 큐 표시 후 실제 추론까지 대기 시간 (기본 2.0s)
"""

import os
import sys
import csv
import json
import asyncio
import argparse
import time
import socket
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py
import onnxruntime as ort

try:
    import websockets
    from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
except ImportError:
    print("❌ websockets 미설치: pip install websockets")
    sys.exit(1)

_SRC_DIR  = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
_DATA_DIR = _ROOT_DIR / "BCI_Research" / "preprocessed" / "member_A"
_ONNX_DIR = _ROOT_DIR / "BCI_Research" / "results" / "onnx"
_LOG_DIR  = _ROOT_DIR / "BCI_Research" / "results" / "vr_sessions"

LABEL_NAMES = {0: "Left MI", 1: "Right MI"}

# CSV 컬럼 정의 (논문 Table용)
_CSV_FIELDS = [
    "session_id", "sid", "trial_no", "timestamp",
    "true_label", "true_label_str",
    "pred_label", "pred_label_str",
    "correct", "skipped",
    "confidence", "prob_left", "prob_right",
    "inference_latency_ms", "e2e_latency_ms",
]


# ════════════════════════════════════════════════════════════════
#  SignalSimulator (h5py, no torch)
# ════════════════════════════════════════════════════════════════

class SignalSimulator:
    """HDF5 trial 재생기 (onnxruntime 버전, torch 불필요)."""

    def __init__(self, h5_path: str, emg_ds_factor: int = 8, shuffle: bool = False):
        self.shuffle = shuffle

        with h5py.File(h5_path, "r") as f:
            eeg = f["eeg/epochs"][:].astype(np.float32)          # (N, 64, 2304)
            lbl = f["labels"][:].astype(np.int64) - 1            # 1/2 → 0/1

            if "emg" in f and "epochs" in f["emg"]:
                emg = f["emg/epochs"][:].astype(np.float32)
            else:
                emg = np.zeros(
                    (eeg.shape[0], 4, eeg.shape[2]), dtype=np.float32
                )

        if emg_ds_factor > 1:
            emg = emg[:, :, ::emg_ds_factor]                      # → (N, 4, 288)

        n = min(eeg.shape[0], emg.shape[0], lbl.shape[0])
        self.eeg      = eeg[:n]
        self.emg      = emg[:n]
        self.lbl      = lbl[:n]
        self.n_trials = n

        self._order = np.arange(n)
        if shuffle:
            np.random.shuffle(self._order)
        self._idx = 0

        print(f"[Simulator] {n}개 trial 로드")
        print(f"  EEG {self.eeg.shape}  EMG {self.emg.shape}")
        print(f"  Left MI: {(self.lbl==0).sum()}  Right MI: {(self.lbl==1).sum()}")

    def next_trial(self):
        if self._idx >= self.n_trials:
            return None
        i = self._order[self._idx]
        self._idx += 1
        return self.eeg[i], self.emg[i], int(self.lbl[i]), self._idx

    def reset(self):
        self._idx = 0
        if self.shuffle:
            np.random.shuffle(self._order)

    @property
    def remaining(self):
        return self.n_trials - self._idx


# ════════════════════════════════════════════════════════════════
#  BCIInferenceEngineONNX
# ════════════════════════════════════════════════════════════════

class BCIInferenceEngineONNX:
    """onnxruntime 기반 추론 엔진."""

    def __init__(self, onnx_path: str):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._in_names  = [i.name for i in self.session.get_inputs()]
        self._out_names = [o.name for o in self.session.get_outputs()]

        print(f"[Engine] ONNX 로드: {Path(onnx_path).name}")
        print(f"  inputs : {self._in_names}")
        print(f"  outputs: {self._out_names}")

        # 워밍업 (첫 추론 JIT 지연 제거)
        dummy_eeg = np.zeros((64, 2304), dtype=np.float32)
        dummy_emg = np.zeros((4, 288),   dtype=np.float32)
        self.predict(dummy_eeg, dummy_emg)
        print("  워밍업 완료")

    def predict(self, eeg: np.ndarray, emg: np.ndarray):
        """
        Returns
        -------
        pred_class : int   0=Left MI, 1=Right MI
        confidence : float
        prob       : np.ndarray (2,)
        latency_ms : float
        """
        t0 = time.perf_counter()

        feeds = {
            "eeg": eeg[np.newaxis].astype(np.float32),  # (1, 64, 2304)
            "emg": emg[np.newaxis].astype(np.float32),  # (1, 4, 288)
        }
        outputs = self.session.run(self._out_names, feeds)

        # outputs 순서: logits, probs, label (metadata 기준)
        # probs 인덱스 찾기
        if "probs" in self._out_names:
            prob_idx = self._out_names.index("probs")
            prob = outputs[prob_idx][0]                 # (2,)
        else:
            # logits → softmax 직접 계산
            logit_idx = self._out_names.index("logits")
            logits = outputs[logit_idx][0]
            e = np.exp(logits - logits.max())
            prob = e / e.sum()

        pred = int(np.argmax(prob))
        conf = float(prob[pred])
        latency = (time.perf_counter() - t0) * 1000.0

        return pred, conf, prob, latency


# ════════════════════════════════════════════════════════════════
#  BCIWebSocketServer
# ════════════════════════════════════════════════════════════════

class BCIWebSocketServer:
    """
    asyncio WebSocket 서버 (ONNX 버전).

    trial_start 메시지에 true_label 포함 → Unity에서 큐 표시 가능.
    cue_duration 초 뒤에 추론 → 참가자가 MI를 수행할 시간 제공.
    """

    def __init__(
        self,
        sid:            int,
        host:           str   = "0.0.0.0",
        port:           int   = 8765,
        interval:       float = 4.0,
        cue_duration:   float = 2.0,
        shuffle:        bool  = False,
        max_trials:     int   = None,
        wait_client:    bool  = False,
        min_confidence: float = 0.6,
        data_dir:       str   = None,
        onnx_dir:       str   = None,
        log_dir:        str   = None,
        session_name:   str   = None,
    ):
        self.sid            = sid
        self.host           = host
        self.port           = port
        self.interval       = interval
        self.cue_duration   = cue_duration
        self.max_trials     = max_trials
        self.wait_client    = wait_client
        self.min_confidence = min_confidence
        self._paused        = False
        self._clients: set  = set()

        # 로그 설정
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id  = session_name or f"s{sid:02d}_{ts}"
        self.log_dir     = Path(log_dir) if log_dir else _LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # end-to-end latency 측정용: {trial_no: send_timestamp}
        self._pred_send_ts: dict = {}
        self._e2e_latencies: dict = {}  # {trial_no: e2e_ms}

        data_dir = data_dir or str(_DATA_DIR)
        onnx_dir = onnx_dir or str(_ONNX_DIR)

        h5_path   = os.path.join(data_dir, f"sub-{sid:02d}_member_A.h5")
        onnx_path = os.path.join(onnx_dir, f"bci_s{sid:02d}.onnx")

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 없음: {h5_path}")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX 없음: {onnx_path}")

        print("=" * 60)
        print(f"  BCI ONNX WebSocket Server  |  Subject s{sid:02d}")
        print(f"  ws://{host}:{port}")
        print(f"  cue_duration={cue_duration}s  interval={interval}s")
        print("=" * 60)

        self.simulator = SignalSimulator(h5_path, shuffle=shuffle)
        self.engine    = BCIInferenceEngineONNX(onnx_path)

        print(f"\n  max_trials={max_trials or 'all'}  wait_client={wait_client}\n")

    # ── 클라이언트 핸들러 ─────────────────────────────────────────

    async def _handler(self, websocket):
        addr = websocket.remote_address
        self._clients.add(websocket)
        print(f"[+] {addr}  (총 {len(self._clients)}개)")

        await self._send_one(websocket, {
            "type":       "connected",
            "sid":        self.sid,
            "n_trials":   self.simulator.n_trials,
            "interval_s": self.interval,
            "cue_duration_s": self.cue_duration,
            "message":    f"BCI ONNX Server — Subject s{self.sid:02d} ready",
        })

        try:
            async for raw in websocket:
                await self._handle_cmd(websocket, raw)
        except (ConnectionClosedOK, ConnectionClosedError):
            pass
        finally:
            self._clients.discard(websocket)
            print(f"[-] {addr}  (총 {len(self._clients)}개)")

    async def _handle_cmd(self, websocket, raw: str):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        cmd = msg.get("cmd", "")
        if cmd == "reset":
            self.simulator.reset()
            self._paused = False
            await self._send_one(websocket, {"type": "ack", "cmd": "reset"})
            print("[Cmd] reset")
        elif cmd == "pause":
            self._paused = True
            await self._broadcast({"type": "ack", "cmd": "pause", "paused": True})
            print("[Cmd] pause")
        elif cmd == "resume":
            self._paused = False
            await self._broadcast({"type": "ack", "cmd": "resume", "paused": False})
            print("[Cmd] resume")
        elif cmd == "status":
            await self._send_one(websocket, {
                "type":      "status",
                "remaining": self.simulator.remaining,
                "n_clients": len(self._clients),
                "paused":    self._paused,
            })
        elif cmd == "latency_ack":
            # Unity가 prediction 수신 직후 전송하는 타임스탬프 ack
            # → end-to-end latency = unity_recv_ts - server_send_ts
            trial_no      = msg.get("trial_no", -1)
            unity_recv_ts = msg.get("unity_recv_ts", 0.0)
            send_ts       = self._pred_send_ts.get(trial_no)
            if send_ts and unity_recv_ts > 0:
                e2e_ms = (unity_recv_ts - send_ts) * 1000.0
                self._e2e_latencies[trial_no] = round(e2e_ms, 2)

    # ── 브로드캐스트 ─────────────────────────────────────────────

    async def _broadcast(self, msg_dict: dict):
        if not self._clients:
            return
        payload = json.dumps(msg_dict, ensure_ascii=False)
        dead    = set()
        results = await asyncio.gather(
            *[ws.send(payload) for ws in self._clients],
            return_exceptions=True,
        )
        for ws, r in zip(list(self._clients), results):
            if isinstance(r, Exception):
                dead.add(ws)
        self._clients -= dead

    async def _send_one(self, websocket, msg_dict: dict):
        try:
            await websocket.send(json.dumps(msg_dict, ensure_ascii=False))
        except Exception:
            pass

    # ── trial 루프 ───────────────────────────────────────────────

    async def _run_trials(self):
        if self.wait_client:
            print("[Trials] 클라이언트 연결 대기...")
            while not self._clients:
                await asyncio.sleep(0.5)
            print("[Trials] 클라이언트 감지 — 시작")
        else:
            await asyncio.sleep(0.5)
            print("[Trials] 클라이언트 없이 시작")

        results      = []
        trial_cnt    = 0
        session_start = time.time()

        while True:
            item = self.simulator.next_trial()
            if item is None:
                print("\n✅ 전체 trial 완료")
                break

            eeg, emg, true_lbl, trial_no = item
            trial_cnt += 1

            # ① trial_start (true_label 포함 → Unity 큐 표시)
            await self._broadcast({
                "type":           "trial_start",
                "trial_no":       trial_no,
                "remaining":      self.simulator.remaining + 1,
                "true_label":     int(true_lbl),
                "true_label_str": LABEL_NAMES[true_lbl],
                "cue_duration_s": self.cue_duration,
            })

            # pause 대기
            while self._paused:
                await asyncio.sleep(0.1)

            # ② cue_duration 대기 (참가자 MI 수행 시간)
            if self.cue_duration > 0:
                await asyncio.sleep(self.cue_duration)

            # ③ 추론
            pred, conf, prob, lat_ms = self.engine.predict(eeg, emg)
            label_str = LABEL_NAMES[pred]
            true_str  = LABEL_NAMES[true_lbl]
            correct   = bool(pred == true_lbl)

            skipped  = conf < self.min_confidence
            msg_type = "prediction_skipped" if skipped else "prediction"

            send_ts = time.time()
            self._pred_send_ts[trial_no] = send_ts

            msg = {
                "type":           msg_type,
                "prediction":     pred,
                "label":          label_str,
                "confidence":     round(float(conf),    4),
                "prob_left":      round(float(prob[0]), 4),
                "prob_right":     round(float(prob[1]), 4),
                "trial_no":       trial_no,
                "true_label":     int(true_lbl),
                "true_label_str": true_str,
                "correct":        correct,
                "latency_ms":     round(float(lat_ms), 2),
                "remaining":      self.simulator.remaining,
                "server_ts":      round(send_ts, 6),    # Unity ack 계산용
            }
            await self._broadcast(msg)

            mark   = "✅" if correct else "❌"
            skip_s = " [SKIP]" if skipped else ""
            print(
                f"  Trial {trial_no:3d} | {label_str:<10} conf={conf:.3f} | "
                f"GT:{true_str:<10} {mark} [{lat_ms:.1f}ms] "
                f"clients={len(self._clients)}{skip_s}"
            )

            results.append({
                "session_id":       self.session_id,
                "sid":              self.sid,
                "trial_no":         trial_no,
                "timestamp":        round(send_ts, 6),
                "true_label":       int(true_lbl),
                "true_label_str":   true_str,
                "pred_label":       pred,
                "pred_label_str":   label_str,
                "correct":          correct,
                "skipped":          skipped,
                "confidence":       round(float(conf),    4),
                "prob_left":        round(float(prob[0]), 4),
                "prob_right":       round(float(prob[1]), 4),
                "inference_latency_ms": round(float(lat_ms), 2),
                "e2e_latency_ms":   None,  # Unity ack 수신 후 채워짐
            })

            if self.max_trials and trial_cnt >= self.max_trials:
                print(f"\n✅ max_trials({self.max_trials}) 도달")
                break

            # interval 중 cue_duration 은 이미 소비했으므로 나머지만 대기
            remaining_sleep = self.interval - self.cue_duration
            if remaining_sleep > 0:
                await asyncio.sleep(remaining_sleep)

        session_end = time.time()
        await self._send_summary(results, session_start, session_end)

    async def _send_summary(self, results: list, session_start: float, session_end: float):
        if not results:
            return

        # e2e latency 채우기 (Unity ack 수신분)
        for r in results:
            tn = r["trial_no"]
            if tn in self._e2e_latencies:
                r["e2e_latency_ms"] = self._e2e_latencies[tn]

        n        = len(results)
        n_skip   = sum(1 for r in results if r["skipped"])
        n_valid  = n - n_skip
        correct  = sum(1 for r in results if r["correct"] and not r["skipped"])
        acc      = correct / n_valid if n_valid > 0 else 0.0

        lats      = [r["inference_latency_ms"] for r in results]
        confs     = [r["confidence"] for r in results]
        e2e_lats  = [r["e2e_latency_ms"] for r in results if r["e2e_latency_ms"] is not None]

        kappa = self._cohen_kappa(results)

        summary_ws = {
            "type":            "summary",
            "n_trials":        n,
            "n_correct":       correct,
            "n_skipped":       n_skip,
            "accuracy":        round(acc,      4),
            "kappa":           round(kappa,    4),
            "avg_confidence":  round(float(np.mean(confs)), 4),
            "avg_latency_ms":  round(float(np.mean(lats)),  2),
            "avg_e2e_ms":      round(float(np.mean(e2e_lats)), 2) if e2e_lats else None,
        }
        await self._broadcast(summary_ws)

        # 파일 저장
        self._save_csv(results)
        self._save_json(results, session_start, session_end, summary_ws)

        print("\n" + "=" * 55)
        print(f"  실행 trial  : {n}개  (스킵: {n_skip}개)")
        print(f"  정확도      : {correct}/{n_valid} ({acc:.1%})")
        print(f"  Cohen κ     : {kappa:.4f}")
        print(f"  평균 신뢰도 : {float(np.mean(confs)):.3f}")
        print(f"  평균 추론   : {float(np.mean(lats)):.1f} ms")
        if e2e_lats:
            print(f"  E2E latency : {float(np.mean(e2e_lats)):.1f} ms (avg, {len(e2e_lats)}개)")
        print(f"\n  로그 저장: {self.log_dir / self.session_id}")
        print("=" * 55)

    # ── 로그 저장 ────────────────────────────────────────────────

    def _save_csv(self, results: list):
        """trial별 CSV 저장 (논문 Table / 통계 분석용)."""
        csv_path = self.log_dir / f"{self.session_id}_trials.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k, "") for k in _CSV_FIELDS})
        print(f"  [CSV] {csv_path.name}")

    def _save_json(self, results: list, t_start: float, t_end: float, summary: dict):
        """세션 메타데이터 + 요약 JSON 저장 (재현성 확보용)."""
        lats = [r["inference_latency_ms"] for r in results]
        e2e  = [r["e2e_latency_ms"] for r in results if r["e2e_latency_ms"] is not None]

        meta = {
            "session_id":    self.session_id,
            "sid":           self.sid,
            "start_time":    datetime.fromtimestamp(t_start).isoformat(),
            "end_time":      datetime.fromtimestamp(t_end).isoformat(),
            "duration_s":    round(t_end - t_start, 1),
            "host":          socket.gethostname(),
            "summary": {
                "n_trials":           summary["n_trials"],
                "n_correct":          summary["n_correct"],
                "n_skipped":          summary["n_skipped"],
                "accuracy":           summary["accuracy"],
                "kappa":              summary["kappa"],
                "avg_confidence":     summary["avg_confidence"],
                "inference_latency": {
                    "mean_ms":   round(float(np.mean(lats)), 2),
                    "std_ms":    round(float(np.std(lats)),  2),
                    "min_ms":    round(float(np.min(lats)),  2),
                    "max_ms":    round(float(np.max(lats)),  2),
                },
                "e2e_latency": {
                    "n_samples": len(e2e),
                    "mean_ms":   round(float(np.mean(e2e)), 2) if e2e else None,
                    "std_ms":    round(float(np.std(e2e)),  2) if e2e else None,
                    "min_ms":    round(float(np.min(e2e)),  2) if e2e else None,
                    "max_ms":    round(float(np.max(e2e)),  2) if e2e else None,
                } if e2e else None,
            },
            "settings": {
                "interval_s":       self.interval,
                "cue_duration_s":   self.cue_duration,
                "min_confidence":   self.min_confidence,
                "shuffle":          self.simulator.shuffle,
            },
        }

        json_path = self.log_dir / f"{self.session_id}_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  [JSON] {json_path.name}")

    @staticmethod
    def _cohen_kappa(results: list) -> float:
        valid = [r for r in results if not r["skipped"]]
        if len(valid) < 2:
            return 0.0
        true_arr = np.array([r["true_label"] for r in valid])
        pred_arr = np.array([r["pred_label"] for r in valid])
        classes  = [0, 1]
        n = len(valid)
        po = float(np.mean(true_arr == pred_arr))
        pe = sum(
            (np.sum(true_arr == c) / n) * (np.sum(pred_arr == c) / n)
            for c in classes
        )
        return (po - pe) / (1.0 - pe) if pe < 1.0 else 0.0

    # ── 서버 시작 ────────────────────────────────────────────────

    def start(self):
        asyncio.run(self._async_main())

    async def _async_main(self):
        print(f"[Server] ws://{self.host}:{self.port}  Ctrl+C 로 종료\n")
        async with websockets.serve(self._handler, self.host, self.port):
            trial_task = asyncio.create_task(self._run_trials())
            try:
                await trial_task
            except asyncio.CancelledError:
                pass
            print("\n[Server] trial 완료. 서버 대기 중 (Ctrl+C)...")
            await asyncio.Future()


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="BCI ONNX WebSocket 서버 (로컬 실행용)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sid",            type=int,   default=3)
    p.add_argument("--host",           type=str,   default="0.0.0.0")
    p.add_argument("--port",           type=int,   default=8765)
    p.add_argument("--interval",       type=float, default=4.0,
                   help="trial 총 간격 (초)")
    p.add_argument("--cue_duration",   type=float, default=2.0,
                   help="큐 표시 → 추론까지 대기 (초, interval 내 포함)")
    p.add_argument("--shuffle",        action="store_true")
    p.add_argument("--max_trials",     type=int,   default=None)
    p.add_argument("--wait_client",    action="store_true",
                   help="클라이언트 연결 후 시작")
    p.add_argument("--min_confidence", type=float, default=0.6)
    p.add_argument("--data_dir",       type=str,   default=None)
    p.add_argument("--onnx_dir",       type=str,   default=None)
    p.add_argument("--log_dir",        type=str,   default=None,
                   help="로그 저장 디렉터리 (기본: BCI_Research/results/vr_sessions)")
    p.add_argument("--session_name",   type=str,   default=None,
                   help="세션 이름 (기본: sXX_YYYYMMDD_HHMMSS 자동 생성)")
    return p.parse_args()


def main():
    args = parse_args()

    # cue_duration 이 interval 보다 크면 경고
    if args.cue_duration >= args.interval:
        print(f"⚠️  cue_duration({args.cue_duration}s) >= interval({args.interval}s)")
        print("   cue_duration 을 interval - 0.5 으로 조정합니다.")
        args.cue_duration = max(0.0, args.interval - 0.5)

    server = BCIWebSocketServer(
        sid            = args.sid,
        host           = args.host,
        port           = args.port,
        interval       = args.interval,
        cue_duration   = args.cue_duration,
        shuffle        = args.shuffle,
        max_trials     = args.max_trials,
        wait_client    = args.wait_client,
        min_confidence = args.min_confidence,
        data_dir       = args.data_dir,
        onnx_dir       = args.onnx_dir,
        log_dir        = args.log_dir,
        session_name   = args.session_name,
    )
    server.start()


if __name__ == "__main__":
    main()
