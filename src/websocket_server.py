
import os
import sys
import json
import asyncio
import argparse
import time
from pathlib import Path

# ── inference.py 에서 공유 컴포넌트 import ───────────────────────
_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))

from inference import (
    SignalSimulator,
    BCIInferenceEngine,
    DEFAULT_CONFIG,
    LABEL_NAMES,
    _DATA_DIR,
    _CKPT_DIR,
)

try:
    import websockets
    from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
except ImportError:
    print("❌ websockets 미설치 — 설치 후 재실행:")
    print("   pip install websockets --break-system-packages")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════
#  BCIWebSocketServer
# ════════════════════════════════════════════════════════════════

class BCIWebSocketServer:
    """
    asyncio 기반 WebSocket 서버.

    SignalSimulator 로 trial 신호를 재생하고, BCIInferenceEngine 으로 추론한 뒤
    연결된 모든 Unity 클라이언트에 JSON 브로드캐스트.

    Parameters
    ----------
    sid            : 피험자 번호 (1~52)
    cfg            : DEFAULT_CONFIG 딕셔너리
    ckpt_dir       : checkpoint .pt 디렉터리
    data_dir       : HDF5 데이터 디렉터리
    host           : WebSocket 바인드 호스트 (기본 "0.0.0.0")
    port           : WebSocket 포트 (기본 8765)
    interval       : trial 간 대기 시간 (초, 기본 4.0)
    shuffle        : trial 순서 랜덤화 여부
    max_trials     : 최대 실행 trial 수 (None = 전체)
    wait_client    : True 면 클라이언트가 1개 이상 연결될 때까지 trial 시작 대기
    device         : 추론 장치 ("cpu" | "cuda")
    min_confidence : 이 값 미만의 예측은 Unity 에 prediction_skipped 로 전송 (기본 0.6)
    """

    def __init__(
        self,
        sid: int,
        cfg: dict,
        ckpt_dir: str,
        data_dir: str,
        host: str       = "0.0.0.0",
        port: int       = 8765,
        interval: float = 4.0,
        shuffle: bool   = False,
        max_trials: int = None,
        wait_client: bool = False,
        device: str     = "cpu",
        min_confidence: float = 0.6,
    ):
        self.sid            = sid
        self.host           = host
        self.port           = port
        self.interval       = interval
        self.max_trials     = max_trials
        self.wait_client    = wait_client
        self.min_confidence = min_confidence
        self._paused        = False

        # 연결된 클라이언트 집합 (thread-safe set 불필요 — asyncio single-thread)
        self._clients: set = set()

        # 컴포넌트 초기화
        h5_path   = os.path.join(data_dir, f"sub-{sid:02d}_member_{cfg['member']}.h5")
        ckpt_path = os.path.join(ckpt_dir, f"best_s{sid:02d}.pt")

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 없음: {h5_path}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"체크포인트 없음: {ckpt_path}")

        print("=" * 60)
        print(f"  BCI WebSocket Server  |  Subject s{sid:02d}")
        print(f"  ws://{host}:{port}")
        print("=" * 60)

        self.simulator = SignalSimulator(h5_path, cfg, shuffle=shuffle)
        self.engine    = BCIInferenceEngine(ckpt_path, cfg, device=device)

        print(f"\n  interval={interval}s  |  max_trials={max_trials or 'all'}")
        print(f"  wait_client={wait_client}  |  device={device}\n")

    # ── 클라이언트 연결 핸들러 ────────────────────────────────────

    async def _handler(self, websocket):
        """새 클라이언트 연결 시 호출. 연결 유지하며 disconnect 감지."""
        addr = websocket.remote_address
        self._clients.add(websocket)
        print(f"[+] 클라이언트 연결: {addr}  (총 {len(self._clients)}개)")

        # 접속 즉시 서버 정보 전송
        await self._send_one(websocket, {
            "type":       "connected",
            "sid":        self.sid,
            "n_trials":   self.simulator.n_trials,
            "interval_s": self.interval,
            "message":    f"BCI WebSocket Server — Subject s{self.sid:02d} ready",
        })

        try:
            # 클라이언트로부터 메시지 수신 대기 (pause/resume/reset 명령)
            async for raw in websocket:
                await self._handle_client_msg(websocket, raw)
        except (ConnectionClosedOK, ConnectionClosedError):
            pass
        finally:
            self._clients.discard(websocket)
            print(f"[-] 클라이언트 해제: {addr}  (총 {len(self._clients)}개)")

    async def _handle_client_msg(self, websocket, raw: str):
        """Unity → 서버 제어 메시지 처리."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        cmd = msg.get("cmd", "")
        if cmd == "reset":
            self.simulator.reset()
            self._paused = False
            print("[Cmd] reset — trial 재시작")
            await self._send_one(websocket, {"type": "ack", "cmd": "reset"})
        elif cmd == "pause":
            self._paused = True
            print("[Cmd] pause")
            await self._broadcast({"type": "ack", "cmd": "pause", "paused": True})
        elif cmd == "resume":
            self._paused = False
            print("[Cmd] resume")
            await self._broadcast({"type": "ack", "cmd": "resume", "paused": False})
        elif cmd == "status":
            await self._send_one(websocket, {
                "type":      "status",
                "remaining": self.simulator.remaining,
                "n_clients": len(self._clients),
                "paused":    self._paused,
            })

    # ── 브로드캐스트 ──────────────────────────────────────────────

    async def _broadcast(self, msg_dict: dict):
        """연결된 모든 클라이언트에 JSON 전송."""
        if not self._clients:
            return
        payload = json.dumps(msg_dict, ensure_ascii=False)
        # 실패한 클라이언트는 조용히 제거
        dead = set()
        results = await asyncio.gather(
            *[ws.send(payload) for ws in self._clients],
            return_exceptions=True,
        )
        for ws, result in zip(list(self._clients), results):
            if isinstance(result, Exception):
                dead.add(ws)
        self._clients -= dead

    async def _send_one(self, websocket, msg_dict: dict):
        """특정 클라이언트 1개에만 전송."""
        try:
            await websocket.send(json.dumps(msg_dict, ensure_ascii=False))
        except Exception:
            pass

    # ── trial 루프 ────────────────────────────────────────────────

    async def _run_trials(self):
        """메인 trial 루프 — 비동기 asyncio 태스크."""
        # wait_client: 클라이언트가 연결될 때까지 대기
        if self.wait_client:
            print("[Trials] 클라이언트 연결 대기 중...")
            while not self._clients:
                await asyncio.sleep(0.5)
            print("[Trials] 클라이언트 감지 — trial 루프 시작")
        else:
            # 서버가 뜨자마자 0.5초 후 시작 (포트 바인딩 완료 대기)
            await asyncio.sleep(0.5)
            print("[Trials] trial 루프 시작 (클라이언트 없어도 진행)")

        results = []
        trial_count = 0

        while True:
            item = self.simulator.next_trial()
            if item is None:
                print("\n✅ 전체 trial 완료 — 요약 브로드캐스트")
                break

            eeg, emg, true_lbl, trial_no = item
            trial_count += 1

            # trial_start 알림 (Unity 가 캐릭터 준비할 시간)
            await self._broadcast({
                "type":      "trial_start",
                "trial_no":  trial_no,
                "remaining": self.simulator.remaining + 1,
            })

            # pause 대기 (non-blocking poll)
            while self._paused:
                await asyncio.sleep(0.1)

            # 추론
            pred, conf, prob, latency_ms = self.engine.predict(eeg, emg)
            label_str    = LABEL_NAMES[pred]
            true_lbl_str = LABEL_NAMES[true_lbl]
            correct      = bool(pred == true_lbl)

            # confidence threshold 필터
            skipped = conf < self.min_confidence
            msg_type = "prediction_skipped" if skipped else "prediction"

            msg = {
                "type":          msg_type,
                "prediction":    pred,
                "label":         label_str,
                "confidence":    round(float(conf),  4),
                "prob_left":     round(float(prob[0]), 4),
                "prob_right":    round(float(prob[1]), 4),
                "trial_no":      trial_no,
                "true_label":    int(true_lbl),
                "true_label_str": true_lbl_str,
                "correct":       correct,
                "latency_ms":    round(float(latency_ms), 2),
                "remaining":     self.simulator.remaining,
            }

            await self._broadcast(msg)

            # 콘솔 로그
            mark   = "✅" if correct else "❌"
            skip_s = " [SKIP low-conf]" if skipped else ""
            print(
                f"  Trial {trial_no:3d} | {label_str:<10} (conf={conf:.3f}) "
                f"| GT: {true_lbl_str:<10} {mark}  [{latency_ms:.1f}ms]"
                f"  clients={len(self._clients)}{skip_s}"
            )

            results.append({
                "trial":      trial_no,
                "pred":       pred,
                "true":       true_lbl,
                "correct":    correct,
                "confidence": conf,
                "latency_ms": latency_ms,
            })

            if self.max_trials and trial_count >= self.max_trials:
                print(f"\n✅ max_trials({self.max_trials}) 도달 — 종료")
                break

            # interval 대기 (non-blocking)
            await asyncio.sleep(self.interval)

        # 요약 브로드캐스트
        await self._send_summary(results)

    async def _send_summary(self, results: list):
        if not results:
            return
        n        = len(results)
        correct  = sum(r["correct"] for r in results)
        acc      = correct / n

        import numpy as np
        avg_lat  = float(np.mean([r["latency_ms"] for r in results]))
        avg_conf = float(np.mean([r["confidence"] for r in results]))

        summary = {
            "type":          "summary",
            "n_trials":      n,
            "n_correct":     correct,
            "accuracy":      round(acc, 4),
            "avg_confidence": round(avg_conf, 4),
            "avg_latency_ms": round(avg_lat, 2),
        }

        await self._broadcast(summary)

        print("\n" + "=" * 55)
        print("  요약")
        print("=" * 55)
        print(f"  실행 trial  : {n}개")
        print(f"  정확도      : {correct}/{n}  ({acc:.1%})")
        print(f"  평균 신뢰도 : {avg_conf:.3f}")
        print(f"  평균 추론   : {avg_lat:.1f} ms")
        print("=" * 55)

    # ── 서버 시작 ─────────────────────────────────────────────────

    def start(self):
        """이벤트 루프를 시작하고 서버를 실행. Ctrl+C 로 종료."""
        asyncio.run(self._async_main())

    async def _async_main(self):
        print(f"[Server] 시작 — ws://{self.host}:{self.port}")
        print("  Ctrl+C 로 종료\n")

        async with websockets.serve(self._handler, self.host, self.port):
            # trial 루프를 백그라운드 태스크로 실행
            trial_task = asyncio.create_task(self._run_trials())
            try:
                await trial_task
            except asyncio.CancelledError:
                pass
            # trial 완료 후에도 서버는 계속 열어둠 (클라이언트가 summary 읽을 수 있도록)
            print("\n[Server] trial 완료. 서버 대기 중 (Ctrl+C 로 종료)...")
            await asyncio.Future()   # run forever until cancelled


# ════════════════════════════════════════════════════════════════
#  CLI 진입점
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="BCI-VR WebSocket 서버",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sid",        type=int,   default=3,
                   help="피험자 번호 (1~52)")
    p.add_argument("--host",       type=str,   default="0.0.0.0",
                   help="WebSocket 바인드 호스트")
    p.add_argument("--port",       type=int,   default=8765,
                   help="WebSocket 포트")
    p.add_argument("--interval",   type=float, default=4.0,
                   help="trial 간격 (초)")
    p.add_argument("--shuffle",    action="store_true",
                   help="trial 순서 랜덤화")
    p.add_argument("--max_trials", type=int,   default=None,
                   help="최대 trial 수 (기본: 전체)")
    p.add_argument("--wait_client", action="store_true",
                   help="클라이언트 1개 이상 연결 후 trial 시작")
    p.add_argument("--ckpt_dir",   type=str,   default=str(_CKPT_DIR),
                   help="checkpoint 디렉터리")
    p.add_argument("--data_dir",   type=str,   default=str(_DATA_DIR),
                   help="HDF5 데이터 디렉터리")
    p.add_argument("--device",         type=str,   default="cpu",
                   choices=["cpu", "cuda"],
                   help="추론 장치")
    p.add_argument("--min_confidence", type=float, default=0.6,
                   help="이 값 미만 예측은 prediction_skipped 로 전송 (기본 0.6)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = dict(DEFAULT_CONFIG)

    server = BCIWebSocketServer(
        sid             = args.sid,
        cfg             = cfg,
        ckpt_dir        = args.ckpt_dir,
        data_dir        = args.data_dir,
        host            = args.host,
        port            = args.port,
        interval        = args.interval,
        shuffle         = args.shuffle,
        max_trials      = args.max_trials,
        wait_client     = args.wait_client,
        device          = args.device,
        min_confidence  = args.min_confidence,
    )
    server.start()


if __name__ == "__main__":
    main()
