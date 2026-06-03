"""
latency_bench.py — VR 없이 E2E latency 전체 측정
==================================================
서버(server_onnx.py)와 mock Unity 클라이언트를 같은 프로세스에서
asyncio 태스크로 동시 실행해 WS 전송 지연까지 포함한 완전한
latency 분해를 측정합니다.

측정 지표
---------
  inference_ms   : 서버 ONNX 추론 시간 (server_onnx 내부 perf_counter)
  ws_ms          : WebSocket 전송 지연  = client_recv_ts - server_send_ts
  server_total_ms: inference + 직렬화 오버헤드 (server_ts - infer_start)
                   → 현재 server_ts = inference 완료 직후 time.time()
  e2e_est_ms     : inference_ms + ws_ms + IK_est (11ms @ 90Hz, 상수)

실행
----
  # 단일 커맨드 (서버 + 클라이언트 자동 시작)
  python src/latency_bench.py --sid 3

  # 빠른 워밍업 확인 (20 trials)
  python src/latency_bench.py --sid 3 --n_trials 20

옵션
----
  --sid          피험자 번호 (기본 3)
  --n_trials     측정 trial 수 (기본 200)
  --port         WebSocket 포트 (기본 8766 — 기존 서버와 충돌 방지)
  --ik_est_ms    Unity IK 추정 지연 (기본 11.0ms, 90Hz 1프레임)
  --no_save      결과 CSV/JSON 저장 안 함
"""

import asyncio
import csv
import json
import os
import socket
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_SRC_DIR  = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent

# server_onnx 컴포넌트 재사용
sys.path.insert(0, str(_SRC_DIR))
from server_onnx import (
    SignalSimulator,
    BCIInferenceEngineONNX,
    BCIWebSocketServer,
    _DATA_DIR,
    _ONNX_DIR,
    _LOG_DIR,
    LABEL_NAMES,
)

try:
    import websockets
    from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
except ImportError:
    print("❌ websockets 미설치: pip install websockets")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════
#  Mock Unity 클라이언트
# ════════════════════════════════════════════════════════════════

class LatencyBenchClient:
    """
    Python mock Unity 클라이언트.
    prediction 수신 즉시 latency_ack를 돌려보내고
    ws_ms = recv_ts - server_ts 를 기록합니다.
    """

    def __init__(self, port: int, ik_est_ms: float = 11.0):
        self.port       = port
        self.ik_est_ms  = ik_est_ms
        self.records: list[dict] = []
        self._done      = asyncio.Event()

    async def run(self):
        uri = f"ws://localhost:{self.port}"

        # 서버 준비 대기 (최대 10초)
        for _ in range(20):
            try:
                async with websockets.connect(uri, ping_interval=None) as ws:
                    await self._session(ws)
                return
            except OSError:
                await asyncio.sleep(0.5)

        print("❌ 서버 연결 실패 (10초 대기 초과)")

    async def _session(self, ws):
        print(f"[Client] 연결 성공 → ws://localhost:{self.port}")

        async for raw in ws:
            recv_ts = time.time()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            t = msg.get("type", "")

            if t == "connected":
                print(f"[Client] s{msg.get('sid'):02d}  "
                      f"total {msg.get('n_trials')} trials")

            elif t in ("prediction", "prediction_skipped"):
                trial_no   = msg.get("trial_no", -1)
                server_ts  = msg.get("server_ts", 0.0)
                infer_ms   = msg.get("latency_ms", 0.0)
                skipped    = (t == "prediction_skipped")

                # latency_ack 즉시 전송 (서버 CSV e2e_latency_ms 채움)
                ack = json.dumps({
                    "cmd": "latency_ack",
                    "trial_no": trial_no,
                    "unity_recv_ts": recv_ts,
                })
                try:
                    await ws.send(ack)
                except Exception:
                    pass

                # WS 전송 지연
                ws_ms = (recv_ts - server_ts) * 1000.0 if server_ts > 0 else None

                self.records.append({
                    "trial_no":   trial_no,
                    "skipped":    skipped,
                    "correct":    msg.get("correct", False),
                    "confidence": msg.get("confidence", 0.0),
                    "true_label": msg.get("true_label", -1),
                    "pred_label": msg.get("prediction", -1),
                    "infer_ms":   round(infer_ms, 3),
                    "ws_ms":      round(ws_ms, 3) if ws_ms is not None else None,
                    "e2e_est_ms": round(infer_ms + (ws_ms or 0) + self.ik_est_ms, 3),
                })

                _print_trial(trial_no, msg, infer_ms, ws_ms, self.ik_est_ms)

            elif t == "summary":
                print(f"\n[Client] 세션 종료 수신")
                break

        self._done.set()


def _print_trial(no, msg, infer_ms, ws_ms, ik_ms):
    mark  = "✅" if msg.get("correct") else "❌"
    skip  = " [SKIP]" if msg.get("type") == "prediction_skipped" else ""
    ws_s  = f"{ws_ms:5.1f}" if ws_ms is not None else "  N/A"
    e2e   = infer_ms + (ws_ms or 0) + ik_ms
    print(
        f"  Trial {no:3d} | "
        f"GT={msg.get('true_label_str','?'):<8} "
        f"pred={msg.get('label','?'):<8} "
        f"{mark}{skip} | "
        f"infer={infer_ms:5.1f}ms  ws={ws_s}ms  e2e≈{e2e:5.1f}ms"
    )


# ════════════════════════════════════════════════════════════════
#  결과 분석 & 저장
# ════════════════════════════════════════════════════════════════

def _percentile(data, p):
    """단순 백분위수 (numpy 없이)."""
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def analyze_and_report(records: list[dict], ik_est_ms: float,
                        save: bool, session_id: str):
    valid   = [r for r in records if not r["skipped"]]
    skipped = [r for r in records if r["skipped"]]

    infer_vals = [r["infer_ms"]   for r in records]
    ws_vals    = [r["ws_ms"]      for r in records if r["ws_ms"] is not None]
    e2e_vals   = [r["e2e_est_ms"] for r in records if r["e2e_est_ms"] is not None]

    correct    = sum(1 for r in valid if r["correct"])
    acc        = correct / len(valid) if valid else 0.0

    print("\n" + "=" * 65)
    print(f"  Latency Benchmark — {session_id}")
    print("=" * 65)
    print(f"  총 trial     : {len(records)}")
    print(f"  Skip (conf<0.6): {len(skipped)}")
    print(f"  정확도       : {correct}/{len(valid)}  ({acc:.1%})")

    def _stats(label, vals):
        if not vals:
            print(f"\n  [{label}]  데이터 없음")
            return
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"\n  [{label}]  n={len(vals)}")
        print(f"    mean ± std : {mean:.2f} ± {std:.2f} ms")
        print(f"    min / max  : {min(vals):.2f} / {max(vals):.2f} ms")
        print(f"    p50 / p95  : {_percentile(vals,50):.2f} / {_percentile(vals,95):.2f} ms")

    _stats("Server-side inference (ONNX + Python overhead)", infer_vals)
    _stats("WebSocket transport  (server_ts → client recv)", ws_vals)
    _stats(f"E2E estimate         (inference + WS + IK {ik_est_ms:.0f}ms)", e2e_vals)

    if ws_vals and infer_vals:
        print(f"\n  [Latency 분해 — 평균]")
        print(f"    ONNX 추론 + Python : {statistics.mean(infer_vals):.1f} ms")
        print(f"    WebSocket 전송     : {statistics.mean(ws_vals):.1f} ms")
        print(f"    Unity IK (추정)    : {ik_est_ms:.1f} ms")
        print(f"    ─────────────────────────────")
        total = statistics.mean(infer_vals) + statistics.mean(ws_vals) + ik_est_ms
        print(f"    E2E 합계 (추정)    : {total:.1f} ms")
        print(f"    목표 (<100 ms)     : {'✅ 통과' if total < 100 else '❌ 초과'}")

    print("=" * 65)

    if not save:
        return

    # ── CSV 저장 ──────────────────────────────────────────────
    log_dir = _LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_path = log_dir / f"{session_id}_latency.csv"
    fields   = ["trial_no", "skipped", "correct", "confidence",
                 "true_label", "pred_label",
                 "infer_ms", "ws_ms", "e2e_est_ms"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    print(f"\n  [저장] {csv_path.name}")

    # ── JSON 저장 ─────────────────────────────────────────────
    def _stat_dict(vals):
        if not vals:
            return None
        return {
            "n":       len(vals),
            "mean_ms": round(statistics.mean(vals), 3),
            "std_ms":  round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 3),
            "min_ms":  round(min(vals), 3),
            "max_ms":  round(max(vals), 3),
            "p50_ms":  round(_percentile(vals, 50), 3),
            "p95_ms":  round(_percentile(vals, 95), 3),
        }

    meta = {
        "session_id":   session_id,
        "timestamp":    datetime.now().isoformat(),
        "host":         socket.gethostname(),
        "n_trials":     len(records),
        "n_skipped":    len(skipped),
        "accuracy":     round(acc, 4),
        "ik_est_ms":    ik_est_ms,
        "inference":    _stat_dict(infer_vals),
        "ws_transport": _stat_dict(ws_vals),
        "e2e_estimate": _stat_dict(e2e_vals),
        "note": (
            "inference_ms = server-side ONNX + Python overhead; "
            "ws_ms = server_ts → client recv; "
            f"e2e_est = inference + ws + ik_est({ik_est_ms}ms)"
        ),
    }

    json_path = log_dir / f"{session_id}_latency.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  [저장] {json_path.name}")


# ════════════════════════════════════════════════════════════════
#  메인: 서버 + 클라이언트 동시 실행
# ════════════════════════════════════════════════════════════════

async def _async_main(args):
    session_id = f"bench_s{args.sid:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    port       = args.port

    print("=" * 65)
    print(f"  BCI Latency Benchmark  |  Subject s{args.sid:02d}")
    print(f"  trials={args.n_trials}  port={port}  ik_est={args.ik_est_ms}ms")
    print("=" * 65 + "\n")

    # 서버 인스턴스 (VR 세션과 동일한 로직)
    server = BCIWebSocketServer(
        sid          = args.sid,
        port         = port,
        interval     = args.interval,
        cue_duration = args.cue_duration,
        max_trials   = args.n_trials,
        wait_client  = True,          # 클라이언트 연결 후 시작
        session_name = session_id,
    )

    # 클라이언트 인스턴스
    client = LatencyBenchClient(port=port, ik_est_ms=args.ik_est_ms)

    # 서버 async 루프 (별도 태스크)
    async def _server_task():
        async with websockets.serve(server._handler, server.host, port):
            await server._run_trials()

    srv_task = asyncio.create_task(_server_task())
    cli_task = asyncio.create_task(client.run())

    # 클라이언트가 끝날 때까지 대기 (서버도 자연히 종료)
    await cli_task
    srv_task.cancel()
    try:
        await srv_task
    except asyncio.CancelledError:
        pass

    # 결과 분석
    analyze_and_report(
        client.records,
        ik_est_ms  = args.ik_est_ms,
        save       = not args.no_save,
        session_id = session_id,
    )


def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="BCI E2E Latency Benchmark (VR 불필요)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sid",          type=int,   default=3,
                   help="피험자 번호")
    p.add_argument("--n_trials",     type=int,   default=200,
                   help="측정 trial 수")
    p.add_argument("--port",         type=int,   default=8766,
                   help="WebSocket 포트 (기존 8765와 충돌 방지)")
    p.add_argument("--interval",     type=float, default=6.0,
                   help="trial 간격 (초)")
    p.add_argument("--cue_duration", type=float, default=2.0,
                   help="큐 대기 시간 (초)")
    p.add_argument("--ik_est_ms",    type=float, default=11.0,
                   help="Unity IK 1프레임 추정 (90Hz=11ms)")
    p.add_argument("--no_save",      action="store_true",
                   help="CSV/JSON 저장 안 함")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(_async_main(args))
