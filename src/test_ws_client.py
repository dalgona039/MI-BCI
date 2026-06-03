"""
test_ws_client.py — BCI WebSocket 서버 테스트 클라이언트
=========================================================
Unity 없이 Python 에서 서버 동작을 검증하고 latency 를 측정합니다.

사용법:
  # 터미널 1: 서버 실행
  python src/websocket_server.py --sid 3 --wait_client --max_trials 10

  # 터미널 2: 테스트 클라이언트 실행
  python src/test_ws_client.py

  # 원격 서버
  python src/test_ws_client.py --host 192.168.0.10 --port 8765
"""

import asyncio
import json
import time
import argparse
import statistics
from datetime import datetime

try:
    import websockets
except ImportError:
    print("❌ websockets 미설치: pip install websockets")
    raise


# ════════════════════════════════════════════════════════════════
#  결과 누적
# ════════════════════════════════════════════════════════════════

class TestResult:
    def __init__(self):
        self.messages        = []
        self.predictions     = []
        self.skipped         = []
        self.latencies_ms    = []
        self.recv_times      = {}   # trial_no → 수신 시각
        self.connected_at    = None
        self.summary         = None

    def record(self, msg: dict, recv_ts: float):
        self.messages.append(msg)
        t = msg.get("type", "")

        if t == "connected":
            self.connected_at = recv_ts

        elif t == "prediction":
            self.predictions.append(msg)
            lat = msg.get("latency_ms", 0.0)
            self.latencies_ms.append(lat)
            self.recv_times[msg.get("trial_no")] = recv_ts

        elif t == "prediction_skipped":
            self.skipped.append(msg)

        elif t == "summary":
            self.summary = msg

    def report(self):
        n_pred  = len(self.predictions)
        n_skip  = len(self.skipped)
        correct = sum(1 for p in self.predictions if p.get("correct"))

        print("\n" + "=" * 60)
        print("  BCI WebSocket 테스트 결과")
        print("=" * 60)
        print(f"  수신 메시지 총계 : {len(self.messages)}개")
        print(f"  prediction       : {n_pred}개  (정확: {correct}/{n_pred})")
        print(f"  prediction_skipped: {n_skip}개")

        if self.latencies_ms:
            lats = self.latencies_ms
            print(f"\n  [추론 latency — 서버 측]")
            print(f"    평균  : {statistics.mean(lats):.2f} ms")
            print(f"    중앙값: {statistics.median(lats):.2f} ms")
            print(f"    최소  : {min(lats):.2f} ms")
            print(f"    최대  : {max(lats):.2f} ms")
            if len(lats) > 1:
                print(f"    표준편차: {statistics.stdev(lats):.2f} ms")

        if self.summary:
            s = self.summary
            print(f"\n  [서버 요약]")
            print(f"    accuracy   : {s.get('accuracy', 0):.1%}")
            print(f"    avg_latency: {s.get('avg_latency_ms', 0):.2f} ms")

        print("\n  [메시지 타입 분포]")
        from collections import Counter
        dist = Counter(m["type"] for m in self.messages)
        for k, v in sorted(dist.items()):
            print(f"    {k:<25} : {v}개")

        # 전체 latency budget 추정
        avg_model_ms = statistics.mean(self.latencies_ms) if self.latencies_ms else 0
        print(f"\n  [Latency Budget 추정 (LAN 환경)]")
        print(f"    모델 추론  : {avg_model_ms:.1f} ms")
        print(f"    WebSocket  : ~2–5 ms")
        print(f"    Unity IK   : ~11–14 ms")
        print(f"    합계 (예상): ~{avg_model_ms + 3 + 12:.0f} ms  (목표 <100 ms)")

        print("=" * 60)

        # JSON 검증 결과
        _validate_message_schema(self.predictions, self.skipped)


def _validate_message_schema(predictions, skipped):
    required_pred = {
        "type", "prediction", "label", "confidence",
        "prob_left", "prob_right", "trial_no",
        "true_label", "true_label_str", "correct",
        "latency_ms", "remaining",
    }
    errors = []
    for p in predictions:
        missing = required_pred - set(p.keys())
        if missing:
            errors.append(f"  Trial {p.get('trial_no')}: 누락 필드 {missing}")

    if errors:
        print("\n  ⚠️  스키마 검증 실패:")
        for e in errors:
            print(e)
    else:
        print(f"\n  ✅ 스키마 검증 통과 ({len(predictions)}개 prediction 메시지)")

    # confidence 범위 확인
    bad_conf = [p for p in predictions if not (0.0 <= p.get("confidence", -1) <= 1.0)]
    if bad_conf:
        print(f"  ⚠️  confidence 범위 이상: {len(bad_conf)}개")
    else:
        print(f"  ✅ confidence 범위 정상 (0~1)")

    # prediction 값 확인
    bad_pred = [p for p in predictions if p.get("prediction") not in (0, 1)]
    if bad_pred:
        print(f"  ⚠️  prediction 값 이상 (0/1 아님): {len(bad_pred)}개")
    else:
        print(f"  ✅ prediction 값 정상 (0=Left / 1=Right)")


# ════════════════════════════════════════════════════════════════
#  비동기 테스트 클라이언트
# ════════════════════════════════════════════════════════════════

async def run_client(host: str, port: int, timeout: float, send_status: bool):
    uri = f"ws://{host}:{port}"
    result = TestResult()

    print(f"[Client] 연결 시도: {uri}")
    print(f"[Client] 타임아웃: {timeout}s\n")

    try:
        async with websockets.connect(uri, ping_interval=None) as ws:
            print(f"[Client] 연결 성공 ✅  ({datetime.now().strftime('%H:%M:%S')})\n")

            # 연결 직후 status 요청 (선택)
            if send_status:
                await ws.send(json.dumps({"cmd": "status"}))

            last_msg_time = time.monotonic()

            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    print(f"\n[Client] {timeout}s 동안 메시지 없음 → 종료")
                    break

                recv_ts = time.monotonic()
                last_msg_time = recv_ts

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    print(f"[Client] JSON 파싱 실패: {raw[:80]}")
                    continue

                result.record(msg, recv_ts)
                _print_msg(msg)

                if msg.get("type") == "summary":
                    break

    except ConnectionRefusedError:
        print(f"❌ 연결 거부 — 서버가 실행 중인지 확인:")
        print(f"   python src/websocket_server.py --sid 3 --wait_client")
        return
    except Exception as e:
        print(f"❌ 연결 오류: {e}")
        return

    result.report()


def _print_msg(msg: dict):
    t = msg.get("type", "?")

    if t == "connected":
        print(f"  [connected] Subject s{msg.get('sid'):02d}, "
              f"{msg.get('n_trials')}개 trial, "
              f"interval={msg.get('interval_s')}s")

    elif t == "trial_start":
        print(f"  [trial_start] Trial {msg.get('trial_no'):3d}  "
              f"남은: {msg.get('remaining')}")

    elif t == "prediction":
        mark = "✅" if msg.get("correct") else "❌"
        print(f"  [prediction]  Trial {msg.get('trial_no'):3d} | "
              f"{msg.get('label'):<10} conf={msg.get('confidence'):.3f} | "
              f"GT={msg.get('true_label_str'):<10} {mark}  "
              f"[{msg.get('latency_ms'):.1f}ms]")

    elif t == "prediction_skipped":
        print(f"  [SKIPPED]     Trial {msg.get('trial_no'):3d} | "
              f"{msg.get('label'):<10} conf={msg.get('confidence'):.3f} (< threshold)")

    elif t == "summary":
        print(f"\n  [summary] {msg.get('n_correct')}/{msg.get('n_trials')} "
              f"({msg.get('accuracy'):.1%})  "
              f"avg_lat={msg.get('avg_latency_ms'):.1f}ms")

    elif t == "status":
        print(f"  [status] remaining={msg.get('remaining')}  "
              f"clients={msg.get('n_clients')}  paused={msg.get('paused')}")

    elif t == "ack":
        print(f"  [ack] cmd={msg.get('cmd')}")

    else:
        print(f"  [{t}] {str(msg)[:80]}")


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="BCI WebSocket 테스트 클라이언트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host",    type=str,   default="localhost",
                   help="서버 호스트")
    p.add_argument("--port",    type=int,   default=8765,
                   help="서버 포트")
    p.add_argument("--timeout", type=float, default=15.0,
                   help="메시지 수신 대기 타임아웃 (초)")
    p.add_argument("--status",  action="store_true",
                   help="연결 직후 status 요청 전송")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_client(args.host, args.port, args.timeout, args.status))
