# BCI-VR 실험 튜토리얼

**환경**: Mac (Python 서버) + Meta Quest 3 (Unity 앱)  
**데이터**: GigaDB Cho2017 52명 EEG+sEMG, ONNX 모델 52개  
**목표**: 오프라인 시뮬레이션 → VR 아바타 팔 실시간 제어 데모

---

## 전체 흐름

```
Mac                          Meta Quest 3
─────────────────────        ─────────────────────
server_onnx.py               Unity 앱
  ↓ HDF5 trial 재생
  ↓ ONNX 추론 (12ms)
  ↓ WebSocket (LAN)    →     AvatarController.cs
                              BCIExperimentManager.cs
                              BCISessionLogger.cs
                                ↓
                             VR 큐 + 아바타 팔 동작
                             세션 로그 저장
```

---

## PART 1 — Mac 서버 실행

### 1-1. Mac LAN IP 확인

```bash
ipconfig getifaddr en0
# 예: 192.168.35.183  → Quest 3에서 이 주소로 접속
```

### 1-2. 서버 실행

```bash
cd /Volumes/a3122a1/MI-BCI

.venv/bin/python src/server_onnx.py \
  --sid 3 \
  --wait_client \
  --interval 6.0 \
  --cue_duration 2.0 \
  --min_confidence 0.6
```

**정상 출력:**
```
BCI ONNX WebSocket Server  |  Subject s03
ws://0.0.0.0:8765
[Trials] 클라이언트 연결 대기...   ← Quest 3 앱 켤 때까지 대기
```

**`--sid` 선택 기준:**

| sid | 정확도 | 용도 |
|-----|--------|------|
| 3   | ~97%   | 데모·논문 시각 자료 |
| 1   | ~74%   | 평균 성능 대표 |
| 7   | 낮음   | Right MI bias 케이스 |

---

## PART 2 — Unity 프로젝트 세팅 (최초 1회)

### 2-1. 패키지 설치

Unity 프로젝트의 `Packages/manifest.json` 을 아래 파일로 교체:

```
/Volumes/a3122a1/MI-BCI/unity/Packages/manifest.json
```

Unity Editor 재시작 시 자동 설치:
- NativeWebSocket
- Meta XR SDK (Oculus)
- Universal Render Pipeline (URP)
- XR Interaction Toolkit
- TextMeshPro

### 2-2. C# 스크립트 복사

아래 4개 파일을 Unity 프로젝트 `Assets/Scripts/` 에 복사:

```
unity/AvatarController.cs        ← WebSocket 연결 + IK 팔 제어
unity/BCIExperimentManager.cs    ← VR 큐 UI + 피드백
unity/BCISessionLogger.cs        ← 논문용 로그 저장
```

### 2-3. 씬 구성

| GameObject | 부착 컴포넌트 | 설명 |
|------------|--------------|------|
| 아바타 | `AvatarController`, `Animator` | 팔 IK 제어 |
| 빈 오브젝트 | `BCIExperimentManager`, `BCISessionLogger` | 실험 관리 |
| WorldSpace Canvas | TMP Text × 3 | VR 내 UI 표시 |
| Left IK Target | `Transform` | 왼팔 목표 위치 |
| Right IK Target | `Transform` | 오른팔 목표 위치 |

### 2-4. AvatarController Inspector 설정

```
Server URL       : ws://192.168.35.183:8765   ← Mac LAN IP
Left Arm Target  : 왼팔 IK Target Transform
Right Arm Target : 오른팔 IK Target Transform
Min Confidence   : 0.6
Hold Duration    : 1.5
```

### 2-5. BCIExperimentManager Inspector 설정

```
Avatar Controller  : AvatarController 드래그
Cue Panel Root     : WorldSpace Canvas 루트
Cue Text           : 중앙 큰 TMP Text
Status Text        : 상단 상태 TMP Text
Result Text        : 결과 TMP Text
Countdown Duration : 1.0
Result Hold Dur    : 1.5
```

### 2-6. Animator IK 설정

Animator Controller → Layer 설정에서 **IK Pass 체크** 필수.

### 2-7. Meta Quest 3 빌드 설정

```
File → Build Settings
  Platform  : Android
  Run Device: Meta Quest 3

Edit → Project Settings
  XR Plugin Management → Oculus 체크
  Player
    Minimum API Level : Android 10 (API 29)
    Graphics API      : OpenGLES3
    Scripting Backend : IL2CPP
    Target Architectures : ARM64
```

---

## PART 3 — 실험 실행

### 실행 순서

```
① Mac 터미널에서 서버 실행 (PART 1-2 명령어)
② Quest 3 착용 후 앱 실행
③ 자동 WebSocket 연결
④ 실험 자동 시작
```

> Mac과 Quest 3가 **같은 WiFi** 에 연결되어 있어야 합니다.

### 1 Trial 흐름 (기본값 interval=6s)

```
│← 1s 카운트다운 →│← 2s 큐 표시 →│← 1.5s 결과 →│← 1.5s 대기 →│
     3  2  1          ← LEFT            팔 동작          준비
                      RIGHT →           ✅ / ❌
```

### 콘솔 출력 예시

```
Trial   1 | Left MI    conf=0.898 | GT:Left MI    ✅ [12.4ms]
Trial   2 | Right MI   conf=0.743 | GT:Right MI   ✅ [11.8ms]
Trial   3 | Left MI    conf=0.421 | GT:Right MI   ❌ [SKIP low-conf]
...
실행 trial  : 200개  (스킵: 3개)
정확도      : 194/197 (98.5%)
Cohen κ     : 0.940
평균 신뢰도 : 0.871
평균 추론   : 12.4 ms
E2E latency : 28.3 ms (avg)

로그 저장: BCI_Research/results/vr_sessions/s03_20260529_143022
```

---

## PART 4 — 실험 후 로그 확인

### 저장 위치

```
BCI_Research/results/vr_sessions/
├── s03_20260529_143022_trials.csv     ← trial별 전체 데이터
└── s03_20260529_143022_summary.json   ← 논문 수치 요약

Quest 3 내부 (ADB로 추출):
/sdcard/Android/data/<패키지명>/files/BCI_Sessions/
└── s03_20260529_143022_unity.csv      ← Unity 수신 타임스탬프
```

### trials.csv 컬럼

| 컬럼 | 설명 |
|------|------|
| `session_id` | 세션 식별자 |
| `trial_no` | trial 번호 |
| `timestamp` | 예측 전송 시각 (Unix) |
| `true_label` | 정답 (0=Left, 1=Right) |
| `pred_label` | 예측 결과 |
| `correct` | 정답 여부 |
| `skipped` | 신뢰도 미달로 스킵 여부 |
| `confidence` | 예측 신뢰도 (0~1) |
| `prob_left` / `prob_right` | 각 클래스 확률 |
| `inference_latency_ms` | ONNX 추론 시간 |
| `e2e_latency_ms` | 서버 전송 → Unity 수신 (end-to-end) |

### summary.json 구조

```json
{
  "session_id": "s03_20260529_143022",
  "sid": 3,
  "start_time": "2026-05-29T14:30:22",
  "summary": {
    "accuracy": 0.97,
    "kappa": 0.94,
    "avg_confidence": 0.871,
    "inference_latency": {
      "mean_ms": 12.4,
      "std_ms": 1.2,
      "min_ms": 8.8,
      "max_ms": 15.5
    },
    "e2e_latency": {
      "mean_ms": 28.3,
      "std_ms": 3.1
    }
  },
  "settings": {
    "interval_s": 6.0,
    "cue_duration_s": 2.0,
    "min_confidence": 0.6
  }
}
```

### 논문에 쓸 수치 위치

| 논문 항목 | 파일 | 필드 |
|-----------|------|------|
| VR 데모 정확도 | `summary.json` | `summary.accuracy` |
| Cohen's κ | `summary.json` | `summary.kappa` |
| 추론 latency | `summary.json` | `inference_latency.mean_ms ± std_ms` |
| End-to-End latency | `summary.json` | `e2e_latency.mean_ms ± std_ms` |
| Trial별 confidence 분포 | `trials.csv` | `confidence` 열 |

---

## 자주 쓰는 실행 옵션

```bash
# 테스트 (20 trial만)
.venv/bin/python src/server_onnx.py --sid 3 --max_trials 20 --wait_client

# 순서 랜덤화 (실제 실험)
.venv/bin/python src/server_onnx.py --sid 3 --wait_client --shuffle

# 세션 이름 지정
.venv/bin/python src/server_onnx.py --sid 3 --wait_client --session_name demo_run_1

# 다른 피험자
.venv/bin/python src/server_onnx.py --sid 1 --wait_client

# 클라이언트 없이 빠른 검증 (Unity 없이)
.venv/bin/python src/server_onnx.py --sid 3 --max_trials 10 --interval 0.5 --cue_duration 0
```

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| Unity가 연결 안 됨 | IP 오타 또는 다른 WiFi | `ipconfig getifaddr en0` 재확인 |
| `ONNX 없음` 오류 | sid에 해당하는 ONNX 파일 없음 | `ls BCI_Research/results/onnx/` 확인 |
| `HDF5 없음` 오류 | 데이터 파일 경로 문제 | `--data_dir` 옵션으로 경로 직접 지정 |
| 팔이 안 움직임 | IK Pass 미활성화 | Animator Layer → IK Pass 체크 |
| 모든 예측이 SKIP | confidence < 0.6 | `--min_confidence 0.3` 으로 낮춰서 테스트 |
| `websockets 미설치` | venv 패키지 문제 | `.venv/bin/pip install websockets` |
