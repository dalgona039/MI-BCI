/*
 * AvatarController.cs
 * BCI-VR Pipeline — S5 Real-time VR Integration
 *
 * 요구사항:
 *   NativeWebSocket — Package Manager > Add package from git URL:
 *   https://github.com/endel/NativeWebSocket.git#upm
 *
 * Unity 설정:
 *   1. GameObject 에 AvatarController 추가
 *   2. leftArmTarget / rightArmTarget: 양팔 IK Target Transform 연결
 *   3. Animator Controller Layer > IK Pass 활성화
 *   4. directionClasses 배열: 각 예측 클래스의 방향 각도 및 팔 설정
 *
 * DirectionClass 각도 기준 (아바타 정면 기준 XY 평면):
 *     90° (up)
 *       │
 * 180° ─┼─ 0° (right)
 *       │
 *    270° (down)
 *
 * Python 서버:
 *   python src/server_onnx.py --sid 3 --wait_client --min_confidence 0.6
 *
 * BCIExperimentManager 와 연동:
 *   OnTrialStart, OnPrediction, OnPredictionSkipped, OnSummary 이벤트 구독
 */

using System;
using System.Collections;
using UnityEngine;
using NativeWebSocket;

public enum ArmSide { Left, Right, Both }

[Serializable]
public class DirectionClass
{
    public string  label       = "Class";
    [Range(0f, 360f)]
    public float   angleDeg    = 0f;
    public ArmSide arm         = ArmSide.Right;
    [Min(0f)]
    public float   distance    = 0.35f;

    public Vector3 GetOffset(Transform avatarTransform)
    {
        float rad = angleDeg * Mathf.Deg2Rad;
        return (Mathf.Cos(rad) * avatarTransform.right
              + Mathf.Sin(rad) * avatarTransform.up) * distance;
    }
}

[RequireComponent(typeof(Animator))]
public class AvatarController : MonoBehaviour
{
    // ── Inspector ────────────────────────────────────────────────
    [Header("WebSocket")]
    public string serverUrl = "ws://localhost:8765";

    [Header("IK Targets")]
    public Transform leftArmTarget;
    public Transform rightArmTarget;

    [Header("IK Parameters")]
    [Range(0f, 1f)] public float ikWeight   = 1f;
    public float lerpSpeed                  = 5f;

    [Header("Direction Classes")]
    [Tooltip("prediction 정수 값을 인덱스로 사용. 기본 8개 = 45° 간격 전방향.")]
    public DirectionClass[] directionClasses;

    [Header("Motion Settings")]
    [Tooltip("목표 위치 유지 시간 (초)")]
    public float holdDuration = 1.5f;
    [Range(0f, 1f)]
    [Tooltip("이 값 미만 신뢰도 예측 무시 (서버 필터와 동기화)")]
    public float minConfidence = 0.6f;

    [Header("Debug UI")]
    public bool showDebugGUI = true;

    // ── C# 이벤트 (BCIExperimentManager 구독용) ─────────────────
    public event Action<BCIMessage>  OnTrialStart;
    public event Action<BCIMessage>  OnPrediction;
    public event Action<BCIMessage>  OnPredictionSkipped;
    public event Action<BCIMessage>  OnSummary;

    // ── 런타임 상태 ─────────────────────────────────────────────
    private WebSocket _ws;
    private Animator  _animator;

    private Vector3 _leftBasePos,  _leftTargetPos;
    private Vector3 _rightBasePos, _rightTargetPos;

    private Coroutine _leftCoroutine;
    private Coroutine _rightCoroutine;

    // Debug 표시용
    private string _lastLabel      = "—";
    private float  _lastConf       = 0f;
    private float  _lastAngle      = -1f;
    private bool   _lastCorrect    = false;
    private int    _trialNo        = 0;
    private int    _remaining      = 0;
    private string _connStatus     = "연결 안됨";
    private float  _avgLatency     = 0f;
    private int    _latencyCount   = 0;

    // E2E latency 측정용
    private float  _avgE2ELatency  = 0f;
    private int    _e2eCount       = 0;

    // Unix epoch 기준 (DateTimeOffset.UtcNow.ToUnixTimeMilliseconds 대신 직접 계산)
    private static readonly DateTime _unixEpoch =
        new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);

    // ── Reset (컴포넌트 첫 추가 시 기본값) ──────────────────────

    private void Reset()
    {
        directionClasses = BuildDefaultClasses();
    }

    private static DirectionClass[] BuildDefaultClasses()
    {
        // 8개 클래스 — 45° 간격
        // 현재 2-class BCI: index 0 = Left MI, index 1 = Right MI
        // 나머지 6개는 향후 다중 클래스 확장용
        string[] labels   = { "Left MI", "Right MI", "45°", "90°", "135°", "180°", "225°", "270°" };
        ArmSide[] sides   = { ArmSide.Left, ArmSide.Right,
                               ArmSide.Both, ArmSide.Both,
                               ArmSide.Both, ArmSide.Both,
                               ArmSide.Both, ArmSide.Both };

        var arr = new DirectionClass[8];
        for (int i = 0; i < 8; i++)
            arr[i] = new DirectionClass
            {
                label    = labels[i],
                angleDeg = i < 2 ? 90f : i * 45f,
                arm      = sides[i],
                distance = 0.35f,
            };
        return arr;
    }

    // ── Unity 생명주기 ───────────────────────────────────────────

    private void Awake()
    {
        _animator = GetComponent<Animator>();
        if (directionClasses == null || directionClasses.Length == 0)
            directionClasses = BuildDefaultClasses();
    }

    private void Start()
    {
        CacheBasePosIfReady();
        ConnectWebSocket();
    }

    private void CacheBasePosIfReady()
    {
        if (leftArmTarget != null)
        {
            _leftBasePos   = leftArmTarget.position;
            _leftTargetPos = _leftBasePos;
        }
        if (rightArmTarget != null)
        {
            _rightBasePos   = rightArmTarget.position;
            _rightTargetPos = _rightBasePos;
        }
    }

    private void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        _ws?.DispatchMessageQueue();
#endif
        if (leftArmTarget != null)
            leftArmTarget.position = Vector3.Lerp(
                leftArmTarget.position, _leftTargetPos, Time.deltaTime * lerpSpeed);
        if (rightArmTarget != null)
            rightArmTarget.position = Vector3.Lerp(
                rightArmTarget.position, _rightTargetPos, Time.deltaTime * lerpSpeed);
    }

    private async void OnDestroy()
    {
        if (_ws != null && _ws.State == WebSocketState.Open)
            await _ws.Close();
    }

    // ── Animator IK ─────────────────────────────────────────────

    private void OnAnimatorIK(int layerIndex)
    {
        if (_animator == null) return;
        if (leftArmTarget != null)
        {
            _animator.SetIKPositionWeight(AvatarIKGoal.LeftHand,  ikWeight);
            _animator.SetIKPosition(AvatarIKGoal.LeftHand, leftArmTarget.position);
        }
        if (rightArmTarget != null)
        {
            _animator.SetIKPositionWeight(AvatarIKGoal.RightHand, ikWeight);
            _animator.SetIKPosition(AvatarIKGoal.RightHand, rightArmTarget.position);
        }
    }

    // ── WebSocket ────────────────────────────────────────────────

    private async void ConnectWebSocket()
    {
        _ws = new WebSocket(serverUrl);
        _ws.OnOpen    += () => { _connStatus = "연결됨"; };
        _ws.OnClose   += (_) => { _connStatus = "연결 끊김"; StartCoroutine(ReconnectAfterDelay(3f)); };
        _ws.OnError   += (e) => { _connStatus = "오류: " + e; };
        _ws.OnMessage += (b) => HandleMessage(System.Text.Encoding.UTF8.GetString(b));
        _connStatus = "연결 중...";
        await _ws.Connect();
    }

    private IEnumerator ReconnectAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);
        if (_ws == null || _ws.State == WebSocketState.Closed)
            ConnectWebSocket();
    }

    // ── 메시지 처리 ──────────────────────────────────────────────

    private void HandleMessage(string json)
    {
        BCIMessage msg;
        try { msg = JsonUtility.FromJson<BCIMessage>(json); }
        catch { return; }

        switch (msg.type)
        {
            case "connected":
                Debug.Log($"[BCI] s{msg.sid:D2} 연결, {msg.n_trials}개 trial, " +
                          $"cue={msg.cue_duration_s}s");
                break;

            case "trial_start":
                _trialNo   = msg.trial_no;
                _remaining = msg.remaining;
                ResetArms();
                OnTrialStart?.Invoke(msg);
                break;

            case "prediction":
                // ── E2E latency 측정: prediction 수신 즉시 Unix 타임스탬프 기록 후 ack 전송
                double unityRecvTs = (DateTime.UtcNow - _unixEpoch).TotalSeconds;
                SendLatencyAck(msg.trial_no, unityRecvTs);

                // server_ts가 유효하면 로컬 E2E 계산 (WS transport latency 기준)
                if (msg.server_ts > 0f)
                {
                    float wsTravelMs = (float)((unityRecvTs - msg.server_ts) * 1000.0);
                    if (wsTravelMs > 0f && wsTravelMs < 500f)  // 이상값 제외
                    {
                        _e2eCount++;
                        _avgE2ELatency += (wsTravelMs - _avgE2ELatency) / _e2eCount;
                    }
                }

                HandlePrediction(msg);
                OnPrediction?.Invoke(msg);
                break;

            case "prediction_skipped":
                _lastLabel = $"{msg.label} (SKIP conf={msg.confidence:F2})";
                _lastConf  = msg.confidence;
                OnPredictionSkipped?.Invoke(msg);
                break;

            case "summary":
                Debug.Log($"[BCI] 요약 {msg.n_correct}/{msg.n_trials} ({msg.accuracy:P1}) " +
                          $"avg={msg.avg_latency_ms:F1}ms");
                OnSummary?.Invoke(msg);
                break;

            case "ack":
                Debug.Log("[BCI] ack: " + msg.cmd);
                break;
        }
    }

    private void HandlePrediction(BCIMessage msg)
    {
        if (msg.confidence < minConfidence) return;

        int idx = msg.prediction;
        if (idx < 0 || idx >= directionClasses.Length)
        {
            Debug.LogWarning($"[BCI] prediction={idx} 범위 초과 (길이={directionClasses.Length})");
            return;
        }

        DirectionClass cls = directionClasses[idx];

        _lastLabel   = $"{cls.label}  {cls.angleDeg}°  ({cls.arm})";
        _lastConf    = msg.confidence;
        _lastAngle   = cls.angleDeg;
        _lastCorrect = msg.correct;
        _trialNo     = msg.trial_no;
        _remaining   = msg.remaining;

        _latencyCount++;
        _avgLatency += (msg.latency_ms - _avgLatency) / _latencyCount;

        Vector3 offset = cls.GetOffset(transform);
        MoveArmByClass(cls.arm, offset);

        Debug.Log($"[BCI] Trial {msg.trial_no} | class={idx} '{cls.label}' " +
                  $"conf={msg.confidence:F3} [{msg.latency_ms:F1}ms] " +
                  $"{(msg.correct ? "✓" : "✗")}");
    }

    // ── 팔 동작 ─────────────────────────────────────────────────

    private void MoveArmByClass(ArmSide side, Vector3 offset)
    {
        if (side == ArmSide.Left || side == ArmSide.Both)
        {
            if (_leftCoroutine != null) StopCoroutine(_leftCoroutine);
            _leftCoroutine = StartCoroutine(ArmMotion(isLeft: true, offset));
        }
        if (side == ArmSide.Right || side == ArmSide.Both)
        {
            if (_rightCoroutine != null) StopCoroutine(_rightCoroutine);
            _rightCoroutine = StartCoroutine(ArmMotion(isLeft: false, offset));
        }
    }

    private IEnumerator ArmMotion(bool isLeft, Vector3 offset)
    {
        if (isLeft)
        {
            _leftTargetPos = _leftBasePos + offset;
            yield return new WaitForSeconds(holdDuration);
            _leftTargetPos = _leftBasePos;
        }
        else
        {
            _rightTargetPos = _rightBasePos + offset;
            yield return new WaitForSeconds(holdDuration);
            _rightTargetPos = _rightBasePos;
        }
    }

    private void ResetArms()
    {
        if (leftArmTarget  != null) _leftTargetPos  = _leftBasePos;
        if (rightArmTarget != null) _rightTargetPos = _rightBasePos;
    }

    // ── 서버 명령 ────────────────────────────────────────────────

    public async void SendPause()   { if (_ws?.State == WebSocketState.Open) await _ws.SendText("{\"cmd\":\"pause\"}"); }
    public async void SendResume()  { if (_ws?.State == WebSocketState.Open) await _ws.SendText("{\"cmd\":\"resume\"}"); }
    public async void SendReset()   { if (_ws?.State == WebSocketState.Open) await _ws.SendText("{\"cmd\":\"reset\"}"); }
    public async void SendStatus()  { if (_ws?.State == WebSocketState.Open) await _ws.SendText("{\"cmd\":\"status\"}"); }
    public async void SendRawCmd(string json) { if (_ws?.State == WebSocketState.Open) await _ws.SendText(json); }

    /// <summary>
    /// E2E latency 측정용 ack — prediction 수신 직후 호출.
    /// Python 서버의 _handle_cmd(latency_ack) 핸들러와 쌍을 이룸.
    /// e2e_ms = (unity_recv_ts - server_ts) * 1000  [ms]
    /// </summary>
    private async void SendLatencyAck(int trialNo, double unityRecvTs)
    {
        if (_ws == null || _ws.State != WebSocketState.Open) return;
        // JSON 직렬화: JsonUtility는 double 미지원 → 문자열 포맷 직접 작성
        string json = $"{{\"cmd\":\"latency_ack\",\"trial_no\":{trialNo}," +
                      $"\"unity_recv_ts\":{unityRecvTs:F6}}}";
        await _ws.SendText(json);
    }

    // ── Debug GUI ────────────────────────────────────────────────

    private void OnGUI()
    {
        if (!showDebugGUI) return;

        var boxStyle = new GUIStyle(GUI.skin.box) { fontSize = 13, alignment = TextAnchor.UpperLeft };

        string e2eStr = _e2eCount > 0
            ? $"{_avgE2ELatency:F1} ms WS ({_e2eCount}개)"
            : "측정 중...";

        string info =
            $"[BCI Debug]\n" +
            $"서버:   {serverUrl}\n" +
            $"상태:   {_connStatus}\n" +
            $"Trial:  {_trialNo}  남은: {_remaining}\n" +
            $"예측:   {_lastLabel}\n" +
            $"신뢰도: {_lastConf:F3}  정답: {(_lastCorrect ? "✅" : "❌")}\n" +
            $"추론:   {_avgLatency:F1} ms (server-side avg)\n" +
            $"WS전송: {e2eStr}";

        GUI.Box(new Rect(10, 10, 310, 170), info, boxStyle);

        DrawCompass(310, 10, 130);

        if (GUI.Button(new Rect(10,  175,  85, 28), "Pause"))  SendPause();
        if (GUI.Button(new Rect(100, 175,  85, 28), "Resume")) SendResume();
        if (GUI.Button(new Rect(190, 175,  85, 28), "Reset"))  SendReset();
    }

    private void DrawCompass(float ox, float oy, float size)
    {
        if (directionClasses == null) return;

        var style = new GUIStyle(GUI.skin.label) { fontSize = 10, alignment = TextAnchor.MiddleCenter };
        float cx = ox + size * 0.5f;
        float cy = oy + size * 0.5f;
        float r  = size * 0.42f;

        GUI.Box(new Rect(ox, oy, size, size), "방향 클래스");

        for (int i = 0; i < directionClasses.Length; i++)
        {
            var cls = directionClasses[i];
            float rad = cls.angleDeg * Mathf.Deg2Rad;
            float px = cx + Mathf.Cos(rad) * r - 14f;
            float py = cy - Mathf.Sin(rad) * r - 10f;

            bool isActive = Mathf.Abs(cls.angleDeg - _lastAngle) < 1f;
            GUI.color = isActive ? Color.yellow : Color.white;
            GUI.Label(new Rect(px, py, 28f, 20f), $"[{i}]\n{cls.angleDeg}°", style);
            GUI.color = Color.white;
        }
    }

    // ── JSON 메시지 구조체 ────────────────────────────────────────

    [Serializable]
    public class BCIMessage
    {
        public string type;
        public int    sid;
        public int    n_trials;
        public float  interval_s;
        public float  cue_duration_s;
        public int    trial_no;
        public int    prediction;
        public string label;
        public float  confidence;
        public float  prob_left;
        public float  prob_right;
        public int    true_label;
        public string true_label_str;
        public bool   correct;
        public float  latency_ms;
        public int    remaining;
        public int    n_correct;
        public float  accuracy;
        public float  avg_confidence;
        public float  avg_latency_ms;
        public string cmd;
        public bool   paused;
        public int    n_clients;
        public float  server_ts;     // prediction 전송 시각 (Unix, e2e latency 계산용)
    }
}
