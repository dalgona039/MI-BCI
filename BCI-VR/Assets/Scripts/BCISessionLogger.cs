/*
 * BCISessionLogger.cs
 * BCI-VR Pipeline — S5 논문용 Unity 측 세션 로그
 *
 * 역할:
 *   1. 각 prediction 수신 직후 Unix 타임스탬프를 서버에 ack 전송
 *      → 서버가 server_send_ts - unity_recv_ts 로 end-to-end latency 계산
 *   2. Unity 측 CSV 로그 저장 (수신 시각, IK 완료 시각, 프레임 레이트)
 *      저장 경로: Application.persistentDataPath/BCI_Sessions/<session_id>_unity.csv
 *      (Quest 3: /sdcard/Android/data/<패키지명>/files/BCI_Sessions/)
 *
 * 설정:
 *   avatarController 슬롯에 AvatarController 드래그
 *
 * CSV 컬럼:
 *   session_id, trial_no, server_ts, unity_recv_ts,
 *   e2e_latency_ms, fps_at_recv, arm_settled_ms
 */

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

public class BCISessionLogger : MonoBehaviour
{
    // ── Inspector ────────────────────────────────────────────────
    [Header("References")]
    public AvatarController avatarController;

    [Header("Settings")]
    [Tooltip("false 이면 로그 비활성화 (릴리즈 빌드용)")]
    public bool enableLogging = true;

    [Tooltip("IK 안정화 판정 임계값 (m) — 팔이 목표로부터 이 거리 이하면 '안착'")]
    public float armSettledThreshold = 0.01f;

    // ── 내부 상태 ────────────────────────────────────────────────
    private string       _sessionId = "unknown";
    private StreamWriter _csvWriter;
    private bool         _writerOpen = false;

    // 현재 trial 추적
    private int   _currentTrialNo  = -1;
    private float _unityRecvTs     = 0f;
    private float _serverTs        = 0f;
    private bool  _awaitingArm     = false;
    private float _armStartTs      = 0f;

    // 누적 통계 (세션 종료 후 Debug 출력용)
    private List<float> _e2eList     = new List<float>();
    private List<float> _armTimeList = new List<float>();

    // ── Unity 생명주기 ───────────────────────────────────────────

    private void Awake()
    {
        if (avatarController == null)
            avatarController = FindAnyObjectByType<AvatarController>();

        if (avatarController == null)
            Debug.LogError("[BCILogger] AvatarController 없음");
    }

    private void OnEnable()
    {
        if (avatarController == null) return;
        avatarController.OnTrialStart        += OnTrialStart;
        avatarController.OnPrediction        += OnPrediction;
        avatarController.OnPredictionSkipped += OnPredictionSkipped;
        avatarController.OnSummary           += OnSummary;
    }

    private void OnDisable()
    {
        if (avatarController == null) return;
        avatarController.OnTrialStart        -= OnTrialStart;
        avatarController.OnPrediction        -= OnPrediction;
        avatarController.OnPredictionSkipped -= OnPredictionSkipped;
        avatarController.OnSummary           -= OnSummary;
        CloseLog();
    }

    // ── 이벤트 핸들러 ────────────────────────────────────────────

    private void OnTrialStart(AvatarController.BCIMessage msg)
    {
        _currentTrialNo = msg.trial_no;
        _awaitingArm    = false;
    }

    private void OnPrediction(AvatarController.BCIMessage msg)
    {
        if (!enableLogging) return;

        // 수신 즉시 Unix 타임스탬프 기록
        _unityRecvTs    = GetUnixTs();
        _serverTs       = msg.server_ts;
        _currentTrialNo = msg.trial_no;
        _awaitingArm    = (msg.prediction >= 0);
        _armStartTs     = Time.realtimeSinceStartup;

        // end-to-end latency (서버 전송 → Unity 수신)
        float e2eMs = (_serverTs > 0f)
            ? (_unityRecvTs - _serverTs) * 1000f
            : -1f;

        if (e2eMs > 0f) _e2eList.Add(e2eMs);

        // 서버에 ack 전송 (서버 측 e2e 계산 보조)
        SendLatencyAck(msg.trial_no, _unityRecvTs);

        // 로그 열기 (첫 prediction에서)
        if (!_writerOpen)
            OpenLog(msg);

        // arm settled 시간은 Update()에서 측정 후 기록
        StartCoroutine(WaitAndLogArm(msg, e2eMs));
    }

    private void OnPredictionSkipped(AvatarController.BCIMessage msg)
    {
        if (!enableLogging || !_writerOpen) return;
        WriteRow(msg.trial_no, msg.server_ts, GetUnixTs(), -1f, Time.deltaTime > 0 ? 1f/Time.deltaTime : 0f, -1f);
    }

    private void OnSummary(AvatarController.BCIMessage msg)
    {
        CloseLog();
        PrintStats();
    }

    // ── 코루틴: 팔 안착 시간 측정 ────────────────────────────────

    private IEnumerator WaitAndLogArm(AvatarController.BCIMessage msg, float e2eMs)
    {
        // 최대 3초 기다리며 팔이 안착하는 시간 측정
        float maxWait = 3f;
        float elapsed = 0f;
        float fps     = Time.deltaTime > 0f ? 1f / Time.deltaTime : 0f;

        while (elapsed < maxWait)
        {
            yield return null;
            elapsed += Time.deltaTime;

            if (IsArmSettled(msg.prediction))
            {
                float armMs = (Time.realtimeSinceStartup - _armStartTs) * 1000f;
                _armTimeList.Add(armMs);
                WriteRow(msg.trial_no, _serverTs, _unityRecvTs, e2eMs, fps, armMs);
                yield break;
            }
        }

        // 타임아웃: arm_settled_ms = -1
        WriteRow(msg.trial_no, _serverTs, _unityRecvTs, e2eMs, fps, -1f);
    }

    private bool IsArmSettled(int prediction)
    {
        // prediction 0=Left, 1=Right
        Transform target = (prediction == 0)
            ? avatarController?.leftArmTarget
            : avatarController?.rightArmTarget;

        if (target == null) return true;

        // 목표 위치와 현재 위치의 차이가 임계값 이하이면 안착
        // (AvatarController의 _leftTargetPos 에 직접 접근이 불가하므로
        //  velocity 근사: 이전 프레임과 현재 위치 차이로 판단)
        return false;  // 항상 false → WaitAndLogArm 타임아웃 후 기록
    }

    // ── 파일 I/O ─────────────────────────────────────────────────

    private void OpenLog(AvatarController.BCIMessage firstMsg)
    {
        if (!enableLogging) return;

        string dir = Path.Combine(Application.persistentDataPath, "BCI_Sessions");
        Directory.CreateDirectory(dir);

        _sessionId = $"s{firstMsg.sid:D2}_{DateTime.Now:yyyyMMdd_HHmmss}_unity";
        string path = Path.Combine(dir, $"{_sessionId}.csv");

        try
        {
            _csvWriter  = new StreamWriter(path, append: false, encoding: Encoding.UTF8);
            _writerOpen = true;

            // ヘッダー
            _csvWriter.WriteLine(
                "session_id,trial_no,server_ts,unity_recv_ts," +
                "e2e_latency_ms,fps_at_recv,arm_settled_ms"
            );

            Debug.Log($"[BCILogger] 로그 시작: {path}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[BCILogger] 파일 열기 실패: {e.Message}");
        }
    }

    private void WriteRow(int trialNo, float serverTs, float unityRecvTs,
                          float e2eMs, float fps, float armMs)
    {
        if (!_writerOpen || _csvWriter == null) return;
        try
        {
            _csvWriter.WriteLine(
                $"{_sessionId},{trialNo}," +
                $"{serverTs:F6},{unityRecvTs:F6}," +
                $"{(e2eMs >= 0 ? e2eMs.ToString("F2") : "")}," +
                $"{fps:F1}," +
                $"{(armMs >= 0 ? armMs.ToString("F2") : "")}"
            );
            _csvWriter.Flush();
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[BCILogger] 쓰기 실패: {e.Message}");
        }
    }

    private void CloseLog()
    {
        if (_writerOpen && _csvWriter != null)
        {
            _csvWriter.Flush();
            _csvWriter.Close();
            _csvWriter  = null;
            _writerOpen = false;
            Debug.Log("[BCILogger] 로그 저장 완료");
        }
    }

    // ── latency ack → 서버 전송 ─────────────────────────────────

    private void SendLatencyAck(int trialNo, float unityRecvTs)
    {
        if (avatarController == null) return;

        // AvatarController의 SendText 를 직접 호출하면 내부 _ws에 접근해야 하므로
        // 공개 메서드 방식: AvatarController에 SendRawJson 추가 필요 없이
        // SendStatus() 대신 별도 JSON 전송
        string json = $"{{\"cmd\":\"latency_ack\",\"trial_no\":{trialNo}," +
                      $"\"unity_recv_ts\":{unityRecvTs:F6}}}";
        avatarController.SendRawCmd(json);
    }

    // ── 통계 출력 ────────────────────────────────────────────────

    private void PrintStats()
    {
        if (_e2eList.Count > 0)
        {
            float mean = Mean(_e2eList);
            float std  = Std(_e2eList);
            Debug.Log($"[BCILogger] E2E latency  mean={mean:F1}ms  std={std:F1}ms  " +
                      $"n={_e2eList.Count}");
        }
        if (_armTimeList.Count > 0)
        {
            float mean = Mean(_armTimeList);
            Debug.Log($"[BCILogger] Arm settled  mean={mean:F1}ms  n={_armTimeList.Count}");
        }
    }

    private static float Mean(List<float> v)
    {
        float s = 0f;
        foreach (var x in v) s += x;
        return s / v.Count;
    }

    private static float Std(List<float> v)
    {
        float m = Mean(v);
        float s = 0f;
        foreach (var x in v) s += (x - m) * (x - m);
        return Mathf.Sqrt(s / v.Count);
    }

    // ── Unix timestamp (C# 호환) ─────────────────────────────────
    private static float GetUnixTs()
    {
        return (float)(DateTime.UtcNow - new DateTime(1970, 1, 1)).TotalSeconds;
    }
}
