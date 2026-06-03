/*
 * BCIExperimentManager.cs
 * BCI-VR Pipeline — S5 VR Experiment UI
 *
 * 역할:
 *   AvatarController 의 이벤트를 구독하여 Meta Quest 3 VR 공간에
 *   실험 프로토콜 UI(큐 화살표, 카운트다운, 피드백, 최종 요약)를 표시.
 *
 * 세팅 방법:
 *   1. 빈 GameObject에 BCIExperimentManager 추가
 *   2. avatarController 슬롯에 AvatarController 컴포넌트 드래그
 *   3. cuePanelRoot: WorldSpace Canvas 루트 (플레이어 앞 1~2m 위치)
 *   4. (선택) experimentResultsPanel: 세션 종료 시 요약 패널
 *
 * 의존성:
 *   - TextMeshPro (Unity Package Manager에서 설치)
 *   - AvatarController.cs (같은 씬)
 *
 * 실험 흐름:
 *   trial_start 수신
 *     → 2초 카운트다운 (3, 2, 1, GO!)
 *     → 큐 표시 (Left ← / Right →)
 *     → cue_duration 동안 큐 유지 (MI 수행)
 *   prediction 수신
 *     → 결과 표시 (팔 이동 + 색상 피드백)
 *     → holdDuration 후 리셋
 *   summary 수신
 *     → 최종 정확도 패널 표시
 */

using System;
using System.Collections;
using UnityEngine;
using TMPro;

[RequireComponent(typeof(Transform))]
public class BCIExperimentManager : MonoBehaviour
{
    // ── Inspector ────────────────────────────────────────────────
    [Header("References")]
    [Tooltip("같은 씬의 AvatarController 드래그")]
    public AvatarController avatarController;

    [Header("VR UI Panels (World Space Canvas 하위)")]
    [Tooltip("큐/카운트다운/피드백을 표시할 WorldSpace Canvas 루트")]
    public GameObject cuePanelRoot;

    [Tooltip("세션 종료 요약 패널 (선택, 없으면 큐 패널 재사용)")]
    public GameObject experimentResultsPanel;

    [Header("Text Fields (TMP)")]
    [Tooltip("큰 중앙 큐 텍스트 (←  / →  / Ready)")]
    public TextMeshProUGUI cueText;

    [Tooltip("상태 줄 (Trial X / Y  |  Acc: 75%)")]
    public TextMeshProUGUI statusText;

    [Tooltip("예측 결과 텍스트 (Left MI ✓ / Right MI ✗)")]
    public TextMeshProUGUI resultText;

    [Tooltip("신뢰도 바 배경 이미지 (없으면 텍스트로 대체)")]
    public UnityEngine.UI.Image confidenceBar;

    [Header("Timing")]
    [Tooltip("카운트다운 표시 시간 (초) — 0이면 생략")]
    public float countdownDuration = 2f;

    [Tooltip("예측 결과 표시 유지 시간 (초)")]
    public float resultHoldDuration = 1.5f;

    [Header("Colors")]
    public Color colorCorrect   = new Color(0.2f, 0.9f, 0.3f);   // 초록
    public Color colorIncorrect = new Color(0.9f, 0.2f, 0.2f);   // 빨강
    public Color colorSkipped   = new Color(0.8f, 0.8f, 0.2f);   // 노랑
    public Color colorCueLeft   = new Color(0.3f, 0.6f, 1.0f);   // 파랑
    public Color colorCueRight  = new Color(1.0f, 0.5f, 0.2f);   // 주황

    // ── 내부 상태 ────────────────────────────────────────────────
    private int   _totalTrials  = 0;
    private int   _doneTrials   = 0;
    private int   _correctCount = 0;
    private float _cueDuration  = 2.0f;

    private Coroutine _uiCoroutine;

    // ── Unity 생명주기 ───────────────────────────────────────────

    private void Awake()
    {
        // avatarController 자동 탐색
        if (avatarController == null)
            avatarController = FindAnyObjectByType<AvatarController>();

        if (avatarController == null)
            Debug.LogError("[BCIExpMgr] AvatarController를 찾을 수 없습니다.");
    }

    private void OnEnable()
    {
        if (avatarController == null) return;
        avatarController.OnTrialStart         += HandleTrialStart;
        avatarController.OnPrediction         += HandlePrediction;
        avatarController.OnPredictionSkipped  += HandleSkipped;
        avatarController.OnSummary            += HandleSummary;
    }

    private void OnDisable()
    {
        if (avatarController == null) return;
        avatarController.OnTrialStart         -= HandleTrialStart;
        avatarController.OnPrediction         -= HandlePrediction;
        avatarController.OnPredictionSkipped  -= HandleSkipped;
        avatarController.OnSummary            -= HandleSummary;
    }

    private void Start()
    {
        SetCuePanelActive(false);
        SetResultsPanelActive(false);
    }

    // ── 이벤트 핸들러 ────────────────────────────────────────────

    private void HandleTrialStart(AvatarController.BCIMessage msg)
    {
        _totalTrials = msg.n_trials;
        _cueDuration = msg.cue_duration_s > 0 ? msg.cue_duration_s : 2.0f;

        if (_uiCoroutine != null) StopCoroutine(_uiCoroutine);
        _uiCoroutine = StartCoroutine(TrialStartSequence(msg));
    }

    private void HandlePrediction(AvatarController.BCIMessage msg)
    {
        _doneTrials++;
        if (msg.correct) _correctCount++;

        if (_uiCoroutine != null) StopCoroutine(_uiCoroutine);
        _uiCoroutine = StartCoroutine(PredictionFeedback(msg));
    }

    private void HandleSkipped(AvatarController.BCIMessage msg)
    {
        _doneTrials++;
        ShowResult(
            $"SKIP  conf={msg.confidence:F2}",
            colorSkipped
        );
        UpdateStatus();
    }

    private void HandleSummary(AvatarController.BCIMessage msg)
    {
        if (_uiCoroutine != null) StopCoroutine(_uiCoroutine);
        StartCoroutine(ShowSummary(msg));
    }

    // ── UI 시퀀스 코루틴 ─────────────────────────────────────────

    private IEnumerator TrialStartSequence(AvatarController.BCIMessage msg)
    {
        SetCuePanelActive(true);
        SetResultsPanelActive(false);

        // ① 카운트다운
        if (countdownDuration > 0f)
        {
            float step = countdownDuration / 3f;
            string[] ticks = { "3", "2", "1" };
            foreach (string t in ticks)
            {
                SetCueText(t, Color.white);
                yield return new WaitForSeconds(step);
            }
        }

        // ② 큐 표시 (어느 쪽 MI를 수행할지)
        bool isLeft    = (msg.true_label == 0);
        string cueStr  = isLeft ? "←  LEFT" : "RIGHT  →";
        Color  cueCol  = isLeft ? colorCueLeft : colorCueRight;
        SetCueText(cueStr, cueCol);

        UpdateStatus();

        // ③ 신뢰도 바 리셋
        SetConfidenceBar(0f, Color.white);
    }

    private IEnumerator PredictionFeedback(AvatarController.BCIMessage msg)
    {
        bool  correct  = msg.correct;
        Color feedCol  = correct ? colorCorrect : colorIncorrect;
        string mark    = correct ? "✓" : "✗";
        string predStr = msg.prediction == 0 ? "← Left MI" : "Right MI →";

        SetCueText(predStr, feedCol);
        ShowResult($"{predStr}  {mark}", feedCol);
        SetConfidenceBar(msg.confidence, feedCol);
        UpdateStatus();

        yield return new WaitForSeconds(resultHoldDuration);

        // 결과 지우고 준비 상태로
        SetCueText("", Color.white);
        ShowResult("", Color.white);
        SetCuePanelActive(false);
    }

    private IEnumerator ShowSummary(AvatarController.BCIMessage msg)
    {
        SetCuePanelActive(false);
        SetResultsPanelActive(true);

        // 요약 패널이 별도 없으면 큐 패널 재활용
        if (experimentResultsPanel == null)
            SetCuePanelActive(true);

        string summaryStr =
            $"실험 완료\n\n" +
            $"{msg.n_correct} / {msg.n_trials}\n" +
            $"정확도: {msg.accuracy:P1}\n" +
            $"평균 지연: {msg.avg_latency_ms:F1} ms";

        SetCueText(summaryStr, colorCorrect);

        if (statusText != null)
            statusText.text = $"완료  |  κ = {CalcKappa(msg.n_correct, msg.n_trials):F3}";

        yield break;
    }

    // ── UI 헬퍼 ─────────────────────────────────────────────────

    private void SetCueText(string text, Color color)
    {
        if (cueText == null) return;
        cueText.text  = text;
        cueText.color = color;
    }

    private void ShowResult(string text, Color color)
    {
        if (resultText == null) return;
        resultText.text  = text;
        resultText.color = color;
    }

    private void SetConfidenceBar(float value, Color color)
    {
        if (confidenceBar == null) return;
        confidenceBar.fillAmount = Mathf.Clamp01(value);
        confidenceBar.color      = color;
    }

    private void UpdateStatus()
    {
        if (statusText == null) return;
        float acc = _doneTrials > 0 ? (float)_correctCount / _doneTrials : 0f;
        statusText.text =
            $"Trial {_doneTrials + 1} / {(_totalTrials > 0 ? _totalTrials.ToString() : "?")}  " +
            $"|  Acc: {acc:P0}";
    }

    private void SetCuePanelActive(bool active)
    {
        if (cuePanelRoot != null)
            cuePanelRoot.SetActive(active);
    }

    private void SetResultsPanelActive(bool active)
    {
        if (experimentResultsPanel != null)
            experimentResultsPanel.SetActive(active);
    }

    // Cohen's κ 근사 (2-class balanced)
    private static float CalcKappa(int correct, int total)
    {
        if (total == 0) return 0f;
        float po = (float)correct / total;
        float pe = 0.5f;  // 균등 2-class
        return pe >= 1f ? 0f : (po - pe) / (1f - pe);
    }
}
