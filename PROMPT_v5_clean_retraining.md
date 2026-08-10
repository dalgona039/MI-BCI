# Claude Code 작업 지시 — v5-clean 재학습 (움직임 오염 trial 배제)

이 파일 전체를 Claude Code에 붙여넣으세요. 저장소 루트(`/Volumes/a3122a1/MI-BCI`)에서 실행합니다.

---

## 0. 배경 — 왜 이걸 돌리는가

JNE 투고 준비 중 원본 데이터 검증에서 확인된 사실입니다.

GigaDB 원본 `.mat`의 `eeg.bad_trial_indices`에는 데이터셋 제작자가 표시한 두 종류의 불량 trial 목록이 들어 있습니다. Cho et al. 2017 (GigaScience 6:gix034) 기준:

- `bad_trial_idx_voltage` — 전압 크기 기준
- `bad_trial_idx_mi` — **EMG 활동과 상관된 trial**. 판정: EMG 0.5 Hz 고역통과 후 **실제 손 움직임 EMG에서 도출한 템플릿**과 상관 > 0.8, Bonferroni 보정 p < 0.01. 즉 **참가자가 실제로 움직였을 가능성이 높은 trial**

**현재 전처리 파이프라인(`notebooks/S1_S2_Preprocessing_MemberA.ipynb`)은 이 필드를 전혀 읽지 않습니다.** 확인된 규모:

| 항목 | 값 |
|---|---|
| 움직임(EMG) 기준 플래그 | **724 / 10,320 trial (7.02%)** |
| 전압 기준 플래그 | 30 |
| 플래그 ≥1건 피험자 | 27 / 52 |
| 플래그 ≥10% 피험자 | s34(92.0%), s29(89.5%), s20(58.5%), s30(39.0%), s19(28.5%), s12(13.0%) |

그리고 이 오염률이 sEMG 관련 결과 전부를 예측합니다 (n=51):

| 상관 | ρ | p |
|---|---|---|
| 오염률 vs sEMG-only 정확도 | +0.426 | 0.0018 |
| 오염률 vs w_sEMG (attention 가중치) | +0.314 | 0.025 |
| 오염률 vs fusion 이득 (Fusion − EEG-only) | +0.403 | 0.0034 |
| 오염률 vs EEG-only 정확도 | −0.259 | 0.067 (반대 방향) |

피험자 단위로 오염자를 배제하면 "게이트가 sEMG 판별력을 추적한다"는 핵심 결과가 사라집니다 (ρ: +0.512 → +0.181, ns). 다만 **배제하면 sEMG가 유용한 피험자 자체가 사라지므로 효과 소멸과 검정력 저하를 구분할 수 없습니다.**

**따라서 결정적 분석은 피험자가 아니라 trial 단위 배제 후 재학습입니다. 그게 이 작업입니다.**

이미 생성된 플래그 집계: `BCI_Research/results/movement_flagged_trials_52.csv`

---

## 1. 목표

`bad_trial_idx_mi` ∪ `bad_trial_idx_voltage`에 해당하는 trial을 **학습·검증·평가 전 단계에서 제거**한 뒤, 기존 v4와 **동일한 프로토콜**로 3개 조건(`eeg_only`, `emg_only`, `fusion`)을 52-fold LOSO 재학습하고, v4 결과와 나란히 비교할 수 있는 산출물을 만듭니다.

**v4 결과 파일은 절대 덮어쓰지 마세요.** 모든 출력은 `BCI_Research/results/ablation_v5_clean/` 아래에 새로 만듭니다.

---

## 2. 검증된 데이터 구조 (이미 확인함 — 재확인만 하고 신뢰해도 됨)

### 2.1 전처리된 h5
```
BCI_Research/preprocessed/member_A/sub-NN_member_A.h5   (NN = 01..52)
  eeg/epochs        (N, 64, 2304)  float32   ← 피험자별 z-score 완료 상태
  eeg/mu_epochs     (N, 64, 2304)  float32
  eeg/beta_epochs   (N, 64, 2304)  float32
  emg/epochs        (N, 4, 2304)   float32   ← RMS envelope 전, 원 시계열
  labels            (N,)           int8      값 1(left) / 2(right)
  metadata/norm_mean (64,), metadata/norm_std (64,)
```
- N = 200 (49명) 또는 240 (3명)
- **에폭 순서: 앞 절반이 전부 left(label=1), 뒤 절반이 전부 right(label=2).** `sub-01`에서 전환 지점 인덱스 99로 확인함
- **`summary_member_A_v4.csv`의 `rejected_epochs`가 52명 전원 0** → 전처리 단계에서 에폭이 하나도 버려지지 않았으므로, **h5의 k번째 에폭 = 원본 .mat의 k번째 trial**로 1:1 대응됩니다

### 2.2 원본 .mat
```
GigaDB_100295/sNN.mat
  eeg.bad_trial_indices.bad_trial_idx_mi        ← 2-element cell {left[], right[]}
  eeg.bad_trial_indices.bad_trial_idx_voltage   ← 2-element cell {left[], right[]}
  eeg.n_imagery_trials                          ← 클래스당 trial 수 (100 또는 120)
```
- 인덱스는 **MATLAB 1-based, 클래스 내부 기준**
- **⚠️ `GigaDB_100295/`에 s06.mat이 없습니다 (51개만 존재).** 3절 D1 참조

### 2.3 매핑 규칙
```python
n = int(eeg.n_imagery_trials)          # 클래스당 trial 수
# left  trial i (1-based) → h5 epoch index  i - 1
# right trial j (1-based) → h5 epoch index  n + j - 1
```
스크립트 시작 시 반드시 assert로 검증하세요:
```python
assert len(labels) == 2 * n
assert (labels[:n] == 1).all() and (labels[n:] == 2).all()
assert all(1 <= idx <= n for idx in flagged_left + flagged_right)
```

### 2.4 기존 학습 코드
`src/ablation_study.py` — 그대로 재사용합니다. 확인된 구조:
- `MODEL_CLASSES = {'eeg_only': EEGOnlyModel, 'emg_only': EMGOnlyModel, 'fusion': FusionModel}`
- 세 조건 모두 독립 아키텍처. EEG-only는 sEMG 인코더·fusion 층 자체가 없음 (입력 zeroing 아님)
- `calc_itr(accuracy, n_classes=2, trial_duration_sec=4.5)`
- `run_one_fold`에 `if use_existing_ckpt and model_type == "fusion":` 분기 존재 → **v5에서는 이 경로를 반드시 끄세요.** fusion도 새로 학습해야 합니다
- 하이퍼파라미터: Adam lr=1e-3, wd=1e-4, cosine annealing T_max=100, max 100 epoch, early stopping patience 15, batch 64, dropout EEGNet 0.5 / LSTM·clf 0.3, seed 42

---

## 3. 시작 전에 반드시 처리할 결정 사항

### D1. s06 원본 파일 누락
`GigaDB_100295/s06.mat`이 없습니다 (`cache/`에도 없음). 선택:
- **(권장)** GigaDB에서 s06.mat을 내려받아 플래그를 확보한 뒤 52명 전체로 진행
- 불가능하면 s06을 **분석에서 제외**하고 n=51로 진행. "플래그 0건"으로 가정하지 마세요 — 근거 없는 가정입니다

어느 쪽이든 최종 리포트에 명시하세요.

### D2. 배제 정책
**주 분석(primary)**: 플래그된 trial을 **학습 fold와 평가 fold 양쪽에서 모두 제거**. 이유는 오염이 학습된 표현 자체를 오염시켰을 가능성을 배제해야 하기 때문입니다.

**부 분석(secondary)**: 학습은 전체 데이터로, **평가만 clean trial로**. 이 대조는 "표현이 오염된 것인지, 평가가 오염된 것인지"를 분리해 줍니다. 계산 비용이 거의 0(기존 v4 체크포인트 재평가)이므로 **반드시 함께 수행하세요.**

**배제 기준**: `bad_trial_idx_mi ∪ bad_trial_idx_voltage` (합집합). `mi`만 쓴 변형도 부록으로 남기면 좋지만 필수는 아닙니다.

### D3. 클래스 불균형
trial 제거 후 피험자별 left/right 개수가 달라집니다 (s34는 left/right가 심하게 치우침). 처리:
- 제거 후 **클래스당 30 trial 미만**인 피험자는 평가에서 제외하고 그 사실을 명시
- κ는 불균형에 강건하므로 **κ를 주 지표로 보고**하고 accuracy는 병기
- 균형 맞추기용 언더샘플링은 **하지 마세요** (정보 손실 + 임의성)

### D4. seed
v4는 seed 42 단일 실행이었고, 이것이 리뷰의 지적 사항 중 하나입니다. v5는 **seed 42, 1337, 2024 최소 3개**로 돌리고 mean ± SD를 보고하세요. 시간이 부족하면 fusion만이라도 3 seed.

---

## 4. 수행할 작업

### T1. 플래그 추출 스크립트 — `src/extract_bad_trials.py` (신규)
`GigaDB_100295/*.mat`을 순회하여 피험자별 배제 에폭 인덱스를 산출, 다음을 저장:
```
BCI_Research/results/ablation_v5_clean/excluded_epochs.json
  { "1": {"n_per_class": 100, "n_total": 200,
          "excl_idx": [12, 47, 133], "n_excl_mi": 2, "n_excl_voltage": 1,
          "n_left_kept": 99, "n_right_kept": 98}, ... }
```
2.3의 assert를 모두 포함하고, 하나라도 실패하면 즉시 중단하세요. 조용히 넘어가면 안 됩니다.

### T2. 학습 스크립트 — `src/ablation_study_v5.py` (신규, `ablation_study.py` 복사 후 수정)
변경점은 최소한으로:
1. `BCIDataset.__init__`에 `exclude_idx` 인자 추가 → 로드 직후 해당 에폭 제거
2. `use_existing_ckpt` 경로 **비활성화** (fusion도 새로 학습)
3. 출력 경로를 `results/ablation_v5_clean/`로 변경
4. `--seed` 인자 추가
5. **per-trial 로그 저장 추가** — 아래 T3 참조

모델 정의·하이퍼파라미터·LOSO 분할·early stopping은 **절대 건드리지 마세요.** v4와의 유일한 차이가 데이터여야 결과를 해석할 수 있습니다.

### T3. per-trial 출력 저장 (v4에 없던 것 — 중요)
v4는 피험자별 집계만 저장해서 trial 단위 주장을 아무도 재현할 수 없었습니다. v5에서는 fold별 평가 시 다음을 남기세요:
```
results/ablation_v5_clean/per_trial/{condition}_seed{S}_s{NN}.csv
  epoch_idx, true_label, pred_label, correct, prob_left, prob_right,
  confidence, w_eeg, w_emg          # w_* 는 fusion 조건만
```

### T4. 실행
```bash
for seed in 42 1337 2024; do
  for cond in eeg_only emg_only fusion; do
    python src/ablation_study_v5.py --model_type $cond --seed $seed \
      --data_dir BCI_Research/preprocessed/member_A \
      --exclude_json BCI_Research/results/ablation_v5_clean/excluded_epochs.json
  done
done
```
장시간 작업이므로 fold 단위 체크포인팅과 재시작 지원을 넣으세요. 진행 상황을 `results/ablation_v5_clean/progress.log`에 남기세요.

### T5. 분석 재실행 — `src/analysis_v5.py` (신규)
v5 결과로 아래를 전부 재계산하고 **v4 값과 나란히** 출력하세요.

| # | 분석 | v4 참조값 |
|---|---|---|
| A1 | 조건별 accuracy / κ / ITR (mean ± SD, seed 간 SD 포함) | Fusion 74.18±11.19%, κ 0.484; EEG 72.15±11.44%, κ 0.443; sEMG 65.00±11.54%, κ 0.300 |
| A2 | Wilcoxon 9개 비교 + Bonferroni (p_corr = 9p, **0.05와 비교**) | Fusion vs EEG p_corr=1.000; Fusion vs sEMG <0.001; EEG vs sEMG 0.021–0.032 |
| A3 | **ρ(mean w_sEMG, sEMG-only accuracy)** ← 핵심 | +0.530, p=5.3e-5 |
| A4 | ρ(mean w_sEMG, resting EMG SNR) | +0.061, p=0.669 (ns) |
| A5 | **ρ(sEMG-only accuracy, fusion 이득)** ← 핵심 | +0.562, p<1e-4 |
| A6 | ρ(w_sEMG, fusion 이득) | +0.504, p=1e-4 |
| A7 | sEMG-only accuracy 사분위별 mean w_sEMG + Q4 vs Q1 Mann–Whitney | 0.255/0.256/0.399/0.476, U=132 p=0.0036 |
| A8 | High/Low-sEMG 중앙분할 subgroup (Fusion vs EEG-only) | High n=27 p_corr=0.043 r=0.528 |
| A9 | trial-level: 정답/오답 trial 간 w_sEMG (피험자 paired Wilcoxon) + confidence 상관 | W=418 p=0.014; ρ=−0.161 |
| A10 | Right MI bias 피험자 수 및 recall 차 | 9명, +0.394±0.062 |
| A11 | 성능 tier (≥80 / 65–80 / <65) 인원·평균 | 14 / 28 / 10명 |

SNR은 `BCI_Research/preprocessed/member_A/summary_member_A_v4.csv`의 `emg_snr_db` 사용 (전처리 재실행 안 하므로 그대로 유효).

### T6. 산출물
```
results/ablation_v5_clean/
  excluded_epochs.json
  ablation_v5_results.csv          # 피험자 × 조건 × seed
  ablation_v5_summary.json
  wilcoxon_v5.csv
  attention_weights_v5.csv         # sid, w_eeg_mean, w_emg_mean, w_*_std, accuracy
  correlations_v4_vs_v5.csv        # A3~A7 대조표
  per_trial/*.csv
  REPORT.md                        # 아래 형식
```

`REPORT.md`에 포함할 것:
1. 배제 규모 (피험자별 표 + 총계), D1~D4 결정 사항이 어떻게 처리됐는지
2. A1~A11 v4 vs v5 대조표
3. **핵심 판정** — 5절의 결정 규칙 적용 결과
4. 예상과 다른 점, 이상 징후, 신뢰할 수 없는 수치

---

## 5. 결정 규칙 — 결과를 어떻게 읽을 것인가

**이 규칙을 미리 고정합니다. 결과를 보고 나서 기준을 바꾸지 마세요.**

### 판정 A — sEMG 신호의 정체
| v5 sEMG-only accuracy | 해석 |
|---|---|
| ≥ 63% 유지 | subthreshold 활동 해석 지지. 논문의 sEMG 전제 유지 가능 |
| 55–63% | 부분 오염. "일부 기여는 움직임"으로 명시하고 톤 하향 |
| < 55% (우연 근처) | **sEMG 기여의 대부분이 실제 움직임.** hybrid fusion 서사 전면 재작성 필요 |

### 판정 B — 게이트 메커니즘 (논문의 축)
| v5 A3 (ρ(w_sEMG, sEMG acc)) | 해석 |
|---|---|
| p < 0.05 이고 ρ ≥ +0.35 | **핵심 주장 확립.** 오염 배제 후에도 성립 → 논문의 중심 결과로 유지 |
| p ≥ 0.05 | 주장 철회. "게이트가 무엇에 반응하는지 특정하지 못했다"는 negative finding으로 전환 |

### 판정 C — 조건부 이득
| v5 A5 (ρ(sEMG acc, 이득)) | 해석 |
|---|---|
| p < 0.05 유지 | 조건부 이득은 살아남음. 판정 B가 실패해도 이것만으로 논문 성립 가능 |
| p ≥ 0.05 | 논문의 결론을 "fusion은 이 데이터셋에서 이점이 없다"로 재구성 |

### 판정 D — 무결성 확인
v5 EEG-only 정확도가 v4(72.15%)에서 **±3%p 이상 벗어나면 파이프라인 오류를 의심**하세요. EEG 스트림은 움직임 오염과 거의 무관해야 하므로 큰 변화는 배제 로직이나 인덱싱이 잘못됐다는 신호입니다. 그 경우 진행하지 말고 원인부터 찾으세요.

---

## 6. 원칙

- **v4 파일을 덮어쓰지 마세요.** 모든 출력은 새 디렉터리로.
- **모델·하이퍼파라미터를 바꾸지 마세요.** 유일한 변수는 배제된 trial이어야 합니다.
- **결과가 나쁘게 나와도 그대로 보고하세요.** 이 실험의 목적은 논문을 구하는 게 아니라 sEMG 결과가 진짜인지 판정하는 것입니다. 판정 A에서 55% 미만이 나오면 그게 정답이고, 그 사실을 아는 편이 리뷰어에게 지적당하는 것보다 낫습니다.
- 중간에 assert가 실패하거나 이상한 값이 나오면 **우회하지 말고 멈추고 보고**하세요.
- 실행 전에 T1을 먼저 돌려 배제 규모가 위 표(724 + 30)와 일치하는지 확인하세요. 불일치하면 매핑이 틀린 것입니다.

---

## 7. 여유가 되면 함께 처리 (선택, 우선순위 순)

### O1. concat baseline
`ablation_study_v5.py`에 `--model_type concat` 추가: EEG·sEMG 인코더 출력을 attention 없이 `torch.cat` → `Linear(512→256)` → 기존 classifier. 나머지 동일.

이유: 현재 원고의 Table 4에 있던 "EEGNet+sEMG concat 69.7%" 행은 **저장소 어디에도 근거가 없습니다.** `results/ablation/`에 결과 파일이 없고, `ablation_study.py`에 concat 분기가 없으며, git 이력에도 없고, `comparison_table.csv`는 이 값을 `Park et al. 2022 Sensors`(문헌값)로 기록하고 있습니다. 그래서 투고본에서 해당 행과 "attention이 concat보다 4.5pp 우수" 주장을 삭제했습니다. 직접 돌리면 복원 가능하고, 여기에 Wilcoxon 검정을 붙이면 attention의 기여를 처음으로 실증하게 됩니다.

### O2. gating 기여 분리 ablation
`--fusion_mode {softmax_gate, fixed_half, static_scalar, concat}` 4조건 비교. `fixed_half`는 (0.5, 0.5) 고정, `static_scalar`는 입력과 무관한 학습 파라미터 1개. attention 자체의 기여를 분리하는 유일한 실험입니다.

### O3. transfer 스크립트 버그 2건
`src/transfer_bcic2a.py`:
- **채널**: `BCIC_22` 리스트 마지막이 `'Oz'`인데 BCI IV 2a 몬타주의 22번째 채널은 **`POz`**입니다. 22채널 중 1개 오매핑. (라벨 매핑 769→0/770→1과 `BCIC_GDF_22_IDX = range(22)`는 검증 결과 **정상**입니다 — 노트북의 769/770 경고는 잘못된 단서입니다)
- **정규화**: GigaDB는 피험자 내 전 에폭에 대한 채널별 z-score인데, transfer 경로는 **trial별** z-score를 씁니다. 주석은 "GigaDB와 동일"이라고 되어 있으나 다릅니다.

둘 다 고치고 재실행. 그래도 우연 수준(49.8%)이면 negative result로 보고할 가치가 있습니다.

### O4. sEMG onset latency 분석
cue 기준 sEMG 발화 시점 분포. cue 후 200 ms 이내 발화는 imagery로 설명하기 어려우므로, 판정 A를 보강하는 독립 증거가 됩니다.

---

## 8. 완료 후

`results/ablation_v5_clean/REPORT.md`와 `correlations_v4_vs_v5.csv`를 확인하고, 판정 A/B/C 결과를 알려주세요. 그에 맞춰 원고(`MI-BCI_paper_JNE_submission_v1.docx`)의 §2.2.3 / §3.5 / §3.6 / §4.2 / §4.6.4 / 초록 / 결론을 다시 씁니다.

관련 문서: `JNE_개정_검증보고서_20260810.md` (검증 내역 전체), `JNE_투고_사전심사_리뷰_20260810.md` (1차 리뷰)
