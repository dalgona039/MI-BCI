"""
extract_bad_trials.py — GigaDB bad-trial 플래그 → h5 에폭 인덱스 변환
======================================================================
Cho et al. 2017 (GigaScience 6:gix034) 의 `eeg.bad_trial_indices` 에서
  - bad_trial_idx_mi       : EMG 활동과 상관된 trial (실제 움직임 의심)
  - bad_trial_idx_voltage  : 전압 크기 기준 불량 trial
를 읽어 클래스 내부 1-based 인덱스를 h5 에폭 인덱스로 매핑한다.

매핑 (2.3):
    n = eeg.n_imagery_trials              # 클래스당 trial 수
    left  trial i (1-based) → epoch  i - 1
    right trial j (1-based) → epoch  n + j - 1

실행:
    python src/extract_bad_trials.py

출력:
    BCI_Research/results/ablation_v5_clean/excluded_epochs.json
    BCI_Research/results/ablation_v5_clean/excluded_epochs_summary.csv
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
MAT_DIR = ROOT / "GigaDB_100295"
H5_DIR = ROOT / "BCI_Research" / "preprocessed" / "member_A"
OUT_DIR = ROOT / "BCI_Research" / "results" / "ablation_v5_clean"

# D3: 제거 후 클래스당 이 수 미만이면 평가에서 제외
MIN_TRIALS_PER_CLASS = 30


def _as_idx_list(cell_entry) -> list:
    """cell 원소를 1-based 정수 리스트로 정규화.

    scalar 는 0-d 배열, 비어있으면 shape (0,) 로 들어온다.
    """
    arr = np.atleast_1d(np.asarray(cell_entry).squeeze())
    if arr.size == 0:
        return []
    return [int(v) for v in arr.ravel()]


def _read_flags(mat_path: Path):
    """(n_per_class, mi_left, mi_right, v_left, v_right) 반환."""
    m = sio.loadmat(str(mat_path), variable_names=["eeg"],
                    struct_as_record=False, squeeze_me=True)
    eeg = m["eeg"]
    n = int(eeg.n_imagery_trials)
    bt = eeg.bad_trial_indices

    mi = bt.bad_trial_idx_mi
    volt = bt.bad_trial_idx_voltage
    assert len(mi) == 2, f"{mat_path.name}: bad_trial_idx_mi 가 2-element cell 이 아님"
    assert len(volt) == 2, f"{mat_path.name}: bad_trial_idx_voltage 가 2-element cell 이 아님"

    return (n, _as_idx_list(mi[0]), _as_idx_list(mi[1]),
            _as_idx_list(volt[0]), _as_idx_list(volt[1]))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(MAT_DIR.glob("s[0-9][0-9].mat"))
    available = [int(p.stem[1:]) for p in mat_files]
    missing = sorted(set(range(1, 53)) - set(available))
    print(f"원본 .mat: {len(available)}개 발견, 누락 sid={missing}")

    out = {}
    rows = []
    tot_mi = tot_volt = tot_union = tot_trials = 0

    for sid in available:
        mat_path = MAT_DIR / f"s{sid:02d}.mat"
        h5_path = H5_DIR / f"sub-{sid:02d}_member_A.h5"
        if not h5_path.exists():
            print(f"  ✗ s{sid:02d}: h5 없음 — 중단")
            return 1

        n, mi_l, mi_r, v_l, v_r = _read_flags(mat_path)

        with h5py.File(h5_path, "r") as f:
            labels = f["labels"][:].astype(int)

        # ── 2.3 assert: 구조 검증 (실패 시 즉시 중단) ────────────
        assert len(labels) == 2 * n, (
            f"s{sid:02d}: h5 에폭 {len(labels)}개 != 2*n_imagery_trials {2*n}. "
            "에폭이 버려졌거나 매핑 전제가 깨졌음"
        )
        assert (labels[:n] == 1).all(), (
            f"s{sid:02d}: 앞 {n}개 에폭이 전부 label=1(left) 이 아님 — "
            f"unique={np.unique(labels[:n])}"
        )
        assert (labels[n:] == 2).all(), (
            f"s{sid:02d}: 뒤 {n}개 에폭이 전부 label=2(right) 이 아님 — "
            f"unique={np.unique(labels[n:])}"
        )
        for name, lst in [("mi_left", mi_l), ("mi_right", mi_r),
                          ("volt_left", v_l), ("volt_right", v_r)]:
            assert all(1 <= i <= n for i in lst), (
                f"s{sid:02d}: {name} 인덱스가 [1,{n}] 범위를 벗어남 — {lst}"
            )

        # ── 클래스 내부 1-based → h5 에폭 인덱스 ────────────────
        ep_mi = {i - 1 for i in mi_l} | {n + j - 1 for j in mi_r}
        ep_volt = {i - 1 for i in v_l} | {n + j - 1 for j in v_r}
        excl = sorted(ep_mi | ep_volt)

        n_left_kept = n - len([e for e in excl if e < n])
        n_right_kept = n - len([e for e in excl if e >= n])

        out[str(sid)] = {
            "n_per_class": n,
            "n_total": 2 * n,
            "excl_idx": excl,
            "n_excl_mi": len(ep_mi),
            "n_excl_voltage": len(ep_volt),
            "n_excl_union": len(excl),
            "n_left_kept": n_left_kept,
            "n_right_kept": n_right_kept,
            "eligible": bool(min(n_left_kept, n_right_kept) >= MIN_TRIALS_PER_CLASS),
        }
        rows.append({
            "sid": sid, "n_per_class": n, "n_total": 2 * n,
            "n_excl_mi": len(ep_mi), "n_excl_voltage": len(ep_volt),
            "n_excl_union": len(excl),
            "n_left_kept": n_left_kept, "n_right_kept": n_right_kept,
            "pct_excluded": round(100.0 * len(excl) / (2 * n), 2),
            "eligible": min(n_left_kept, n_right_kept) >= MIN_TRIALS_PER_CLASS,
        })

        tot_mi += len(ep_mi)
        tot_volt += len(ep_volt)
        tot_union += len(excl)
        tot_trials += 2 * n

    df = pd.DataFrame(rows).sort_values("sid")
    df.to_csv(OUT_DIR / "excluded_epochs_summary.csv", index=False)
    with open(OUT_DIR / "excluded_epochs.json", "w") as f:
        json.dump(out, f, indent=2)

    # ── 검증 리포트 ──────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  총 trial          : {tot_trials}")
    print(f"  MI(움직임) 플래그 : {tot_mi}   (기대값 724)")
    print(f"  전압 플래그       : {tot_volt}   (기대값 30)")
    print(f"  합집합 배제       : {tot_union}  ({100*tot_union/tot_trials:.2f}%)")
    print(f"  중복(mi∩voltage)  : {tot_mi + tot_volt - tot_union}")
    print(f"  플래그 ≥1 피험자  : {(df.n_excl_union > 0).sum()} / {len(df)}")
    print(f"{'='*64}")

    ineligible = df[~df.eligible]
    if len(ineligible):
        print(f"\n  ⚠ 클래스당 <{MIN_TRIALS_PER_CLASS} trial → 평가 제외 대상 (D3):")
        for _, r in ineligible.iterrows():
            print(f"    s{int(r.sid):02d}: left {int(r.n_left_kept)}, "
                  f"right {int(r.n_right_kept)} 잔존")

    print(f"\n  상위 배제율 피험자:")
    for _, r in df.nlargest(8, "pct_excluded").iterrows():
        print(f"    s{int(r.sid):02d}: {r.pct_excluded:5.1f}%  "
              f"(mi {int(r.n_excl_mi)}, volt {int(r.n_excl_voltage)})")

    ok = (tot_mi == 724 and tot_volt == 30)
    print(f"\n  기대 규모 일치: {'✅ 예' if ok else '❌ 아니오 — 매핑 확인 필요'}")
    print(f"  저장: {OUT_DIR/'excluded_epochs.json'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
