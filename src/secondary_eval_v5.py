"""
secondary_eval_v5.py — D2 부(secondary) 분석
=============================================================
"학습은 v4 전체 데이터로, 평가만 clean trial 로."

v4 fusion 체크포인트(results/checkpoints_A/best_sNN.pt)를 그대로 불러
  (a) 전체 trial  — v4 재현 검증
  (b) clean trial 만 — 오염 배제 평가
두 조건에서 평가한다. 학습된 표현은 동일하므로, 두 값의 차이는
**평가 데이터의 오염분** 만을 반영한다. 주(primary) 분석과 대조하면
"표현이 오염된 것인지, 평가가 오염된 것인지" 를 분리할 수 있다.

⚠ eeg_only / emg_only 는 v4에서 체크포인트를 저장하지 않았으므로
   (ablation_study.py 가 매 fold 재학습 후 폐기) 부 분석 대상이 아니다.
   fusion 만 가능하다.

실행:
    python src/secondary_eval_v5.py

출력:
    results/ablation_v5_clean/secondary_eval_fusion.csv
    results/ablation_v5_clean/per_trial_v4ckpt/fusion_v4ckpt_s{NN}.csv
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_study_v5 import (BCIDataset, FusionModel, CFG, evaluate,
                               _metrics, pick_device, _save_atomic)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "BCI_Research" / "preprocessed" / "member_A"
CKPT_DIR = ROOT / "BCI_Research" / "results" / "checkpoints_A"
OUT_DIR = ROOT / "BCI_Research" / "results" / "ablation_v5_clean"
V4_FUSION = ROOT / "BCI_Research" / "results" / "ablation" / "ablation_fusion_results.csv"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pt_dir = OUT_DIR / "per_trial_v4ckpt"
    pt_dir.mkdir(exist_ok=True)

    with open(OUT_DIR / "excluded_epochs.json") as f:
        excl = json.load(f)

    device = pick_device()
    cfg = dict(CFG)
    print(f"device={device}  |  v4 fusion 체크포인트 재평가\n")

    v4_ref = None
    if V4_FUSION.exists():
        v4_ref = pd.read_csv(V4_FUSION).set_index("sid")

    rows = []
    for sid in sorted(int(k) for k in excl.keys()):
        ckpt = CKPT_DIR / f"best_s{sid:02d}.pt"
        h5 = DATA_DIR / f"sub-{sid:02d}_member_A.h5"
        if not ckpt.exists():
            print(f"  s{sid:02d}: 체크포인트 없음 — 건너뜀")
            continue

        model = FusionModel(cfg).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=True))

        e = excl[str(sid)]
        ex_set = set(e["excl_idx"])

        # (a) 전체 trial — v4 재현
        ds_all = BCIDataset(str(h5), cfg["emg_ds_factor"])
        ld_all = DataLoader(ds_all, batch_size=64, shuffle=False)
        pr_a, tr_a, prob_a, w_a = evaluate(model, ld_all, device,
                                           return_detail=True)
        m_all = _metrics(sid, pr_a, tr_a)

        # (b) clean trial 만 — 같은 모델, 같은 예측의 부분집합
        keep = np.array([i not in ex_set for i in ds_all.epoch_idx])
        m_cln = _metrics(sid, pr_a[keep], tr_a[keep])

        # per-trial 저장 (clean 여부 플래그 포함)
        pd.DataFrame({
            "epoch_idx": ds_all.epoch_idx,
            "is_clean": keep.astype(int),
            "true_label": tr_a + 1, "pred_label": pr_a + 1,
            "correct": (pr_a == tr_a).astype(int),
            "prob_left": prob_a[:, 0], "prob_right": prob_a[:, 1],
            "confidence": prob_a.max(axis=1),
            "w_eeg": w_a[:, 0], "w_emg": w_a[:, 1],
        }).to_csv(pt_dir / f"fusion_v4ckpt_s{sid:02d}.csv", index=False)

        row = {
            "sid": sid,
            "n_all": len(tr_a), "n_clean": int(keep.sum()),
            "n_excluded": len(ex_set),
            "acc_all": m_all["accuracy"], "acc_clean": m_cln["accuracy"],
            "kappa_all": m_all["kappa"], "kappa_clean": m_cln["kappa"],
            "w_emg_all": float(w_a[:, 1].mean()),
            "w_emg_clean": float(w_a[keep, 1].mean()),
            "w_emg_excl": float(w_a[~keep, 1].mean()) if (~keep).any() else np.nan,
            "acc_excl": float((pr_a[~keep] == tr_a[~keep]).mean())
                        if (~keep).any() else np.nan,
        }
        if v4_ref is not None and sid in v4_ref.index:
            row["acc_v4_reported"] = float(v4_ref.loc[sid, "accuracy"])
            row["repro_delta"] = row["acc_all"] - row["acc_v4_reported"]
        rows.append(row)

        print(f"  s{sid:02d}: all={row['acc_all']:.4f}  clean={row['acc_clean']:.4f}"
              f"  (Δ={row['acc_clean']-row['acc_all']:+.4f}, "
              f"제거 {len(ex_set)})"
              + (f"  v4재현Δ={row['repro_delta']:+.4f}"
                 if "repro_delta" in row else ""))

    df = pd.DataFrame(rows)
    _save_atomic(df, str(OUT_DIR / "secondary_eval_fusion.csv"))

    print(f"\n{'='*64}")
    if "repro_delta" in df.columns:
        md = df["repro_delta"].abs().max()
        print(f"  v4 재현 검증: 최대 |Δ| = {md:.6f} "
              f"{'✅ 완전 일치' if md < 1e-6 else '⚠ 불일치 — 확인 필요'}")
    print(f"  fusion acc  전체 : {df.acc_all.mean():.4f} ± {df.acc_all.std():.4f}")
    print(f"  fusion acc  clean: {df.acc_clean.mean():.4f} ± {df.acc_clean.std():.4f}")
    print(f"  fusion κ    전체 : {df.kappa_all.mean():.4f}")
    print(f"  fusion κ    clean: {df.kappa_clean.mean():.4f}")
    sub = df[df.n_excluded > 0]
    print(f"\n  오염 trial 보유 {len(sub)}명 한정:")
    print(f"    clean trial 정확도 : {sub.acc_clean.mean():.4f}")
    print(f"    배제 trial 정확도  : {sub.acc_excl.mean():.4f}")
    print(f"    w_emg  clean/배제  : {sub.w_emg_clean.mean():.4f} / "
          f"{sub.w_emg_excl.mean():.4f}")
    print(f"{'='*64}")
    print(f"  저장: {OUT_DIR/'secondary_eval_fusion.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
