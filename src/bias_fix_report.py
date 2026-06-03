"""
bias_fix_report.py — Right MI Bias 수정 전략 최종 비교 리포트
============================================================
v4 baseline, Calibration (전체/조건부), Flip Aug (fine-tune/full retrain),
Cal+Flip 여섯 전략의 성능을 비교하고 bias 피험자 9명의 회복률을 보고합니다.

전제:
  - calibration.py 실행 완료            →  calibration_results.csv
  - calibration.py --conditional 실행   →  conditional_calibration_results.csv
  - train_flip_aug.py 실행 완료         →  flip_aug_results.csv, cal_flip_results.csv
  - train_flip_full.py 실행 완료        →  flip_full_results.csv

사용법:
  # Colab
  python bias_fix_report.py --drive_root /content/drive/MyDrive/MI-BCI

  # 로컬
  python src/bias_fix_report.py
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BIAS_SUBJECTS   = [1, 5, 7, 11, 12, 15, 24, 34, 36]
FIX_THRESHOLD   = 0.30   # right_recall - left_recall < 이 값 → bias 개선
KAPPA_LOSS_LIMIT = 0.03  # kappa 손실 허용 한계


# ════════════════════════════════════════════════════════════════
#  데이터 로드
# ════════════════════════════════════════════════════════════════

def load_csv(path: str, desc: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"  [경고] {desc} 없음: {path}")
        return None
    df = pd.read_csv(path)
    print(f"  로드: {desc}  ({len(df)}명)")
    return df


def load_all(drive_root: str) -> dict:
    bci_root = os.path.join(drive_root, "BCI_Research")
    abl_dir  = os.path.join(bci_root, "results", "ablation")
    cal_dir  = os.path.join(bci_root, "results", "calibration")

    print("\n  결과 파일 로드 중...")

    # v4 baseline: calibration_results.csv의 _before 컬럼을 기준으로 사용
    cal_df       = load_csv(os.path.join(cal_dir, "calibration_results.csv"),             "Calibration (전체)")
    cond_cal_df  = load_csv(os.path.join(cal_dir, "conditional_calibration_results.csv"), "Calibration (조건부)")
    flip_df      = load_csv(os.path.join(abl_dir, "flip_aug_results.csv"),                "Flip Aug (fine-tune)")
    cal_flip_df  = load_csv(os.path.join(abl_dir, "cal_flip_results.csv"),                "Cal+Flip")
    flip_full_df = load_csv(os.path.join(abl_dir, "flip_full_results.csv"),               "Flip (full retrain)")

    return {
        "cal":       cal_df,
        "cond_cal":  cond_cal_df,
        "flip":      flip_df,
        "cal_flip":  cal_flip_df,
        "flip_full": flip_full_df,
    }


# ════════════════════════════════════════════════════════════════
#  전략별 요약 계산
# ════════════════════════════════════════════════════════════════

def _bias_fix_rate(left_recall: pd.Series, right_recall: pd.Series,
                   bias_sids: list, all_sids: pd.Series) -> str:
    mask  = all_sids.isin(bias_sids)
    rdiff = right_recall[mask] - left_recall[mask]
    fixed = int((rdiff < FIX_THRESHOLD).sum())
    total = int(mask.sum())
    return f"{fixed}/{total}"


def _kappa_verdict(kappa: float, baseline_kappa: float) -> str:
    delta = kappa - baseline_kappa
    if delta < -KAPPA_LOSS_LIMIT:
        return f"[기각] Δκ={delta:+.4f}"
    return f"[통과] Δκ={delta:+.4f}"


def build_report(data: dict) -> str:
    cal_df       = data["cal"]
    cond_cal_df  = data["cond_cal"]
    flip_df      = data["flip"]
    cal_flip_df  = data["cal_flip"]
    flip_full_df = data["flip_full"]

    lines = []
    lines.append("\n" + "=" * 105)
    lines.append("  Right MI Bias 수정 전략 비교 리포트")
    lines.append("=" * 105)

    rows = []  # (name, acc, kappa, left_recall, right_recall, bias_fix, verdict)

    # ── v4 Baseline ──────────────────────────────────────────────
    # 우선순위: calibration_results.csv → conditional_calibration_results.csv → flip_full_results.csv
    baseline_src = cal_df if cal_df is not None else (
        cond_cal_df if cond_cal_df is not None else None
    )
    if baseline_src is not None:
        acc_col = "accuracy_before"
        k_col   = "kappa_before"
        lr_col  = "left_recall_before"
        rr_col  = "right_recall_before"
        acc  = baseline_src[acc_col].mean()
        k    = baseline_src[k_col].mean()
        lr   = baseline_src[lr_col].mean()
        rr   = baseline_src[rr_col].mean()
        bfr  = _bias_fix_rate(baseline_src[lr_col], baseline_src[rr_col],
                               BIAS_SUBJECTS, baseline_src["sid"])
        rows.append(("v4 baseline", acc, k, lr, rr, bfr, "—"))
        baseline_kappa = k
    elif flip_full_df is not None:
        acc  = flip_full_df["acc_v4"].mean()
        k    = flip_full_df["kappa_v4"].mean()
        lr   = flip_full_df["left_recall_v4"].mean()
        rr   = flip_full_df["right_recall_v4"].mean()
        bfr  = _bias_fix_rate(flip_full_df["left_recall_v4"], flip_full_df["right_recall_v4"],
                               BIAS_SUBJECTS, flip_full_df["sid"])
        rows.append(("v4 baseline", acc, k, lr, rr, bfr, "—"))
        baseline_kappa = k
    else:
        print("  [경고] v4 baseline 출처 없음 — fallback kappa=0.484 사용")
        baseline_kappa = 0.484

    # ── Calibration (전체) ────────────────────────────────────────
    if cal_df is not None:
        acc = cal_df["accuracy_after"].mean()
        k   = cal_df["kappa_after"].mean()
        lr  = cal_df["left_recall_after"].mean()
        rr  = cal_df["right_recall_after"].mean()
        bfr = _bias_fix_rate(cal_df["left_recall_after"],
                              cal_df["right_recall_after"],
                              BIAS_SUBJECTS, cal_df["sid"])
        rows.append(("+ Cal (전체)", acc, k, lr, rr, bfr,
                     _kappa_verdict(k, baseline_kappa)))

    # ── Calibration (조건부) ──────────────────────────────────────
    if cond_cal_df is not None:
        acc = cond_cal_df["accuracy_after"].mean()
        k   = cond_cal_df["kappa_after"].mean()
        lr  = cond_cal_df["left_recall_after"].mean()
        rr  = cond_cal_df["right_recall_after"].mean()
        bfr = _bias_fix_rate(cond_cal_df["left_recall_after"],
                              cond_cal_df["right_recall_after"],
                              BIAS_SUBJECTS, cond_cal_df["sid"])
        rows.append(("+ Cal (조건부)", acc, k, lr, rr, bfr,
                     _kappa_verdict(k, baseline_kappa)))

    # ── Flip Aug (fine-tune) ──────────────────────────────────────
    if flip_df is not None:
        acc = flip_df["accuracy_flip"].mean()
        k   = flip_df["kappa_flip"].mean()
        lr  = flip_df["left_recall_flip"].mean()
        rr  = flip_df["right_recall_flip"].mean()
        bfr = _bias_fix_rate(flip_df["left_recall_flip"],
                              flip_df["right_recall_flip"],
                              BIAS_SUBJECTS, flip_df["sid"])
        rows.append(("+ Flip (fine-tune)", acc, k, lr, rr, bfr,
                     _kappa_verdict(k, baseline_kappa)))

    # ── Flip Aug (full retrain) ───────────────────────────────────
    if flip_full_df is not None:
        acc = flip_full_df["acc_flip"].mean()
        k   = flip_full_df["kappa_flip"].mean()
        lr  = flip_full_df["left_recall_flip"].mean()
        rr  = flip_full_df["right_recall_flip"].mean()
        bfr = _bias_fix_rate(flip_full_df["left_recall_flip"],
                              flip_full_df["right_recall_flip"],
                              BIAS_SUBJECTS, flip_full_df["sid"])
        rows.append(("+ Flip (full)", acc, k, lr, rr, bfr,
                     _kappa_verdict(k, baseline_kappa)))

    # ── Cal + Flip ────────────────────────────────────────────────
    if cal_flip_df is not None:
        acc = cal_flip_df["accuracy_cal_flip"].mean()
        k   = cal_flip_df["kappa_cal_flip"].mean()
        lr  = cal_flip_df["left_recall_cal_flip"].mean()
        rr  = cal_flip_df["right_recall_cal_flip"].mean()
        bfr = _bias_fix_rate(cal_flip_df["left_recall_cal_flip"],
                              cal_flip_df["right_recall_cal_flip"],
                              BIAS_SUBJECTS, cal_flip_df["sid"])
        rows.append(("+ Cal+Flip", acc, k, lr, rr, bfr,
                     _kappa_verdict(k, baseline_kappa)))

    # ── 표 출력 ───────────────────────────────────────────────────
    header = (f"  {'Strategy':<20} | {'Accuracy':>8} | {'Kappa':>7} | "
              f"{'Left Rec':>8} | {'Right Rec':>9} | {'Bias Fix':>10} | Verdict")
    sep    = "  " + "-" * 101

    lines.append(header)
    lines.append(sep)

    for name, acc, k, lr, rr, bfr, verdict in rows:
        lines.append(
            f"  {name:<20} | {acc:>8.4f} | {k:>7.4f} | "
            f"{lr:>8.4f} | {rr:>9.4f} | {bfr:>10} | {verdict}"
        )

    lines.append(sep)
    lines.append(f"\n  Bias Fix 기준: right_recall - left_recall < {FIX_THRESHOLD:.2f} 로 개선된 피험자 수 / {len(BIAS_SUBJECTS)}명")
    lines.append(f"  Kappa 허용 손실: {KAPPA_LOSS_LIMIT:.2f} (기각 기준: |Δκ| > {KAPPA_LOSS_LIMIT:.2f})")

    # ── Bias 피험자 상세 ──────────────────────────────────────────
    lines.append("\n" + "=" * 105)
    lines.append(f"  Bias 피험자 ({len(BIAS_SUBJECTS)}명) 상세 recall_diff 변화")
    lines.append("=" * 105)
    lines.append(
        f"  {'SID':<6} | {'v4 base':>8} | {'Cal(전체)':>9} | "
        f"{'Cal(조건부)':>10} | {'Flip(fine)':>10} | {'Flip(full)':>10} | {'Cal+Flip':>9}"
    )
    lines.append("  " + "-" * 80)

    def _rd(df, sid, rcol, lcol):
        if df is None:
            return "    —  "
        row = df[df["sid"] == sid]
        if len(row) == 0:
            return "    —  "
        rd = float(row[rcol].iloc[0] - row[lcol].iloc[0])
        mark = "*" if rd < FIX_THRESHOLD else " "
        return f"{rd:>+.3f}{mark} "

    for sid in BIAS_SUBJECTS:
        v4_s  = _rd(baseline_src if baseline_src is not None else None,
                    sid, "right_recall_before", "left_recall_before") \
                if baseline_src is not None else "    —  "
        cal_s = _rd(cal_df,       sid, "right_recall_after",  "left_recall_after")
        cca_s = _rd(cond_cal_df,  sid, "right_recall_after",  "left_recall_after")
        flf_s = _rd(flip_df,      sid, "right_recall_flip",   "left_recall_flip")
        fll_s = _rd(flip_full_df, sid, "right_recall_flip",   "left_recall_flip")
        cfp_s = _rd(cal_flip_df,  sid, "right_recall_cal_flip", "left_recall_cal_flip")
        lines.append(
            f"  s{sid:02d}    | {v4_s:>8} | {cal_s:>9} | "
            f"{cca_s:>10} | {flf_s:>10} | {fll_s:>10} | {cfp_s:>9}"
        )

    lines.append("  " + "-" * 80)
    lines.append("  (* recall_diff = right_recall - left_recall, * = bias 개선됨 (< +0.30))")
    lines.append("=" * 105)

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════

def run(drive_root: str):
    data   = load_all(drive_root)
    report = build_report(data)
    print(report)

    out_dir  = os.path.join(drive_root, "BCI_Research", "results", "calibration")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bias_fix_report.txt")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\n  저장: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Bias Fix Strategy Comparison Report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--drive_root", type=str, default=None,
                   help="MI-BCI 루트 경로 (Colab: /content/drive/MyDrive/MI-BCI)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = args.drive_root or str(Path(__file__).resolve().parent.parent)
    run(root)
