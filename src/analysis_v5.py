"""
analysis_v5.py — A1~A11 재계산 및 v4 대조 (T5)
=================================================================
v5-clean 결과가 있으면 v4와 나란히, 없으면 v4만 출력한다.
통계 관례는 v4 스크립트(wilcoxon_analysis.py / subgroup_analysis.py /
bias_analysis.py)와 동일하게 맞췄다.

⚠ 피험자 집합 불일치 주의
   v4 = 52명, v5 = D1(s06 원본 부재) + D3(클래스당 <30 trial) 적용 후 49명.
   집합이 다르면 차이가 "배제 효과"인지 "피험자 구성 변화"인지 알 수 없으므로
   v4 를 **동일 피험자로 재계산한 열(v4_matched)** 을 항상 함께 낸다.

실행:
    python src/analysis_v5.py

출력:
    results/ablation_v5_clean/
      ├── correlations_v4_vs_v5.csv     (A3~A7)
      ├── wilcoxon_v5.csv               (A2)
      ├── attention_weights_v5.csv
      └── analysis_v5_summary.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu


def bonferroni(pvals):
    """p_corr = n*p, 1.0 로 절단 — statsmodels 의 'bonferroni' 와 동일."""
    pv = np.asarray(pvals, float)
    return np.minimum(pv * len(pv), 1.0)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "BCI_Research" / "results"
V4_DIR = RES / "ablation"
V5_DIR = RES / "ablation_v5_clean"
ATTN_V4 = RES / "attention" / "attention_weights_per_subject.csv"
SUMMARY = ROOT / "BCI_Research" / "preprocessed" / "member_A" / "summary_member_A_v4.csv"

CONDS = ["eeg_only", "emg_only", "fusion"]
BIAS_THRESHOLD = 0.30


# ── v4 공통 헬퍼 (원본과 동일 정의) ──────────────────────────────
def wilcoxon_r(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    diff = x - y
    nz = int((diff != 0).sum())
    if nz < 2:
        return np.nan, np.nan, np.nan, np.nan
    res = wilcoxon(x, y, alternative="two-sided")
    stat, p = float(res.statistic), float(res.pvalue)
    mu = nz * (nz + 1) / 4.0
    sigma = np.sqrt(nz * (nz + 1) * (2 * nz + 1) / 24.0)
    z = (stat - mu) / sigma
    return stat, p, float(z), float(abs(z) / np.sqrt(nz))


def effect_label(r):
    if np.isnan(r):
        return "n/a"
    if r >= 0.5:
        return "large"
    if r >= 0.3:
        return "medium"
    if r >= 0.1:
        return "small"
    return "negligible"


def _rho(x, y):
    """Spearman ρ. 유효 표본 <3 이면 NaN."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3:
        return np.nan, np.nan, int(m.sum())
    r = spearmanr(x[m], y[m])
    return float(r.statistic), float(r.pvalue), int(m.sum())


# ════════════════════════════════════════════════════════════════
#  데이터 로드
# ════════════════════════════════════════════════════════════════

def load_v4():
    """v4 피험자 단위 테이블 (sid, acc_*, kappa_*, recall, w_emg, snr)."""
    df = pd.read_csv(V4_DIR / "ablation_results.csv")
    attn = pd.read_csv(ATTN_V4)[["sid", "w_eeg_mean", "w_emg_mean",
                                 "w_eeg_std", "w_emg_std", "emg_snr_db"]]
    df = df.merge(attn, on="sid", how="left")
    df["gain"] = df["acc_fusion"] - df["acc_eeg_only"]
    return df


def load_v5():
    """v5 결과. seed 간 평균 후 피험자 단위 테이블로 환원."""
    path = V5_DIR / "ablation_v5_results.csv"
    if not path.exists():
        return None, None
    raw = pd.read_csv(path)

    wide = None
    for cond in CONDS:
        sub = raw[raw.model_type == cond]
        if sub.empty:
            print(f"  ⚠ v5 에 {cond} 결과 없음 — v5 분석 생략")
            return None, None
        agg = sub.groupby("sid").agg(
            **{f"acc_{cond}": ("accuracy", "mean"),
               f"kappa_{cond}": ("kappa", "mean"),
               f"itr_{cond}": ("itr", "mean"),
               f"left_recall_{cond}": ("left_recall", "mean"),
               f"right_recall_{cond}": ("right_recall", "mean")}
        ).reset_index()
        wide = agg if wide is None else wide.merge(agg, on="sid")

    fus = raw[raw.model_type == "fusion"]
    if "w_emg_mean" in fus.columns:
        w = fus.groupby("sid").agg(
            w_emg_mean=("w_emg_mean", "mean"),
            w_eeg_mean=("w_eeg_mean", "mean"),
            w_emg_std=("w_emg_std", "mean"),
        ).reset_index()
        wide = wide.merge(w, on="sid", how="left")

    snr = pd.read_csv(SUMMARY)[["sid", "emg_snr_db"]]
    wide = wide.merge(snr, on="sid", how="left")
    wide["gain"] = wide["acc_fusion"] - wide["acc_eeg_only"]
    return wide, raw


# ════════════════════════════════════════════════════════════════
#  A1 / A2
# ════════════════════════════════════════════════════════════════

def a1(df, raw, label, out):
    print(f"\n── A1 조건별 성능 [{label}] (n={len(df)}) ──")
    rec = {}
    for cond in CONDS:
        acc, kap, itr = (df[f"acc_{cond}"], df[f"kappa_{cond}"],
                         df.get(f"itr_{cond}"))
        seed_sd = None
        if raw is not None and "seed" in raw.columns:
            per_seed = raw[raw.model_type == cond].groupby("seed")["accuracy"].mean()
            if len(per_seed) > 1:
                seed_sd = float(per_seed.std())
        rec[cond] = {
            "acc_mean": float(acc.mean()), "acc_sd": float(acc.std()),
            "kappa_mean": float(kap.mean()), "kappa_sd": float(kap.std()),
            "itr_mean": float(itr.mean()) if itr is not None else None,
            "acc_sd_across_seeds": seed_sd,
        }
        line = (f"  {cond:9s} acc={acc.mean()*100:5.2f}±{acc.std()*100:5.2f}%  "
                f"κ={kap.mean():.3f}±{kap.std():.3f}")
        if itr is not None:
            line += f"  ITR={itr.mean():5.2f}"
        if seed_sd is not None:
            line += f"  (seed간 SD={seed_sd*100:.2f}%p)"
        print(line)
    out[f"A1_{label}"] = rec
    return rec


def a2(df, label, out, save=None):
    """9개 Wilcoxon (3 pair × 3 metric) + Bonferroni — v4와 동일 구성."""
    pairs = [("fusion", "eeg_only"), ("fusion", "emg_only"),
             ("eeg_only", "emg_only")]
    metrics = [("accuracy", "acc"), ("kappa", "kappa"), ("itr", "itr")]
    rows = []
    for a, b in pairs:
        for mname, pre in metrics:
            ca, cb = f"{pre}_{a}", f"{pre}_{b}"
            if ca not in df.columns or cb not in df.columns:
                continue
            stat, p, z, r = wilcoxon_r(df[ca], df[cb])
            rows.append({"comparison": f"{a} vs {b}", "metric": mname,
                         "stat": stat, "p_raw": p, "z": z, "effect_r": r,
                         "effect": effect_label(r)})
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["p_corr"] = bonferroni(res["p_raw"].fillna(1.0))
    print(f"\n── A2 Wilcoxon + Bonferroni [{label}] ──")
    for _, r in res.iterrows():
        sig = "*" if r.p_corr < 0.05 else "ns"
        print(f"  {r.comparison:22s} {r.metric:8s} p_corr={r.p_corr:7.4f} "
              f"r={r.effect_r:.3f} {sig}")
    if save:
        res.to_csv(save, index=False)
    out[f"A2_{label}"] = res.to_dict("records")
    return res


# ════════════════════════════════════════════════════════════════
#  A3~A7 상관
# ════════════════════════════════════════════════════════════════

CORR_DEFS = [
    ("A3", "rho(w_sEMG, sEMG-only acc)", "w_emg_mean", "acc_emg_only", 0.530),
    ("A4", "rho(w_sEMG, resting EMG SNR)", "w_emg_mean", "emg_snr_db", 0.061),
    ("A5", "rho(sEMG-only acc, fusion gain)", "acc_emg_only", "gain", 0.562),
    ("A6", "rho(w_sEMG, fusion gain)", "w_emg_mean", "gain", 0.504),
]


def correlations(v4, v4m, v5, out):
    rows = []
    for key, name, cx, cy, ref in CORR_DEFS:
        entry = {"id": key, "analysis": name, "v4_paper_ref_rho": ref}
        for lbl, d in [("v4", v4), ("v4_matched", v4m), ("v5", v5)]:
            if d is None or cx not in d.columns or cy not in d.columns:
                entry[f"{lbl}_rho"] = entry[f"{lbl}_p"] = entry[f"{lbl}_n"] = np.nan
                continue
            r, p, n = _rho(d[cx], d[cy])
            entry[f"{lbl}_rho"], entry[f"{lbl}_p"], entry[f"{lbl}_n"] = r, p, n
        rows.append(entry)

    df = pd.DataFrame(rows)
    print("\n── A3~A6 상관 (v4 vs v5) ──")
    for _, r in df.iterrows():
        s = f"  {r.id} {r.analysis:34s}"
        for lbl in ["v4", "v4_matched", "v5"]:
            rho, p = r[f"{lbl}_rho"], r[f"{lbl}_p"]
            s += (f" | {lbl}: ρ={rho:+.3f} p={p:.4g}"
                  if not np.isnan(rho) else f" | {lbl}: —")
        print(s)
    out["A3_A6"] = df.to_dict("records")
    return df


def a7(df, label, out):
    """sEMG-only 정확도 사분위별 w_sEMG + Q4 vs Q1 Mann-Whitney."""
    if df is None or "w_emg_mean" not in df.columns:
        return None
    d = df.dropna(subset=["acc_emg_only", "w_emg_mean"]).copy()
    if len(d) < 8:
        return None
    d["q"] = pd.qcut(d["acc_emg_only"], 4, labels=[1, 2, 3, 4])
    means = d.groupby("q", observed=True)["w_emg_mean"].mean()
    q1 = d[d.q == 1]["w_emg_mean"].values
    q4 = d[d.q == 4]["w_emg_mean"].values
    u, p = mannwhitneyu(q4, q1, alternative="two-sided")
    print(f"\n── A7 사분위별 mean w_sEMG [{label}] ──")
    print("  " + " / ".join(f"Q{int(k)}={v:.3f}" for k, v in means.items()))
    print(f"  Q4 vs Q1: U={u:.0f}  p={p:.4f}")
    out[f"A7_{label}"] = {"quartile_means": {str(k): float(v) for k, v in means.items()},
                          "U": float(u), "p": float(p)}
    return means


def a8(df, label, out):
    """sEMG 품질 median split → 각 그룹 Fusion vs EEG-only."""
    if df is None:
        return None
    med = float(df["acc_emg_only"].median())
    groups = {"High-sEMG": df[df.acc_emg_only >= med],
              "Low-sEMG": df[df.acc_emg_only < med]}
    rows = []
    for gname, g in groups.items():
        for mname, pre in [("accuracy", "acc"), ("kappa", "kappa"), ("itr", "itr")]:
            ca, cb = f"{pre}_fusion", f"{pre}_eeg_only"
            if ca not in g.columns or cb not in g.columns:
                continue
            stat, p, z, r = wilcoxon_r(g[ca], g[cb])
            rows.append({"group": gname, "n": len(g), "metric": mname,
                         "p_raw": p, "effect_r": r})
    res = pd.DataFrame(rows)
    if res.empty:
        return None
    res["p_corr"] = bonferroni(res["p_raw"].fillna(1.0))
    print(f"\n── A8 median split (Fusion vs EEG-only) [{label}]  med={med:.4f} ──")
    for _, r in res[res.metric == "accuracy"].iterrows():
        print(f"  {r.group:10s} n={r.n:2d}  p_corr={r.p_corr:.4f}  r={r.effect_r:.3f}")
    out[f"A8_{label}"] = res.to_dict("records")
    return res


def a9(label, out, per_trial_dir, pattern="fusion_seed*_s*.csv"):
    """trial-level: 정답/오답 trial 간 w_sEMG (피험자 paired Wilcoxon)."""
    files = sorted(Path(per_trial_dir).glob(pattern)) if Path(per_trial_dir).exists() else []
    if not files:
        return None
    corr_w, wrong_w, confs, ws = [], [], [], []
    for f in files:
        d = pd.read_csv(f)
        if "w_emg" not in d.columns:
            continue
        c, w = d[d.correct == 1]["w_emg"], d[d.correct == 0]["w_emg"]
        if len(c) and len(w):
            corr_w.append(c.mean())
            wrong_w.append(w.mean())
        confs.append(d["confidence"].values)
        ws.append(d["w_emg"].values)
    if len(corr_w) < 3:
        return None
    stat, p, z, r = wilcoxon_r(corr_w, wrong_w)
    rho, prho, n = _rho(np.concatenate(ws), np.concatenate(confs))
    print(f"\n── A9 trial-level w_sEMG [{label}]  (피험자 {len(corr_w)}명) ──")
    print(f"  정답 {np.mean(corr_w):.4f} vs 오답 {np.mean(wrong_w):.4f}  "
          f"W={stat:.0f} p={p:.4f}")
    print(f"  ρ(w_sEMG, confidence) = {rho:+.3f}  p={prho:.3g}  (n={n} trial)")
    out[f"A9_{label}"] = {"n_subjects": len(corr_w), "W": stat, "p": p,
                          "w_correct": float(np.mean(corr_w)),
                          "w_incorrect": float(np.mean(wrong_w)),
                          "rho_conf": rho, "p_conf": prho}
    return out[f"A9_{label}"]


def a10(df, label, out):
    d = df["right_recall_fusion"] - df["left_recall_fusion"]
    mask = d > BIAS_THRESHOLD
    print(f"\n── A10 Right MI bias [{label}] ──")
    print(f"  bias 피험자 {int(mask.sum())}명 / {len(df)}  "
          f"(recall차 {d[mask].mean():+.3f}±{d[mask].std():.3f})"
          if mask.any() else f"  bias 피험자 0명 / {len(df)}")
    out[f"A10_{label}"] = {"n_bias": int(mask.sum()), "n_total": len(df),
                           "sids": df.loc[mask, "sid"].tolist(),
                           "recall_diff_mean": float(d[mask].mean()) if mask.any() else None,
                           "recall_diff_sd": float(d[mask].std()) if mask.any() else None}


def a11(df, label, out):
    acc = df["acc_fusion"] * 100
    tiers = {"high(>=80)": (acc >= 80), "mid(65-80)": (acc >= 65) & (acc < 80),
             "low(<65)": acc < 65}
    print(f"\n── A11 성능 tier [{label}] ──")
    rec = {}
    for k, m in tiers.items():
        rec[k] = {"n": int(m.sum()),
                  "mean_acc": float(acc[m].mean()) if m.any() else None}
        print(f"  {k:12s} n={int(m.sum()):2d}  "
              f"mean={acc[m].mean():.2f}%" if m.any() else f"  {k:12s} n=0")
    out[f"A11_{label}"] = rec


# ════════════════════════════════════════════════════════════════

def main():
    V5_DIR.mkdir(parents=True, exist_ok=True)
    out = {}

    v4 = load_v4()
    v5, v5raw = load_v5()

    if v5 is None:
        print("ℹ v5 학습 결과 없음 — v4 기준선만 재계산합니다.\n"
              "   (ablation_v5_results.csv 생성 후 다시 실행하세요)")
        v4m = None
    else:
        keep = set(v5["sid"])
        v4m = v4[v4.sid.isin(keep)].reset_index(drop=True)
        print(f"피험자 집합: v4={len(v4)}명, v5={len(v5)}명, "
              f"matched={len(v4m)}명")

    a1(v4, None, "v4", out)
    if v4m is not None:
        a1(v4m, None, "v4_matched", out)
    if v5 is not None:
        a1(v5, v5raw, "v5", out)

    a2(v4, "v4", out)
    if v5 is not None:
        a2(v5, "v5", out, save=V5_DIR / "wilcoxon_v5.csv")

    corr = correlations(v4, v4m, v5, out)
    corr.to_csv(V5_DIR / "correlations_v4_vs_v5.csv", index=False)

    for lbl, d in [("v4", v4), ("v4_matched", v4m), ("v5", v5)]:
        if d is None:
            continue
        a7(d, lbl, out)
        a8(d, lbl, out)
        a10(d, lbl, out)
        a11(d, lbl, out)

    a9("v4ckpt_alltrials", out, V5_DIR / "per_trial_v4ckpt",
       pattern="fusion_v4ckpt_s*.csv")
    a9("v5", out, V5_DIR / "per_trial", pattern="fusion_seed*_s*.csv")

    if v5 is not None and "w_emg_mean" in v5.columns:
        cols = ["sid", "w_eeg_mean", "w_emg_mean", "w_emg_std", "acc_fusion"]
        v5[[c for c in cols if c in v5.columns]].to_csv(
            V5_DIR / "attention_weights_v5.csv", index=False)

    with open(V5_DIR / "analysis_v5_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n저장: {V5_DIR/'analysis_v5_summary.json'}")

    # ── 판정 A/B/C (5절 결정 규칙) ──────────────────────────────
    if v5 is not None:
        print(f"\n{'='*60}\n  판정 (5절 결정 규칙)\n{'='*60}")
        sacc = v5["acc_emg_only"].mean() * 100
        verdict_a = ("subthreshold 해석 지지" if sacc >= 63 else
                     "부분 오염 — 톤 하향" if sacc >= 55 else
                     "대부분 실제 움직임 — 서사 재작성")
        print(f"  판정 A: sEMG-only = {sacc:.2f}%  → {verdict_a}")
        r3 = corr[corr.id == "A3"].iloc[0]
        print(f"  판정 B: A3 ρ={r3.v5_rho:+.3f} p={r3.v5_p:.4g} → "
              + ("핵심 주장 확립" if (r3.v5_p < 0.05 and r3.v5_rho >= 0.35)
                 else "주장 철회 (negative finding)"))
        r5 = corr[corr.id == "A5"].iloc[0]
        print(f"  판정 C: A5 ρ={r5.v5_rho:+.3f} p={r5.v5_p:.4g} → "
              + ("조건부 이득 유지" if r5.v5_p < 0.05 else "이득 없음으로 재구성"))
        eeg = v5["acc_eeg_only"].mean() * 100
        base = v4m["acc_eeg_only"].mean() * 100
        print(f"  판정 D: EEG-only v5={eeg:.2f}% vs v4_matched={base:.2f}% "
              f"(Δ={eeg-base:+.2f}%p) → "
              + ("정상" if abs(eeg - base) < 3 else "⚠ 파이프라인 오류 의심"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
