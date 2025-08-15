#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from scipy.stats import kruskal, chi2, norm

TARGET_DEFAULT = "AGE"
DEFAULT_BINS = [0, 18, 45, 65, 75, 200]
METRICS_DEFAULT = [
    "encounter_count",
    "num_medications",
    "num_procedures",
    "num_immunizations",
    "num_devices",
]

DEFAULT_P_NORM = "arctan"   # 'arctan' / 'exp' / 'log1p'
DEFAULT_P_SCALE = 10.0
DEFAULT_P_CAP = 300.0

def find_col_case_insensitive(df: pd.DataFrame, name: str) -> str | None:
    low = name.lower()
    for c in df.columns:
        if c.lower() == low:
            return c
    return None

def make_age_groups_by_custom_bins(age_series: pd.Series, custom_bins: list[float]) -> pd.Series:
    age = pd.to_numeric(age_series, errors="coerce")
    bins = np.array(sorted(custom_bins), dtype=float)
    if len(bins) < 3:
        raise ValueError("--custom-bins は 3 個以上の境界が必要です（>=2 群）。")
    labels = []
    for i in range(len(bins) - 1):
        a, b = bins[i], bins[i + 1]
        labels.append(f"{int(a)}+" if i == len(bins) - 2 else f"{int(a)}–{int(b)-1}")
    finite_max = np.nanmax(age.values) if np.isfinite(np.nanmax(age.values)) else bins[-1]
    extended = bins.copy()
    extended[-1] = max(bins[-1], finite_max) + 1e-9
    return pd.cut(age, bins=extended, right=False, labels=labels, include_lowest=True)

def chi2_logp_safe(H: float, dfree: int) -> tuple[float, str]:
    logp = chi2.logsf(H, dfree)
    if np.isfinite(logp):
        mlog10p = -logp / np.log(10.0)
        p_str = f"1e-{mlog10p:.1f}" if mlog10p > 300 else f"{np.exp(logp):.3e}"
        return float(mlog10p), p_str

    v = float(dfree)
    w = (H / v) ** (1.0 / 3.0)
    mu = 1.0 - 2.0 / (9.0 * v)
    sigma = np.sqrt(2.0 / (9.0 * v))
    Z = (w - mu) / sigma
    logp_norm = norm.logsf(abs(Z)) + np.log(2.0)
    if np.isfinite(logp_norm):
        mlog10p = -logp_norm / np.log(10.0)
        p_str = f"1e-{mlog10p:.1f}" if mlog10p > 300 else f"{np.exp(logp_norm):.3e}"
        return float(mlog10p), p_str

    Z = abs(Z)
    mlog10p = (Z * Z) / (2.0 * np.log(10.0)) + (np.log(Z) + 0.5 * np.log(2.0 * np.pi)) / np.log(10.0)
    p_str = f"1e-{mlog10p:.1f}"
    return float(mlog10p), p_str

def eta2_kw(H: float, n_eff: int, k_used: int) -> float:
    if n_eff <= 1:
        return 0.0
    val = (H - (k_used - 1.0)) / (n_eff - 1.0)
    return float(np.clip(val, 0.0, 1.0))

def rank_eta2(y: np.ndarray, g_codes: np.ndarray, G: int) -> float:
    ranks = pd.Series(y).rank(method="average").to_numpy()
    ybar = ranks.mean()
    ssb = 0.0
    for c in range(G):
        mask = (g_codes == c)
        if mask.any():
            n_c = mask.sum()
            m_c = ranks[mask].mean()
            ssb += n_c * (m_c - ybar) ** 2
    sst = ((ranks - ybar) ** 2).sum()
    return float(ssb / sst) if sst > 0 else 0.0

def vargha_delaney_A(x: np.ndarray, y: np.ndarray) -> float:
    s = np.concatenate([x, y])
    r = pd.Series(s).rank(method="average").to_numpy()
    n1 = len(x)
    r1 = r[:n1].sum()
    A = (r1 - n1 * (n1 + 1) / 2.0) / (n1 * len(y))
    return float(np.clip(A, 0.0, 1.0))

def multi_group_A_metrics(groups: list[np.ndarray]) -> tuple[float, float]:
    A_sum = 0.0
    Asym_sum = 0.0
    w_sum = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            xi, xj = groups[i], groups[j]
            if len(xi) == 0 or len(xj) == 0:
                continue
            A = vargha_delaney_A(xi, xj)
            w = len(xi) * len(xj)
            A_sum += w * A
            Asym_sum += w * (2.0 * abs(A - 0.5))
            w_sum += w
    if w_sum == 0:
        return 0.5, 0.0
    return float(A_sum / w_sum), float(Asym_sum / w_sum)

def h_max_no_ties(counts: list[int] | np.ndarray) -> float:
    c = np.asarray(counts, dtype=float)
    n = c.sum()
    if n <= 1 or (c <= 0).any():
        return 0.0
    prefix = np.concatenate(([0.0], np.cumsum(c[:-1])))
    Rbar = prefix + (c + 1.0) / 2.0
    overall = (n + 1.0) / 2.0
    ssb = np.sum(c * (Rbar - overall) ** 2)
    Hmax = (12.0 / (n * (n + 1.0))) * ssb
    return float(max(Hmax, 0.0))

def normalize_mlog10p(x, method=DEFAULT_P_NORM, scale=DEFAULT_P_SCALE, cap=DEFAULT_P_CAP):
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x) & (x >= 0), x, 0.0)
    if method == "arctan":
        return (2.0 / np.pi) * np.arctan(x / float(scale))
    elif method == "exp":
        return 1.0 - np.exp(-x / float(scale))
    else:  # "log1p"
        cap = float(cap)
        return np.log1p(np.minimum(x, cap)) / np.log1p(cap)

def main():
    ap = argparse.ArgumentParser(
        description="Kruskal–Wallis（安定p・0～1効果量・Hの正規化）"
    )
    ap.add_argument("csv", help="入力CSV")
    ap.add_argument("--age-col", default=TARGET_DEFAULT, help="年齢列名（既定: AGE）")
    ap.add_argument("--metrics", default=",".join(METRICS_DEFAULT),
                    help="解析する指標列（カンマ区切り）既定: " + ",".join(METRICS_DEFAULT))
    ap.add_argument("--custom-bins", default="",
                    help=f"臨床カスタム区切り（例: 0,18,45,65,75,200）。未指定なら {DEFAULT_BINS} を使用")
    ap.add_argument("--min-per-group", type=int, default=2, help="各群の最小サンプル数（既定: 2）")
    ap.add_argument("--p-norm", choices=["arctan", "exp", "log1p"], default=DEFAULT_P_NORM,
                    help="minus_log10_p の 0–1 正規化方式（既定: arctan）")
    ap.add_argument("--p-scale", type=float, default=DEFAULT_P_SCALE,
                    help="arctan/exp のスケール（大きいほどゆっくり1に近づく）既定: 10")
    ap.add_argument("--p-cap", type=float, default=DEFAULT_P_CAP,
                    help="log1p 正規化の上限（既定: 300）")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)

    age_col = find_col_case_insensitive(df, args.age_col)
    if age_col is None:
        raise SystemExit(f"年齢列 {args.age_col} が見つかりません。")

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    metrics, not_found = [], []
    for m in metric_names:
        col = find_col_case_insensitive(df, m)
        (metrics if col is not None else not_found).append(col or m)
    if not metrics:
        raise SystemExit("解析対象の指標列が見つかりません。--metrics を確認してください。")
    if not_found:
        print(f"※ 見つからなかった列: {', '.join(not_found)}")

    custom_bins = DEFAULT_BINS if not args.custom_bins.strip() else [float(x) for x in args.custom_bins.split(",") if x.strip()]
    groups_ser = make_age_groups_by_custom_bins(df[age_col], custom_bins)

    rows = []
    for m in metrics:
        y_all = pd.to_numeric(df[m], errors="coerce")
        mask = y_all.notna() & groups_ser.notna()
        y = y_all[mask].to_numpy(dtype=float)
        g = pd.Categorical(groups_ser[mask])
        labels = list(g.categories.astype(str))
        k = len(labels)

        grp_vals = [y[g.codes == i] for i in range(k)]
        sizes = [len(v) for v in grp_vals]
        used_vals = [arr for arr in grp_vals if len(arr) >= args.min_per_group]
        used_labels = [lab for lab, arr in zip(labels, grp_vals) if len(arr) >= args.min_per_group]
        n_eff = sum(len(arr) for arr in used_vals)
        k_used = len(used_vals)

        if k_used < 2:
            rows.append({
                "metric": m,
                "group_sizes": "; ".join(f"{lab}:{sz}" for lab, sz in zip(labels, sizes)),
                "H_norm": np.nan,
                "p_value": "NA",
                "minus_log10_p_norm": np.nan,
                "epsilon2": np.nan, "eta2_kw": np.nan, "rank_eta2": np.nan,
                "A_pair_avg": np.nan, "A_pair_sym": np.nan,
#                "H_cdf": np.nan, "H_scaled_max": np.nan,
                "note": f"有効群不足（min_per_group={args.min_per_group}）"
            })
            continue

        # Kruskal–Wallis
        try:
            H, _ = kruskal(*used_vals)
        except Exception:
            rows.append({
                "metric": m,
                "group_sizes": "; ".join(f"{lab}:{sz}" for lab, sz in zip(labels, sizes)),
                "H_norm": np.nan,
                "p_value": "NA",
                "minus_log10_p_norm": np.nan,
                "epsilon2": np.nan, "eta2_kw": np.nan, "rank_eta2": np.nan,
                "A_pair_avg": np.nan, "A_pair_sym": np.nan,
#                "H_cdf": np.nan, "H_scaled_max": np.nan,
                "note": "kruskal計算エラー"
            })
            continue

        dfree = k_used - 1
        mlog10p, p_str = chi2_logp_safe(float(H), dfree)

        # 効果量
        eps = (H - dfree) / (n_eff - dfree) if (n_eff - dfree) > 0 else 0.0
        eps = float(np.clip(eps, 0.0, 1.0))
        eta = eta2_kw(float(H), int(n_eff), int(k_used))
        g_codes = np.concatenate([np.full(len(arr), i, dtype=int) for i, arr in enumerate(used_vals)])
        r_eta = rank_eta2(y=np.concatenate(used_vals), g_codes=g_codes, G=k_used)
        A_avg, A_sym = multi_group_A_metrics(used_vals)

        # H の 0-1 正規化
        H_cdf = float(chi2.cdf(H, dfree))
        Hmax = h_max_no_ties([len(arr) for arr in used_vals])
        H_scaled_max = float(H / Hmax) if Hmax > 0 else 0.0
        H_scaled_max = float(np.clip(H_scaled_max, 0.0, 1.0))
        H_norm = H_scaled_max

        # minus_log10_p の 0-1 正規化（飽和しにくい）
        pnorm = float(normalize_mlog10p(
            mlog10p, method=args.p_norm, scale=args.p_scale, cap=args.p_cap
        ))

        rows.append({
            "metric": m,
            "group_sizes": "; ".join(f"{lab}:{len(arr)}" for lab, arr in zip(used_labels, used_vals)),
            "H_norm": H_norm,
            "p_value": p_str,
            "minus_log10_p_norm": pnorm,
            "epsilon2": eps,
            "eta2_kw": eta,
            "rank_eta2": float(r_eta),
            "A_pair_avg": A_avg,
            "A_pair_sym": A_sym,
#            "H_cdf": H_cdf,
#            "H_scaled_max": H_scaled_max,
            "note": ""
        })

    # ★ 出力列：H と minus_log10_p は出力しない
    out = pd.DataFrame(rows, columns=[
        "metric",
        "group_sizes",
#        "H",
        "H_norm",
#        "p_value",
#        "minus_log10_p",
        "minus_log10_p_norm",
        "epsilon2", "eta2_kw", "rank_eta2", "A_pair_avg", "A_pair_sym",
#        "H_cdf", "H_scaled_max",
        "note"
    ])

    pd.set_option("display.max_columns", None)
    with pd.option_context('display.float_format', lambda x: f"{x:.6g}"):
        print("\n=== Kruskal–Wallis + 0–1 Effect Measures + H Normalization (custom age bins) ===")
        print(out.to_string(index=False))

if __name__ == "__main__":
    main()
