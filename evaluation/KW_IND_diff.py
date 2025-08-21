#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KW_IND_diff.py
- 2つのCSVを入力し、各CSVに対して KW_IND.py と同等の 0～1 指標群を算出
  （H_norm, minus_log10_p_norm, epsilon2, eta2_kw, rank_eta2, A_pair_avg, A_pair_sym）
- metric ごとに (file2 - file1) の差分を出力（group_sizesは出さない）
- NaN は 0 に置換
- 末尾に全差分の最大絶対値を出力

オプションは KW_IND.py と同様（age 列名、metrics、custom-bins 等）
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import kruskal, chi2, norm

# ==== 既定値（KW_IND.py と同じ） ====
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

# ==== ヘルパ（KW_IND.py 相当） ====

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

# ==== 1つのCSVに対して KW 指標を算出 ====

NUM_COLS = ["H_norm", "minus_log10_p_norm", "epsilon2", "eta2_kw", "rank_eta2", "A_pair_avg", "A_pair_sym"]

def compute_kw_table(csv_path: str,
                     age_col_name: str,
                     metrics_csv: str,
                     custom_bins_str: str,
                     min_per_group: int,
                     p_norm: str, p_scale: float, p_cap: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    age_col = find_col_case_insensitive(df, age_col_name)
    if age_col is None:
        raise SystemExit(f"[{csv_path}] 年齢列 {age_col_name} が見つかりません。")

    metric_names = [m.strip() for m in metrics_csv.split(",") if m.strip()]
    metrics, not_found = [], []
    for m in metric_names:
        col = find_col_case_insensitive(df, m)
        (metrics if col is not None else not_found).append(col or m)

    if not metrics:
        # どれも無ければ空表（あとで0埋め差分可）
        return pd.DataFrame(columns=["metric"] + NUM_COLS)

    custom_bins = DEFAULT_BINS if not custom_bins_str.strip() else [float(x) for x in custom_bins_str.split(",") if x.strip()]
    groups_ser = make_age_groups_by_custom_bins(df[age_col], custom_bins)

    rows = []
    for m in metrics:
        if m not in df.columns:
            # 無い metric は0埋め（差分用）
            rows.append({"metric": m, **{k: 0.0 for k in NUM_COLS}})
            continue

        y_all = pd.to_numeric(df[m], errors="coerce")
        mask = y_all.notna() & groups_ser.notna()
        y = y_all[mask].to_numpy(dtype=float)
        g = pd.Categorical(groups_ser[mask])
        labels = list(g.categories.astype(str))
        k = len(labels)

        # 各群
        grp_vals = [y[g.codes == i] for i in range(k)]
        used_vals = [arr for arr in grp_vals if len(arr) >= min_per_group]
        n_eff = sum(len(arr) for arr in used_vals)
        k_used = len(used_vals)

        if k_used < 2:
            rows.append({"metric": m, **{k: 0.0 for k in NUM_COLS}})
            continue

        # Kruskal–Wallis
        try:
            H, _ = kruskal(*used_vals)
        except Exception:
            rows.append({"metric": m, **{k: 0.0 for k in NUM_COLS}})
            continue

        dfree = k_used - 1
        mlog10p, _ = chi2_logp_safe(float(H), dfree)

        # 効果量
        eps = (H - dfree) / (n_eff - dfree) if (n_eff - dfree) > 0 else 0.0
        eps = float(np.clip(eps, 0.0, 1.0))
        eta = eta2_kw(float(H), int(n_eff), int(k_used))
        g_codes = np.concatenate([np.full(len(arr), i, dtype=int) for i, arr in enumerate(used_vals)])
        r_eta = rank_eta2(y=np.concatenate(used_vals), g_codes=g_codes, G=k_used)
        A_avg, A_sym = multi_group_A_metrics(used_vals)

        # H の 0-1 正規化（H/Hmax）
        Hmax = h_max_no_ties([len(arr) for arr in used_vals])
        H_scaled_max = float(H / Hmax) if Hmax > 0 else 0.0
        H_scaled_max = float(np.clip(H_scaled_max, 0.0, 1.0))

        # minus_log10_p の 0-1 正規化（飽和しにくい）
        pnorm = float(normalize_mlog10p(mlog10p, method=p_norm, scale=p_scale, cap=p_cap))

        rows.append({
            "metric": m,
            "H_norm": H_scaled_max,
            "minus_log10_p_norm": pnorm,
            "epsilon2": eps,
            "eta2_kw": eta,
            "rank_eta2": float(r_eta),
            "A_pair_avg": A_avg,
            "A_pair_sym": A_sym
        })

    out = pd.DataFrame(rows, columns=["metric"] + NUM_COLS)
    # 欠損は0埋め
    for c in NUM_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out = out.sort_values("metric", kind="mergesort").reset_index(drop=True)
    return out

def eval(path_to_csv1, path_to_csv2, print_details=False):
    # fix optional hyperparameters with their default values
    age_col = TARGET_DEFAULT
    metrics = ",".join(METRICS_DEFAULT)
    custom_bins = ""
    min_per_group = 2
    p_norm = DEFAULT_P_NORM
    p_scale = DEFAULT_P_SCALE
    p_cap = DEFAULT_P_CAP

    t1 = compute_kw_table(
        path_to_csv1, age_col, metrics, custom_bins,
        min_per_group, p_norm, p_scale, p_cap
    )
    t2 = compute_kw_table(
        path_to_csv2, age_col, metrics, custom_bins,
        min_per_group, p_norm, p_scale, p_cap
    )

    # metric の和集合で揃え、欠側は0埋め
    metrics_all = sorted(set(t1["metric"]).union(set(t2["metric"])))
    t1i = t1.set_index("metric").reindex(metrics_all).fillna(0.0)
    t2i = t2.set_index("metric").reindex(metrics_all).fillna(0.0)

    # 差分 (file2 - file1)
    diff = (t2i[NUM_COLS] - t1i[NUM_COLS]).reset_index()
    diff = diff.rename(columns={"index": "metric"}).sort_values("metric", kind="mergesort").reset_index(drop=True)

    # 表示（KW_IND 風ヘッダ、group_sizes なし）
    if print_details:
        with pd.option_context("display.max_columns", None,
                               "display.width", None,
                               "display.float_format", lambda x: f"{x:.6g}"):
            print("\n=== KW_IND Diff (file2 - file1) — H_norm / p_norm / effect sizes (0–1) ===")
            print(diff.to_string(index=False))

    # 全差分の最大絶対値
    max_abs = float(np.nanmax(np.abs(diff[NUM_COLS].to_numpy()))) if not diff.empty else 0.0
    if print_details:
        print(f"\nMAX_ABS_DIFF {max_abs:.6g}")

    # 保存オプション
    # if args.out:
    #     diff.to_csv(args.out, index=False)

    return max_abs

def main():
    ap = argparse.ArgumentParser(description="KW_IND 差分: 2つのCSVの 0～1 指標を (file2 - file1) で出力し、最後に最大絶対差を表示（group_sizes非表示）。")
    ap.add_argument("csv1", help="1つ目のCSV（baseline）")
    ap.add_argument("csv2", help="2つ目のCSV（to compare）")
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
    ap.add_argument("-o", "--out", default=None, help="差分テーブルをCSV保存するパス（任意）")
    args = ap.parse_args()

    t1 = compute_kw_table(
        args.csv1, args.age_col, args.metrics, args.custom_bins,
        args.min_per_group, args.p_norm, args.p_scale, args.p_cap
    )
    t2 = compute_kw_table(
        args.csv2, args.age_col, args.metrics, args.custom_bins,
        args.min_per_group, args.p_norm, args.p_scale, args.p_cap
    )

    # metric の和集合で揃え、欠側は0埋め
    metrics_all = sorted(set(t1["metric"]).union(set(t2["metric"])))
    t1i = t1.set_index("metric").reindex(metrics_all).fillna(0.0)
    t2i = t2.set_index("metric").reindex(metrics_all).fillna(0.0)

    # 差分 (file2 - file1)
    diff = (t2i[NUM_COLS] - t1i[NUM_COLS]).reset_index()
    diff = diff.rename(columns={"index": "metric"}).sort_values("metric", kind="mergesort").reset_index(drop=True)

    # 表示（KW_IND 風ヘッダ、group_sizes なし）
    with pd.option_context("display.max_columns", None,
                           "display.width", None,
                           "display.float_format", lambda x: f"{x:.6g}"):
        print("\n=== KW_IND Diff (file2 - file1) — H_norm / p_norm / effect sizes (0–1) ===")
        print(diff.to_string(index=False))

    # 全差分の最大絶対値
    max_abs = float(np.nanmax(np.abs(diff[NUM_COLS].to_numpy()))) if not diff.empty else 0.0
    print(f"\nMAX_ABS_DIFF {max_abs:.6g}")

    # 保存オプション
    if args.out:
        diff.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
