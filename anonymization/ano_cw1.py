# synth_generate.py
import numpy as np
import pandas as pd
from scipy.stats import norm
import os

# --- ユーティリティ関数 ---
def make_age_group(age_series):
    bins = [ -1, 17, 44, 64, 74, 200 ]  # 年齢のビン境界
    labels = ['0-17','18-44','45-64','65-74','75+']
    return pd.cut(age_series, bins=bins, labels=labels)

def empirical_u_from_series(x):
    # returns uniform ranks in (0,1) matching empirical CDF
    # use (rank-0.5)/n to avoid 0 or 1
    n = x.size
    ranks = x.rank(method='average')
    u = (ranks - 0.5) / n
    # ensure strictly inside (eps,1-eps)
    eps = 1e-6
    u = np.clip(u, eps, 1-eps)
    return u

def build_empirical_inverse(values):
    # returns a function that maps u in [0,1] to empirical quantile of values
    vals = np.sort(values)
    n = vals.size
    # empirical probabilities (use midpoints)
    probs = (np.arange(1, n+1) - 0.5) / n
    def inv(u):
        # u can be scalar or array
        u = np.asarray(u)
        # clip
        u = np.clip(u, probs[0], probs[-1])
        return np.interp(u, probs, vals)
    return inv

def nearest_pos_def(mat):
    # simple stabilizer: add epsilon to diagonal until pos-def
    eps = 1e-8
    k = 0
    while True:
        try:
            # try cholesky
            np.linalg.cholesky(mat + eps*np.eye(mat.shape[0]))
            return mat + eps*np.eye(mat.shape[0])
        except np.linalg.LinAlgError:
            eps *= 10
            k += 1
            if k > 10:
                # fallback: add large regularization
                return mat + (1e-2)*np.eye(mat.shape[0])

def fit_gaussian_copula_sampler(df_numeric):
    # df_numeric: pandas DataFrame of numeric columns
    # Returns a function sample(m) -> DataFrame of generated numeric columns
    cols = df_numeric.columns.tolist()
    n_rows = len(df_numeric)
    if n_rows == 0:
        raise ValueError("Empty dataframe passed to copula fitter.")
    # For each column, build inverse empirical CDF
    inv_cdfs = {}
    const_cols = set()
    for c in cols:
        vals = df_numeric[c].values
        if np.all(vals == vals[0]):
            const_cols.add(c)
        inv_cdfs[c] = build_empirical_inverse(vals)

    # transform to gaussian space
    Z = []
    for c in cols:
        if c in const_cols:
            # constant column -> use zeros
            Z.append(np.zeros(n_rows))
            continue
        u = empirical_u_from_series(df_numeric[c])
        z = norm.ppf(u)
        Z.append(z)
    Z = np.vstack(Z).T  # shape (n_rows, n_cols)
    # compute correlation matrix
    cov = np.corrcoef(Z, rowvar=False)
    # np.corrcoef can produce NaNs when a column has zero variance; replace NaNs with 0
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    # ensure diagonal is 1 (self-correlation)
    np.fill_diagonal(cov, 1.0)
    cov = nearest_pos_def(cov)
    # precompute cholesky, with a small regularization fallback if needed
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # add tiny regularization and retry
        eps = 1e-8
        cov_reg = cov + eps * np.eye(cov.shape[0])
        cov_reg = nearest_pos_def(cov_reg)
        L = np.linalg.cholesky(cov_reg)

    def sampler(m, random_state=None):
        rng = np.random.RandomState(random_state)
        k = len(cols)
        # sample independent normals
        U = rng.normal(size=(m, k))
        # introduce correlation
        Zs = U.dot(L.T)  # shape (m,k)
        # map back to uniform
        Us = norm.cdf(Zs)
        # map to original via inverse empirical
        out = np.zeros_like(Us)
        for j, c in enumerate(cols):
            if c in const_cols:
                out[:, j] = df_numeric[c].values[0]
            else:
                out[:, j] = inv_cdfs[c](Us[:, j])
        out_df = pd.DataFrame(out, columns=cols)
        return out_df

    return sampler

# --- メイン処理 ---
def synthesize(csv_path, out_path='synthetic.csv', random_seed=1):
    np.random.seed(random_seed)
    df = pd.read_csv(csv_path)

    # 1) AGE_GROUP を作成
    df['AGE_GROUP'] = make_age_group(df['AGE'])

    # 列の分類（元仕様に合わせる）
    categorical_cols = ['GENDER','RACE','ETHNICITY','AGE_GROUP']
    # 数値列は上記以外で、IDなどが無ければ全て数値とする
    numeric_cols = [c for c in df.columns if c not in categorical_cols]

    # 2) 元データの集計・統計を表示（必要な統計はここで計算できます）
    print("原データ行数:", len(df))
    print("\n各カテゴリ列の比率（サンプル）:")
    for c in categorical_cols:
        print(c)
        print(df[c].value_counts(normalize=True).to_string())
        print()

    print("数値列の要約（min,max,mean,std,25%,50%,75%）:")
    print(df[numeric_cols].describe().T[['min','max','mean','std','25%','50%','75%']])

    print("\n数値間のPearson相関行列（サンプル）:")
    print(df[numeric_cols].corr(method='pearson').round(3))

    # 3) カテゴリ結合分布（joint）に従ってカテゴリ列を合成（カテゴリ×カテゴリの比率維持）
    joint = df[categorical_cols].value_counts(normalize=True).rename('prob').reset_index()
    # prepare sampling table: joint combinations and probs
    # sample N rows
    N = len(df)
    rng = np.random.RandomState(random_seed)
    probs = joint['prob'].values
    choices = rng.choice(len(joint), size=N, p=probs)
    sampled_joint = joint.iloc[choices][categorical_cols].reset_index(drop=True)

    # 4) 数値列の合成（AGE_GROUPごとにcopulaを作って生成）
    synth_numeric_parts = []
    # 生成は AGE_GROUP ごとにまとめて行い、後で sampled_joint にマージする
    sample_counts_by_group = sampled_joint['AGE_GROUP'].value_counts().to_dict()
    # prepare original grouped numeric dfs
    numeric_df = df[numeric_cols].copy()
    df_with_numeric = df[categorical_cols + numeric_cols].copy()

    # for each AGE_GROUP, fit sampler and produce required number
    generated_numeric_list = []
    for grp, count in sample_counts_by_group.items():
        # get original rows with this group
        sub = df_with_numeric[df_with_numeric['AGE_GROUP'] == grp][numeric_cols]
        if len(sub) < 5:
            # データが少ない場合は全体のデータを用いる（fallback）
            sub = df_with_numeric[numeric_cols]
        sampler = fit_gaussian_copula_sampler(sub)
        gen = sampler(count, random_state=rng.randint(0,999999))
        # record AGE_GROUP value for alignment
        gen['_AGE_GROUP_'] = grp
        generated_numeric_list.append(gen)

    # concat generated numerics and shuffle within groups to match sampled_joint order
    generated_numeric = pd.concat(generated_numeric_list, ignore_index=True)
    # Now we need to align rows: for each AGE_GROUP in sampled_joint (in order), pick one row from generated_numeric with same _AGE_GROUP_
    # To do this deterministically, group generated_numeric by _AGE_GROUP_ and pop rows sequentially
    grouped_gen = {grp: g.drop(columns=['_AGE_GROUP_']).reset_index(drop=True) for grp, g in generated_numeric.groupby('_AGE_GROUP_')}
    # create final numeric rows in same order as sampled_joint
    numeric_rows = []
    ptrs = {grp:0 for grp in grouped_gen.keys()}
    for idx, row in sampled_joint.iterrows():
        grp = row['AGE_GROUP']
        if grp not in grouped_gen:
            # fallback to any group
            grp = list(grouped_gen.keys())[0]
        p = ptrs[grp]
        # if exhausted, wrap around
        if p >= len(grouped_gen[grp]):
            p = 0
        numeric_rows.append(grouped_gen[grp].iloc[p].to_dict())
        ptrs[grp] = p + 1
    synth_numeric_df = pd.DataFrame(numeric_rows, columns=numeric_cols)

    # 5) 結合して完成
    synthetic_df = pd.concat([sampled_joint.reset_index(drop=True), synth_numeric_df.reset_index(drop=True)], axis=1)

    # 6) 列ごとの丸め（元仕様に従う）
    # 元仕様の小数桁：
    int_cols = ['AGE','encounter_count','num_procedures','num_medications',
                'num_immunizations','num_allergies','num_devices',
                'asthma_flag','stroke_flag','obesity_flag','depression_flag']
    # make sure columns exist
    int_cols = [c for c in int_cols if c in synthetic_df.columns]
    synthetic_df[int_cols] = synthetic_df[int_cols].round(0).astype(int)

    # float columns to 2 decimals
    float_cols = ['mean_systolic_bp','mean_diastolic_bp','mean_bmi','mean_weight']
    float_cols = [c for c in float_cols if c in synthetic_df.columns]
    synthetic_df[float_cols] = synthetic_df[float_cols].round(2)

    # AGE should be in [2,110], clip
    if 'AGE' in synthetic_df.columns:
        synthetic_df['AGE'] = synthetic_df['AGE'].clip(lower=2, upper=110).astype(int)

    # 7) 出力: 評価ツールのフォーマットに合わせて AGE_GROUP 列は出力に含めない
    if 'AGE_GROUP' in synthetic_df.columns:
        out_df = synthetic_df.drop(columns=['AGE_GROUP'])
    else:
        out_df = synthetic_df
    out_df.to_csv(out_path, index=False)
    print(f"\n合成データを {out_path} に出力しました。行数: {len(out_df)}")

    # 8) 合成データの統計を表示して比較
    print("\n--- 合成データの要約 ---")
    print(synthetic_df[numeric_cols].describe().T[['min','max','mean','std','25%','50%','75%']])

    print("\n合成データの数値相関行列（Pearson）:")
    print(synthetic_df[numeric_cols].corr(method='pearson').round(3))

    print("\n合成データのカテゴリ比率（サンプル）:")
    for c in categorical_cols:
        print(c)
        print(synthetic_df[c].value_counts(normalize=True).to_string())
        print()

    return synthetic_df

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic medical data preserving basic statistics.')
    parser.add_argument('input_csv', help='Input CSV file path')
    parser.add_argument('output_csv', nargs='?', default='synthetic.csv', help='Output CSV file path (default: synthetic.csv)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"入力ファイル {args.input_csv} が見つかりません。パスを確認してください。")
    else:
        synth = synthesize(args.input_csv, out_path=args.output_csv, random_seed=args.seed)