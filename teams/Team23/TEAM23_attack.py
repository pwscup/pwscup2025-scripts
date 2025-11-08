import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist

def csv(path):
    cols = [
        "AGE", "encounter_count", "num_procedures", "num_medications",
        "num_immunizations", "num_devices", "mean_height"
    ]
    df = pd.read_csv(path)
    df["mean_height"] = np.sqrt(df["mean_weight"] / df["mean_bmi"])
    return df[cols]

def min_max(path1, path2):
    df1 = csv(path1)
    df2 = csv(path2)
    min_vals = df1.min()
    max_vals = df1.max()
    df1_range = (max_vals - min_vals).replace(0, 1)
    df1_norm = 2 * (df1 - min_vals) / df1_range - 1
    df2_norm = 2 * (df2 - min_vals) / df1_range - 1
    return df1_norm, df2_norm

def dsort(df1, df2, cols):
    sdf1 = df1.sort_values(by=cols, ascending=True, ignore_index=True)
    sdf2 = df2.sort_values(by=cols, ascending=True, ignore_index=True)
    return sdf1, sdf2

def bcorr(df, cols, p=0.05):
    n = len(df)
    block_size = max(1, int(n * p))
    corrs = []
    for i in range(0, n, block_size):
        block = df.iloc[i:i + block_size]
        x = block[cols[:-1]].sum(axis=1)
        corr, _ = spearmanr(block[cols[-1]], x)
        corrs.append(corr)
    return corrs

# def reorder(df, cols, corrs, n=100):
#     mean = np.nanmean(corrs)
#     df_cpy = df.copy()
#     best = -np.inf
#     for _ in range(n):
#         shuffled = df.copy()
#         shuffled[cols[-1]] = np.random.permutation(shuffled[cols[-1]])
#         x = shuffled[cols[:-1]].sum(axis=1)
#         corr, _ = spearmanr(shuffled[cols[-1]], x)
#         if abs(corr - mean) < abs(best - mean):
#             best, df_cpy = corr, shuffled
#     return df_cpy

def reorder(df, cols, corrs, replace_ratio=0.1, n=100):
    mean = np.nanmean(corrs)
    target_col = cols[-1]
    df_cpy = df.copy()

    best_corr = np.inf
    best_df = df_cpy.copy()

    m = len(df)
    n_replace = max(1, int(m * replace_ratio))

    for _ in range(n):
        temp_df = df.copy()
        idx = np.random.choice(m, n_replace, replace=False)
        swap_values = np.random.choice(df[target_col].values, n_replace, replace=False)
        temp_df.loc[idx, target_col] = swap_values
        x = temp_df[cols[:-1]].sum(axis=1)
        corr, _ = spearmanr(temp_df[target_col], x)
        if abs(corr - mean) < abs(best_corr - mean):
            best_corr = corr
            best_df = temp_df.copy()
    return best_df

def attack(df1, df2, thres=1.0):
    cov_inv = np.linalg.pinv(np.cov(df1.values.T))
    dists = cdist(df1.values, df2.values, metric="mahalanobis", VI=cov_inv)
    min_vals = dists.min(axis=0)
    idx = dists.argmin(axis=0)
    idx_filtered = idx[min_vals <= thres]
    return idx_filtered

def process_block(df1, block, df2, cols, p, thres):
    corrs = bcorr(block, cols, p)
    reordered = reorder(df2, cols, corrs)
    idx = attack(df1, reordered, thres)
    return idx.tolist()

def main(df1, df2, p=0.05, thres=1.0):
    cols = np.random.choice(df1.columns.to_list(), 4, replace=False)
    n = len(df1)
    block_size = max(1, int(n * p))
    blocks = [df1.iloc[i:i + block_size] for i in range(0, n, block_size)]

    results = []
    for block in tqdm(blocks):
        idx = process_block(df1, block, df2, cols, p, thres)
        results.extend(idx)

    all_idx = list(set(results))
    return all_idx

if __name__ == "__main__":
    i = 20
    a_path = f"pre_data/A/A{i:02d}.csv"
    c_path = f"pre_data/C/C{i:02d}.csv"
    r_path = f"pre_data/R/R{i:02d}.csv"
    z_path = f"pre_data/Z/Z{i:02d}.csv"
    
    a, c = min_max(a_path, c_path)
    result = main(a, c)

    flag = np.zeros(len(a), dtype=int)
    flag[np.unique(result)] = 1
    r = pd.DataFrame(flag)
    r.to_csv(r_path, header=None, index=None)

    z = pd.read_csv(z_path, header=None).squeeze("columns").to_numpy()
    r = pd.read_csv(r_path, header=None).squeeze("columns").to_numpy()
    idx = np.where(z == 1)[0]
    com = (z[idx] == r[idx]).astype(int)
    print(len(set(result)), sum(com))
