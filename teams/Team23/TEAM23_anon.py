import subprocess
import numpy as np
import pandas as pd
from scipy.stats import norm

def probs(df, column):
    _ = df[column].value_counts(normalize=True)
    df[column] = np.random.choice(_.index, size=len(df), p=_.values)
    return df

def threshold(df, column, step, low, high, noise_scale=0.05, z="int"):
    df[column] = df[column].clip(lower=low, upper=high)

    col_min, col_max = df[column].min(), df[column].max()
    df_norm = (df[column] - col_min) / (col_max - col_min)
    mean_val, std_val = df_norm.mean(), df_norm.std()

    n = len(df)
    k = int(n * 0.03)
    sorted_idx = df_norm.argsort()
    lower_idx = sorted_idx[:k]
    upper_idx = sorted_idx[-k:]
    replace_idx = np.concatenate([lower_idx, upper_idx])

    noises = np.random.normal(loc=0.0, scale=noise_scale, size=len(replace_idx))
    df_norm.iloc[replace_idx] = mean_val + noises

    df[column] = df_norm * (col_max - col_min) + col_min
    df[column] = (np.round((df[column] - low) / step) * step + low).clip(low, high).astype(z)
    return df

def probs_pair(df, group_cols, target_col):
    dist = (
        df.groupby(group_cols)[target_col]
          .value_counts(normalize=True)
          .rename("prob")
          .reset_index()
    )
    def sample(row):
        sub = dist
        for col in group_cols:
            sub = sub[sub[col] == row[col]]
        return np.random.choice(sub[target_col], p=sub["prob"])
    df[target_col] = df.apply(sample, axis=1)
    return df

def norm_distribution(df, columns, mus, scales, ranges, noise_range=(0.01, 10.93)):
    u = np.random.rand(len(df))
    df[columns[0]] = norm.ppf(u, loc=mus[0], scale=scales[0])
    df[columns[0]] = df[columns[0]].clip(ranges[0][0], ranges[0][1])
    
    base_B = norm.ppf(u, loc=mus[1], scale=scales[1])
    noise_scale = np.random.uniform(noise_range[0], noise_range[1], size=len(df))
    noise_sign = np.random.choice([-1, 1], size=len(df))
    noise = noise_sign * np.random.normal(0, noise_scale)
    
    df[columns[1]] = base_B + noise
    df[columns[1]] = df[columns[1]].clip(ranges[1][0], ranges[1][1])
    return df

def norm2(loc, scale, steps, size=0, rng=None):
    dist = norm.rvs(loc=loc, scale=scale, size=int(1e4-size))
    dist = np.round(dist/steps) * steps
    if size != 0:
        u = np.random.uniform(rng[0], rng[1], size)
        u = np.round(u/steps) * steps
        dist = np.concatenate([dist, u])
    return dist

def calc_BMI(df):
    df_cpy = df.copy()

    prmh = [1.68, 0.1337758856, 0.05, 1900, [0.50,1.50]]
    prmb = [31.294998, 8.446967482-6.8, 0.05]

    heights = norm2(prmh[0], prmh[1], prmh[2], size=prmh[3], rng=prmh[4])
    mask = df["num_immunizations"] > 20
    heights[mask] = np.random.uniform(0.5, 1.0, size=mask.sum())
    df_cpy["mean_height"] = heights

    noise_scale = np.random.uniform(0.50, 3.50, size=len(df))
    noise_sign = np.random.choice([-1, 1], size=len(df))
    noise = noise_sign * np.random.normal(0, noise_scale)    
    df["mean_bmi"] = norm2(prmb[0], prmb[1], prmb[2]) + noise

    df["mean_weight"] = (df_cpy["mean_height"]) ** 2 * df["mean_bmi"]
    mean_val = df["mean_weight"].mean()
    df["mean_weight"] = (df["mean_weight"] - mean_val) * (24.74 / df["mean_weight"].std()) + mean_val
    return df

def category(df):
    #1 GENDER, RACE, ALLERGIES < 元データ依存なし
    df["GENDER"] = np.random.choice(["F", "M"], size=len(df))
    df["RACE"] = "white"
    df["num_allergies"] = 0

    #2 ETHNICITY, ASTHMA, DEPRESSION < 割合依存
    df = probs(df, "ETHNICITY")
    df = probs(df, "asthma_flag")
    df = probs(df, "depression_flag")
    
    #3 前処理
    df = threshold(df, "AGE", 2, 10, 80)
    df = threshold(df, "encounter_count", 2, 10, 100)
    df = threshold(df, "num_procedures", 2, 0, 200)
    df = threshold(df, "num_medications", 2, 0, 150)
    df = threshold(df, "num_immunizations", 2, 5, 40)
    df = threshold(df, "num_devices", 1, 0, 10)

    #4 ENCOUNTER, PROCEDURES, MEDICATIONS, IMMUNIZATIONS, DEVICES, STROKE, OBESITY < 年代割合依存
    df = probs_pair(df, ["AGE", "num_procedures", "encounter_count"], "num_medications")
    # df = probs_pair(df, ["AGE", "encounter_count"], "num_medications")
    # df = probs_pair(df, ["AGE", "num_procedures"], "num_immunizations")
    # df = probs_pair(df, ["AGE"], "num_devices")
    df = probs_pair(df, ["AGE"], "stroke_flag")
    df = probs_pair(df, ["AGE"], "obesity_flag")

    #5 SYSTOLIC, DIASTOLIC < 年代正規分布
    cols = ["mean_systolic_bp","mean_diastolic_bp", "mean_bmi", "mean_weight"]
    df = norm_distribution(df, cols[:2], [118.63, 79.31], [15, 8], [[56.81,168.73], [35.33,128.09]])
    df = calc_BMI(df)
    df[cols] = df[cols].applymap(lambda x: f"{x:.2f}")

    return df

m = 2
B_path = f"BB23_{m}.csv"
C_path = B_path.replace("BB", "CC")
B = category(pd.read_csv(B_path))
B.to_csv(C_path, index=None)

subprocess.run([
    "python",
    "sample/util/check_csv.py",
    C_path,
    "sample/data/columns_range.json"
])

subprocess.run([
    "python",
    "programs/eval_all.py",
    B_path,
    C_path
])