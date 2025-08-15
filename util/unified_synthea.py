#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
import re

# 使い方:
#   python3 unified_synthea.py <output.csv>

if len(sys.argv) < 2:
    print("Usage: python3 unified_synthea.py <output.csv>")
    sys.exit(1)

output_csv_file = sys.argv[1]

# --- 入力CSV群を同一ディレクトリ前提で読み込み ---
patients      = pd.read_csv("patients.csv")
encounters    = pd.read_csv("encounters.csv")
conditions    = pd.read_csv("conditions.csv")
procedures    = pd.read_csv("procedures.csv")
observations  = pd.read_csv("observations.csv")
immunizations = pd.read_csv("immunizations.csv")
medications   = pd.read_csv("medications.csv")
allergies     = pd.read_csv("allergies.csv")
devices       = pd.read_csv("devices.csv")

# --- 集計のベース（encounters × patients） ---
base = encounters.merge(patients, left_on="PATIENT", right_on="Id", how="left")

# 各種件数
encounter_count    = base.groupby("PATIENT").size().reset_index(name="encounter_count")
procedure_count    = procedures.groupby("PATIENT").size().reset_index(name="num_procedures")
medication_count   = medications.groupby("PATIENT").size().reset_index(name="num_medications")
immunization_count = immunizations.groupby("PATIENT").size().reset_index(name="num_immunizations")
allergy_count      = allergies.groupby("PATIENT").size().reset_index(name="num_allergies")
device_count       = devices.groupby("PATIENT").size().reset_index(name="num_devices")

# --- 疾患フラグ作成（conditions + medications の複合判定） ---
chronic_flags = conditions.copy()
chronic_flags["DESCRIPTION"] = chronic_flags["DESCRIPTION"].astype(str)
chronic_flags["CODE"]        = chronic_flags["CODE"].astype(str)

# Stroke（TIA含む）
stroke_pattern = re.compile(
    r"(?:stroke|cerebrovascular|TIA|transient ischemic attack|ischemic|hemorrhag(?:e|ic))",
    re.IGNORECASE,
)
stroke_snomed = {
    "422504002",  # Ischemic stroke
    "230690007",  # Hemorrhagic stroke
    "266257000",  # Transient ischemic attack
}

# Depression（語彙拡張）
depression_pattern = re.compile(
    r"(?:"
    r"depress(?:ion|ive)\b|"
    r"major\s+depressive\b|"
    r"recurrent\s+depress(?:ion|ive)\b|"
    r"persistent\s+depress(?:ive\s+disorder|ion)\b|"
    r"dysthymi(?:a|c)\b|"
    r"melancholia\b|"
    r"post(?:partum|natal)\s+depress(?:ion|ive)\b|"
    r"peripartum\s+depress(?:ion|ive)\b|"
    r"(?:seasonal\s+affective|SAD)\s+disorder\b|"
    r"adjustment\s+disorder(?:.*)depress(?:ed|ive)\s+mood\b|"
    r"depressive\s+episode\b|"
    r"single\s+episode\s+depress(?:ion|ive)\b|"
    r"MDD\b"
    r")",
    re.IGNORECASE,
)

# Asthma（語彙拡張・誤記含む）
asthma_pattern = re.compile(
    r"(?:"
    r"asthma\b|asthema\b|"
    r"status\s+asthmaticus\b|"
    r"exercise[-\s]?induced\s+asthma\b|"
    r"cough[-\s]?variant\s+asthma\b|"
    r"allergic\s+asthma\b|"
    r"(?:mild|moderate|severe)(?:\s+|-)persistent(?:\s+asthma)?\b|"
    r"mild(?:\s+|-)intermittent(?:\s+asthma)?\b"
    r")",
    re.IGNORECASE,
)

# conditions ベースのフラグ
chronic_flags["asthma_flag"]     = chronic_flags["DESCRIPTION"].str.contains(asthma_pattern,     na=False).astype(int)
chronic_flags["stroke_flag"]     = (
    chronic_flags["DESCRIPTION"].str.contains(stroke_pattern, na=False)
    | chronic_flags["CODE"].isin(stroke_snomed)
).astype(int)
chronic_flags["obesity_flag"]    = chronic_flags["DESCRIPTION"].str.contains("obesity", case=False, na=False).astype(int)
chronic_flags["depression_flag"] = chronic_flags["DESCRIPTION"].str.contains(depression_pattern,  na=False).astype(int)

# --- medications による補強 ---
meds = medications.copy()
meds["DESCRIPTION"] = meds["DESCRIPTION"].astype(str)

antidepressants = re.compile(
    r"(?:fluoxetine|paroxetine|sertraline|citalopram|escitalopram|fluvoxamine|"
    r"venlafaxine|desvenlafaxine|duloxetine|milnacipran|levomilnacipran|"
    r"bupropion|mirtazapine|trazodone|vortioxetine|vilazodone|agomelatine|"
    r"amitriptyline|nortriptyline|imipramine|clomipramine|desipramine|doxepin|trimipramine|"
    r"phenelzine|tranylcypromine|isocarboxazid|moclobemide)",
    re.IGNORECASE,
)

asthma_meds = re.compile(
    r"(?:albuterol|salbutamol|levalbuterol|terbutaline|ipratropium|"
    r"salmeterol|formoterol|vilanterol|indacaterol|olodaterol|"
    r"fluticasone|budesonide|beclomethasone|mometasone|ciclesonide|"
    r"tiotropium|umeclidinium|glycopyrronium|aclidinium|"
    r"montelukast|zafirlukast|zileuton|"
    r"theophylline|aminophylline|"
    r"budesonide[-\s]?formoterol|fluticasone[-\s]?salmeterol|fluticasone[-\s]?vilanterol|mometasone[-\s]?formoterol|beclo?metasone[-\s]?formoterol|"
    r"omalizumab|mepolizumab|reslizumab|benralizumab|dupilumab|tezepelumab)",
    re.IGNORECASE,
)

dep_pat = meds[meds["DESCRIPTION"].str.contains(antidepressants, na=False)]["PATIENT"].unique()
ast_pat = meds[meds["DESCRIPTION"].str.contains(asthma_meds,     na=False)]["PATIENT"].unique()

# conditions で 0 でも薬があれば 1 に引き上げ
chronic_flags.loc[chronic_flags["PATIENT"].isin(dep_pat), "depression_flag"] = 1
chronic_flags.loc[chronic_flags["PATIENT"].isin(ast_pat), "asthma_flag"]     = 1

# 患者単位にまとめる
chronic_summary = (
    chronic_flags.groupby("PATIENT")[["asthma_flag", "stroke_flag", "obesity_flag", "depression_flag"]]
    .max()
    .reset_index()
)

# --- 観測値（バイタル・検査） ---
obs = observations[["PATIENT", "DESCRIPTION", "VALUE"]].copy()
obs["VALUE"] = pd.to_numeric(obs["VALUE"], errors="coerce")

def extract_mean(desc_keyword: str) -> pd.DataFrame:
    return (
        obs[obs["DESCRIPTION"].str.contains(desc_keyword, case=False, na=False)]
        .groupby("PATIENT")["VALUE"]
        .mean()
        .reset_index()
    )

mean_systolic  = extract_mean("systolic")
mean_diastolic = extract_mean("diastolic")
mean_bmi       = extract_mean("body mass index")
mean_weight    = extract_mean("body weight")

# --- 最終統合 ---
df = (
    patients[["Id", "GENDER", "BIRTHDATE", "RACE", "ETHNICITY"]]
    .rename(columns={"Id": "PATIENT"})
    .merge(encounter_count, on="PATIENT", how="left")
    .merge(procedure_count, on="PATIENT", how="left")
    .merge(medication_count, on="PATIENT", how="left")
    .merge(immunization_count, on="PATIENT", how="left")
    .merge(allergy_count, on="PATIENT", how="left")
    .merge(device_count, on="PATIENT", how="left")
    .merge(chronic_summary, on="PATIENT", how="left")
    .merge(mean_systolic.rename(columns={"VALUE": "mean_systolic_bp"}), on="PATIENT", how="left")
    .merge(mean_diastolic.rename(columns={"VALUE": "mean_diastolic_bp"}), on="PATIENT", how="left")
    .merge(mean_bmi.rename(columns={"VALUE": "mean_bmi"}), on="PATIENT", how="left")
    .merge(mean_weight.rename(columns={"VALUE": "mean_weight"}), on="PATIENT", how="left")
)

# --- 年齢（誕生日を過ぎたかどうかで厳密に計算） ---
bd = pd.to_datetime(df["BIRTHDATE"], errors="coerce")
today = pd.Timestamp.today()
had_birthday = (bd.dt.month < today.month) | ((bd.dt.month == today.month) & (bd.dt.day <= today.day))
AGE = (today.year - bd.dt.year - (~had_birthday).astype("Int64")).astype("Int64")
df["AGE"] = AGE

# --- 小数は2桁丸め（AGEなど整数は対象外） ---
for col in df.select_dtypes(include=["float", "float64"]).columns:
    df[col] = df[col].round(2)

# --- 列並び：BIRTHDATE の位置に AGE を置き、BIRTHDATE は出力しない ---
cols = list(df.columns)
if "BIRTHDATE" in cols:
    bidx = cols.index("BIRTHDATE")
    cols.remove("BIRTHDATE")      # BIRTHDATE を除去
    # いったん AGE を末尾から外して所定位置へ
    cols.remove("AGE")
    cols.insert(bidx, "AGE")
    df = df[cols]

# 出力前に内部キーを落とす
df = df.drop(columns=["PATIENT"])

df.to_csv(output_csv_file, index=False)
