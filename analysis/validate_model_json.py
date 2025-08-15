#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_model_json.py (v2)
- attributes をトップレベル / learner.attributes の両方で許容
- attributes.feature_names は JSON文字列（復元して配列）
- attributes.target, attributes.xgboost_version を必須
- len(feature_names) == learner.learner_model_param.num_feature
- objective == binary:logistic, trees >= 1
- learner.feature_names があれば attributes と完全一致を要求
"""

import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+([.-].+)?$")

def get_path(d: Dict[str, Any], path: List[str]) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(".".join(path))
        cur = cur[k]
    return cur

def ok(msg: str): print(f"[OK]  {msg}")
def warn(msg: str): print(f"[WARN]{msg}")
def err(msg: str): print(f"[ERR] {msg}")

def validate_one(path: str) -> bool:
    errors: List[str] = []
    warnings: List[str] = []

    # 0) load
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception as e:
        err(f"{path}: JSONとして読み込めません: {e}")
        return False

    # 1) attributes: top-level or learner.attributes を許容
    attrs_src = "top"
    attrs = j.get("attributes")
    if not isinstance(attrs, dict):
        attrs = j.get("learner", {}).get("attributes")
        attrs_src = "learner"
    if not isinstance(attrs, dict):
        errors.append("attributes セクションがありません（top または learner.attributes のどちらにも見つからず）")
        attrs = {}

    # 2) feature_names（JSON文字列必須、復元して配列）
    F: Optional[List[str]] = None
    fn_raw = attrs.get("feature_names")
    if not isinstance(fn_raw, str):
        errors.append('attributes.feature_names は JSON文字列である必要があります（例: "[\\"AGE\\",...]"）')
    else:
        try:
            F_loaded = json.loads(fn_raw)
            if not isinstance(F_loaded, list):
                errors.append("attributes.feature_names を json.loads した結果が配列ではありません")
            elif not F_loaded or not all(isinstance(x, str) and x != "" for x in F_loaded):
                errors.append("feature_names 配列が空、または非文字列/空文字を含みます")
            elif len(F_loaded) != len(set(F_loaded)):
                errors.append("feature_names に重複があります")
            else:
                F = F_loaded
        except Exception as e:
            errors.append(f"attributes.feature_names が正しく JSON として復元できません: {e}")

    # 3) target
    tgt = attrs.get("target")
    if not (isinstance(tgt, str) and tgt):
        errors.append("attributes.target が見つからないか、空文字です")

    # 4) xgboost_version
    xv = attrs.get("xgboost_version") or attrs.get("sgboost_version")  # 誤記救済
    if not (isinstance(xv, str) and xv):
        errors.append("attributes.xgboost_version が見つからないか、空文字です")
    elif not SEMVER_RE.match(xv):
        warnings.append(f"xgboost_version の形式が semver と一致しません: '{xv}'（例: 1.7.6）")

    # 5) learner 側
    try:
        lmp = get_path(j, ["learner", "learner_model_param"])
    except KeyError:
        errors.append("learner.learner_model_param が見つかりません")
        lmp = {}
    # num_feature
    nf = None
    nf_raw = lmp.get("num_feature")
    if isinstance(nf_raw, (int, float, str)):
        try:
            nf = int(nf_raw)
            if nf <= 0:
                errors.append("learner.learner_model_param.num_feature は正の整数である必要があります")
        except Exception:
            errors.append("learner.learner_model_param.num_feature を整数に解釈できません")
    else:
        errors.append("learner.learner_model_param.num_feature が数値（または数値文字列）ではありません")

    # objective
    try:
        obj_name = get_path(j, ["learner", "objective", "name"])
        if obj_name != "binary:logistic":
            errors.append(f'learner.objective.name は "binary:logistic" 必須（現値: {obj_name!r}）')
    except KeyError:
        errors.append("learner.objective.name が見つかりません")

    # trees
    try:
        trees = get_path(j, ["learner", "gradient_booster", "model", "trees"])
        if not (isinstance(trees, list) and len(trees) >= 1):
            errors.append("learner.gradient_booster.model.trees は長さ>=1の配列である必要があります")
    except KeyError:
        errors.append("learner.gradient_booster.model.trees が見つかりません")

    # 6) 一貫性: len(F) == num_feature
    if (F is not None) and (nf is not None) and (len(F) != nf):
        errors.append(f"len(feature_names) と num_feature が不一致: len(F)={len(F)} vs num_feature={nf}")

    # learner.feature_names がある場合は一致要求
    try:
        lfeat = get_path(j, ["learner", "feature_names"])
        if isinstance(lfeat, list) and F is not None:
            if lfeat != F:
                errors.append("learner.feature_names と attributes.feature_names の配列順が一致しません")
    except KeyError:
        pass

    # 7) レポート
    if errors:
        err(f"{path}: 仕様違反 {len(errors)} 件")
        for e in errors: err(f"  - {e}")
        for w in warnings: import sys; print(f"[WARN]  - {w}")
        return False
    else:
        ok_msg = f"{path}: OK"
        if F is not None and nf is not None and tgt:
            ok_msg += f" (#features={nf}, target={tgt}, attrs_src={attrs_src})"
        ok(ok_msg)
        for w in warnings: warn(f"  - {w}")
        return True

def main():
    ap = argparse.ArgumentParser(description="Validate model JSON format for xgbt_pred.py (contest strict spec; attrs top/learner both OK).")
    ap.add_argument("files", nargs="+", help="model JSON file(s)")
    args = ap.parse_args()
    all_ok = True
    for p in args.files:
        all_ok &= validate_one(p)
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
