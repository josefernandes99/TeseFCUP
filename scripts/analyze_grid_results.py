#!/usr/bin/env python3
import json
import os
import sys
from collections import defaultdict

# Ensure this directory is on sys.path so sibling modules can be imported
_here = os.path.dirname(__file__)
if _here not in sys.path:
    sys.path.insert(0, _here)

import config as cfg


def find_results_files():
    files = []
    for model in ("SVM", "RandomForest"):
        p = os.path.join(cfg.ROUNDS_DIR, f"grid_{model}_summary", "results.json")
        if os.path.exists(p):
            files.append((model, p))
    return files


def parse_name(name: str):
    # Example SVM: C-2_gamma-scale_cw-None_th-0.5_method-sigmoid_cf-3_
    # Example RF:  ne-100_md-8_ml-1_cw-None_th-0.5_
    parts = [p for p in name.strip("_").split("_") if p]
    out = {}
    for p in parts:
        if "-" not in p:
            continue
        k, v = p.split("-", 1)
        # Try to cast v to numeric if possible
        if v in ("None", "balanced"):
            out[k] = None if v == "None" else v
            continue
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def summarize(results: dict, model: str, top_n: int = 10, results_path: str | None = None):
    log_lines = []
    def pline(s: str):
        print(s)
        log_lines.append(s)
    items = []
    for name, metrics in results.items():
        params = parse_name(name)
        items.append((name, params, metrics))
    # Sort by F1 desc, break ties by AUC desc
    items.sort(key=lambda x: (x[2].get("f1", 0.0), x[2].get("auc", 0.0)), reverse=True)

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)

    pline(f"\n=== {model}: Top {min(top_n,len(items))} combinations by F1 ===")
    for i, (name, params, m) in enumerate(items[:top_n], 1):
        pline(f"{i:2d}. {name}")
        pline(f"    params: {params}")
        pline(f"    f1= {fmt(m.get('f1'))} acc= {fmt(m.get('accuracy'))} auc= {fmt(m.get('auc'))} ap= {fmt(m.get('average_precision'))} mcc= {fmt(m.get('mcc'))}")

    # Aggregations
    by_th = defaultdict(list)
    by_cw = defaultdict(list)
    by_method = defaultdict(list)
    by_cf = defaultdict(list)
    by_feats = defaultdict(list)
    # Initialize to avoid potential static-analysis warnings
    by_c = defaultdict(list); by_gamma = defaultdict(list)
    by_ne = defaultdict(list); by_md = defaultdict(list); by_ml = defaultdict(list)
    if model.lower() == "svm":
        by_c = defaultdict(list)
        by_gamma = defaultdict(list)
    else:
        by_ne = defaultdict(list)
        by_md = defaultdict(list)
        by_ml = defaultdict(list)

    for _, params, m in items:
        th = params.get("th") or params.get("threshold")
        if th is not None:
            by_th[th].append(m.get("f1", 0.0))
        cw = params.get("cw")
        by_cw[cw].append(m.get("f1", 0.0))
        feats = params.get("feats")
        if feats is not None:
            by_feats[feats].append(m.get("f1", 0.0))
        if model.lower() == "svm":
            by_c[params.get("C")].append(m.get("f1", 0.0))
            by_gamma[params.get("gamma")].append(m.get("f1", 0.0))
            meth = params.get("method")
            if meth is not None:
                by_method[meth].append(m.get("f1", 0.0))
            cf = params.get("cf") or params.get("cal_folds")
            if cf is not None:
                by_cf[cf].append(m.get("f1", 0.0))
        else:
            by_ne[params.get("ne")].append(m.get("f1", 0.0))
            by_md[params.get("md")].append(m.get("f1", 0.0))
            by_ml[params.get("ml")].append(m.get("f1", 0.0))

    import numpy as np

    def show_group(title, d):
        pline(f"\n{title}")
        for k in sorted(d, key=lambda x: (str(type(x)), x)):
            arr = np.array(d[k], dtype=float)
            pline(f" - {k}: mean_f1={arr.mean():.4f} (n={arr.size})")

    show_group("F1 by threshold", by_th)
    show_group("F1 by class_weight", by_cw)
    if model.lower() == "svm":
        show_group("F1 by C", by_c)
        show_group("F1 by gamma", by_gamma)
        if by_method:
            show_group("F1 by calibration method", by_method)
        if by_cf:
            show_group("F1 by calibration folds", by_cf)
        if by_feats:
            show_group("F1 by feature set", by_feats)
    else:
        show_group("F1 by n_estimators", by_ne)
        show_group("F1 by max_depth", by_md)
        show_group("F1 by min_samples_leaf", by_ml)

    # Feature importance aggregation (if present)
    try:
        if results_path:
            base_dir = os.path.dirname(results_path)
            imp_dir = os.path.join(base_dir, "feature_importance")
            if os.path.isdir(imp_dir):
                import glob, json as _json
                # Aggregate mean importance per feature across available combos
                feat_vals = {}
                feat_cnts = {}
                files = glob.glob(os.path.join(imp_dir, "*.json"))
                for fp in files:
                    try:
                        with open(fp, 'r') as f:
                            d = _json.load(f)
                        pairs = d.get("all") or d.get("top") or []
                        for name, val in pairs:
                            try:
                                v = float(val)
                            except Exception:
                                continue
                            feat_vals[name] = feat_vals.get(name, 0.0) + v
                            feat_cnts[name] = feat_cnts.get(name, 0) + 1
                    except Exception:
                        continue
                feat_mean = {k: (feat_vals[k] / max(1, feat_cnts[k])) for k in feat_vals}

                # Family grouping
                def family_of(feat_name: str) -> str:
                    n = feat_name
                    if n in ("ELEVATION", "SLOPE", "ASPECT_SIN", "ASPECT_COS"):
                        return "topography"
                    if any(n.startswith(b) for b in ("B2_", "B3_", "B4_", "B8_", "B11_", "B12_")):
                        return "spectral"
                    if any(n.startswith(p) for p in ("NDVI_s", "EVI_s", "EVI2_s", "NBR_s", "NDMI_s")):
                        return "indices_base"
                    if any(n.startswith(p) for p in ("NDWI_s", "BSI_s", "NDBI_s", "SAVI_s")):
                        return "indices_extra"
                    if ("_var3_s" in n) or ("_var5_s" in n):
                        return "textures_var"
                    if n.endswith("_sobel_s1") or n.endswith("_sobel_s2") or n.endswith("_sobel_s3") or ("_sobel_s" in n):
                        return "textures_sobel"
                    if n.startswith("NDVI_") or n.startswith("NDMI_"):
                        return "temporal"
                    return "other"

                fam_sums = {}
                for k, v in feat_mean.items():
                    fam = family_of(k)
                    fam_sums[fam] = fam_sums.get(fam, 0.0) + v

                # Print top features and families
                top_feats = sorted(feat_mean.items(), key=lambda x: x[1], reverse=True)[:20]
                pline("\nTop 20 features by mean permutation importance (F1):")
                for n, v in top_feats:
                    pline(f" - {n}: {v:.6f}")
                pline("\nFeature family contributions (sum of means):")
                for fam, val in sorted(fam_sums.items(), key=lambda x: x[1], reverse=True):
                    pline(f" - {fam}: {val:.6f}")

                # Append to copyable block (compact)
                top_line = "TopFeats=" + ",".join([f"{n}={v:.4f}" for n, v in top_feats[:10]])
                fam_line = "Families=" + ",".join([f"{f}={val:.4f}" for f, val in sorted(fam_sums.items(), key=lambda x: x[1], reverse=True)])
                pline(top_line)
                pline(fam_line)
    except Exception as e:
        pline(f"(Feature importance aggregation failed: {e})")

    # End of summary


def main():
    # Optional: user can pass an explicit results.json path
    if len(sys.argv) > 1:
        path = sys.argv[1]
        model = os.path.basename(os.path.dirname(path)).replace("grid_", "").replace("_summary", "")
        with open(path, "r") as f:
            results = json.load(f)
        summarize(results, model, results_path=path)
        return

    files = find_results_files()
    if not files:
        print("No grid search results found under", cfg.ROUNDS_DIR)
        print("Expected path: data/phase1/rounds/grid_<Model>_summary/results.json")
        sys.exit(1)

    for model, path in files:
        print(f"Reading {path}")
        with open(path, "r") as f:
            results = json.load(f)
        summarize(results, model, results_path=path)


if __name__ == "__main__":
    main()
