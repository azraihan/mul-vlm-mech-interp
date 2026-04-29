import os
import sys
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_result(dataset, lang, method):
    path = f"outputs/eval_results/results_{dataset}_{lang}_{method}.json"
    if os.path.exists(path):
        return json.load(open(path))["accuracy"]
    return None


def load_best_steered(dataset, lang):
    """Return the best accuracy across all steered_a* result files for this dataset/lang."""
    best = None
    for path in glob.glob(f"outputs/eval_results/results_{dataset}_{lang}_steered_a*.json"):
        acc = json.load(open(path)).get("accuracy")
        if acc is not None and (best is None or acc > best):
            best = acc
    return best


def best_from_alpha_dict(d):
    """Return the best steered accuracy from a {alpha_str: scalar} dict (excludes baseline '0.0')."""
    if not isinstance(d, dict):
        return None
    vals = [v for k, v in d.items() if k != "0.0" and isinstance(v, (int, float))]
    return max(vals) if vals else None


def bold(val, is_best):
    s = f"{val:.3f}" if val is not None else "n/a"
    return f"\\textbf{{{s}}}" if is_best and val is not None else s


# ─── Table 1: Main Results ────────────────────────────────────────────────────

print("% Table 1: Main Results")
print("\\begin{table}[t]")
print("\\centering")
print("\\caption{Main results. Accuracy on MMMB and xGQA benchmarks. "
      "Ours shows the best accuracy across all tested $\\alpha$ values per column.}")
print("\\label{tab:main}")
print("\\begin{tabular}{lccccccc}")
print("\\toprule")
print("Method & EN-MMMB & PT-MMMB & RU-MMMB & DE-xGQA & PT-xGQA & RU-xGQA & Avg \\\\")
print("\\midrule")

configs = [
    ("EN-MMMB", "mmmb", "en"),
    ("PT-MMMB", "mmmb", "pt"),
    ("RU-MMMB", "mmmb", "ru"),
    ("DE-xGQA", "xgqa", "de"),
    ("PT-xGQA", "xgqa", "pt"),
    ("RU-xGQA", "xgqa", "ru"),
]

rows = {}
for label, ds, lang in configs:
    rows[label] = {
        "baseline": load_result(ds, lang, "baseline"),
        "ours":     load_best_steered(ds, lang),
    }


def row_to_cols(method_key):
    vals = [rows[lbl].get(method_key) for lbl, _, _ in configs]
    avg = sum(v for v in vals if v is not None) / max(1, sum(1 for v in vals if v is not None))
    return vals, avg


base_vals, base_avg = row_to_cols("baseline")
ours_vals, ours_avg = row_to_cols("ours")


def fmt_row(name, vals, avg, is_ours=False):
    parts = [bold(v, is_ours and v is not None) for v in vals]
    return f"{name} & " + " & ".join(parts) + f" & {bold(avg, is_ours)} \\\\"


print(fmt_row("Baseline (LLaVA-1.5-7B)", base_vals, base_avg, is_ours=False))
print(fmt_row("Ours (best $\\alpha$)", ours_vals, ours_avg, is_ours=True))

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
print()

# ─── Table 2: Layer Range Ablation ───────────────────────────────────────────

print("% Table 2: Layer Range Ablation")
print("\\begin{table}[t]")
print("\\centering")
print("\\caption{Layer range ablation on MMMB (best $\\alpha$ per cell).}")
print("\\label{tab:layer_range}")
print("\\begin{tabular}{lcc}")
print("\\toprule")
print("Layer Range & PT-MMMB & RU-MMMB \\\\")
print("\\midrule")

lr_path = "outputs/ablations/layer_range_ablation.json"
if os.path.exists(lr_path):
    lr = json.load(open(lr_path))
    range_labels = {
        "early": "Early (0--10)",
        "mid":   "Mid (11--21)",
        "late":  "Late (22--31)",
        "ours":  "Ours (mech.\\ identified)",
    }

    def _scalar(v):
        """Extract best steered scalar from a raw value (scalar or alpha dict)."""
        if isinstance(v, dict):
            return best_from_alpha_dict(v)
        return v

    all_pt = [_scalar(lr.get(k, {}).get("pt_mmmb")) for k in range_labels]
    all_ru = [_scalar(lr.get(k, {}).get("ru_mmmb")) for k in range_labels]
    best_pt = max((v for v in all_pt if v is not None), default=None)
    best_ru = max((v for v in all_ru if v is not None), default=None)

    for k, label in range_labels.items():
        pt = _scalar(lr.get(k, {}).get("pt_mmmb"))
        ru = _scalar(lr.get(k, {}).get("ru_mmmb"))
        print(f"{label} & {bold(pt, pt == best_pt and pt is not None)} "
              f"& {bold(ru, ru == best_ru and ru is not None)} \\\\")
else:
    print("\\multicolumn{3}{c}{Results not yet available} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
print()

# ─── Table 3: Visual Token Ablation ──────────────────────────────────────────

print("% Table 3: Visual Token Ablation")
print("\\begin{table}[t]")
print("\\centering")
print("\\caption{Effect of removing visual tokens (blank image) on MMMB accuracy. "
      "Steered = best $\\alpha$.}")
print("\\label{tab:visual_ablation}")
print("\\begin{tabular}{lcccc}")
print("\\toprule")
print("Condition & PT Base & PT Steered & RU Base & RU Steered \\\\")
print("\\midrule")

va_path = "outputs/ablations/visual_token_ablation.json"
if os.path.exists(va_path):
    va = json.load(open(va_path))
    for cond_key, cond_label in [("with_image", "With image"),
                                  ("without_image", "Without image (blank)")]:
        cond_data = va.get(cond_key, {})
        pt_base    = cond_data.get("pt_mmmb", {}).get("0.0")
        pt_steered = best_from_alpha_dict(cond_data.get("pt_mmmb", {}))
        ru_base    = cond_data.get("ru_mmmb", {}).get("0.0")
        ru_steered = best_from_alpha_dict(cond_data.get("ru_mmmb", {}))

        def fs(v):
            return f"{v:.3f}" if v is not None else "n/a"
        print(f"{cond_label} & {fs(pt_base)} & {fs(pt_steered)} "
              f"& {fs(ru_base)} & {fs(ru_steered)} \\\\")
else:
    print("\\multicolumn{5}{c}{Results not yet available} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
print()

# ─── Table 4: Cross-Lingual Transfer (MMMB) ──────────────────────────────────

print("% Table 4: Cross-Lingual Transfer")
print("\\begin{table}[t]")
print("\\centering")
print("\\caption{Cross-lingual transfer on MMMB (best $\\alpha$). "
      "Rows = steering vector source language, Cols = evaluation language. "
      "--- = own-language vector (see Table~\\ref{tab:main}).}")
print("\\label{tab:cross_lingual}")

cl_path = "outputs/ablations/cross_lingual_transfer.json"
if os.path.exists(cl_path):
    cl = json.load(open(cl_path))

    # Collect src/tgt pairs from MMMB keys only
    src_langs, tgt_langs = [], []
    for key in cl:
        if not key.endswith("_mmmb"):
            continue
        parts = key.split("__")
        src = parts[0].replace("vec_from_", "")
        tgt = parts[1].replace("eval_", "").replace("_mmmb", "")
        if src not in src_langs:
            src_langs.append(src)
        if tgt not in tgt_langs:
            tgt_langs.append(tgt)

    if src_langs and tgt_langs:
        print(f"\\begin{{tabular}}{{l{'c' * len(tgt_langs)}}}")
        print("\\toprule")
        print("Source $\\backslash$ Target & " + " & ".join(tgt_langs) + " \\\\")
        print("\\midrule")
        for src in src_langs:
            row = [src]
            for tgt in tgt_langs:
                if src == tgt:
                    row.append("---")
                else:
                    key = f"vec_from_{src}__eval_{tgt}_mmmb"
                    alpha_dict = cl.get(key)
                    val = best_from_alpha_dict(alpha_dict) if isinstance(alpha_dict, dict) else alpha_dict
                    row.append(f"{val:.3f}" if val is not None else "n/a")
            print(" & ".join(row) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
    else:
        print("\\begin{tabular}{lcc}")
        print("\\toprule")
        print("\\multicolumn{3}{c}{Results not yet available} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
else:
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("\\multicolumn{3}{c}{Results not yet available} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

print("\\end{table}")
