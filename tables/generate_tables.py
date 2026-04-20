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


def bold(val, is_best):
    s = f"{val:.3f}" if val is not None else "n/a"
    return f"\\textbf{{{s}}}" if is_best and val is not None else s


def find_best(vals):
    """Return index of best non-baseline value (index 0 is baseline)."""
    non_base = [(i, v) for i, v in enumerate(vals[1:], start=1) if v is not None]
    if not non_base:
        return -1
    return max(non_base, key=lambda x: x[1])[0]


# ─── Table 1: Main Results ────────────────────────────────────────────────────

print("% Table 1: Main Results")
print("\\begin{table}[t]")
print("\\centering")
print("\\caption{Main results. Accuracy on MMMB and xGQA benchmarks.}")
print("\\label{tab:main}")
print("\\begin{tabular}{lcccccc}")
print("\\toprule")
print("Method & EN-MMMB & ZH-MMMB & AR-MMMB & AR-xGQA & BN-xGQA & Avg \\\\")
print("\\midrule")

# Collect numbers
configs = [
    ("EN-MMMB",  "mmmb",  "en"),
    ("ZH-MMMB",  "mmmb",  "zh"),
    ("AR-MMMB",  "mmmb",  "ar"),
    ("AR-xGQA",  "xgqa",  "ar"),
    ("BN-xGQA",  "xgqa",  "bn"),
]

rows = {}
for label, ds, lang in configs:
    base_val  = load_result(ds, lang, "baseline")
    ours_val  = load_result(ds, lang, "steered_a1.0")
    rows[label] = {"baseline": base_val, "ours": ours_val}

def row_to_cols(method_key):
    vals = [rows[lbl].get(method_key) for lbl, _, _ in configs]
    avg = sum(v for v in vals if v is not None) / max(1, sum(1 for v in vals if v is not None))
    return vals, avg

base_vals, base_avg = row_to_cols("baseline")
ours_vals, ours_avg = row_to_cols("ours")

def fmt_row(name, vals, avg, is_ours=False):
    parts = []
    for i, v in enumerate(vals):
        is_best = is_ours and v is not None
        parts.append(bold(v, is_best))
    avg_str = bold(avg, is_ours)
    return f"{name} & " + " & ".join(parts) + f" & {avg_str} \\\\"

print(fmt_row("Baseline (LLaVA-1.5-7B)", base_vals, base_avg, is_ours=False))
print(fmt_row("Ours (\\alpha=1.0)", ours_vals, ours_avg, is_ours=True))

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
print()

# ─── Table 2: Layer Range Ablation ───────────────────────────────────────────

print("% Table 2: Layer Range Ablation")
print("\\begin{table}[t]")
print("\\centering")
print("\\caption{Layer range ablation on MMMB (\\alpha=1.0).}")
print("\\label{tab:layer_range}")
print("\\begin{tabular}{lcc}")
print("\\toprule")
print("Layer Range & ZH-MMMB & AR-MMMB \\\\")
print("\\midrule")

lr_path = "outputs/ablations/layer_range_ablation.json"
if os.path.exists(lr_path):
    lr = json.load(open(lr_path))
    range_labels = {"early": "Early (0--10)", "mid": "Mid (11--21)",
                    "late": "Late (22--31)", "ours": "Ours (mech.\\ identified)"}
    all_zh = [lr.get(k, {}).get("zh_mmmb") for k in ["early", "mid", "late", "ours"]]
    all_ar = [lr.get(k, {}).get("ar_mmmb") for k in ["early", "mid", "late", "ours"]]
    best_zh = max((v for v in all_zh if v is not None), default=None)
    best_ar = max((v for v in all_ar if v is not None), default=None)
    for i, (k, label) in enumerate(range_labels.items()):
        zh = lr.get(k, {}).get("zh_mmmb")
        ar = lr.get(k, {}).get("ar_mmmb")
        zh_s = bold(zh, zh == best_zh)
        ar_s = bold(ar, ar == best_ar)
        print(f"{label} & {zh_s} & {ar_s} \\\\")
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
print("\\caption{Effect of removing visual tokens (blank image) on steering.}")
print("\\label{tab:visual_ablation}")
print("\\begin{tabular}{lcccc}")
print("\\toprule")
print("Condition & ZH Base & ZH Steered & AR Base & AR Steered \\\\")
print("\\midrule")

va_path = "outputs/ablations/visual_token_ablation.json"
if os.path.exists(va_path):
    va = json.load(open(va_path))
    for cond_key, cond_label in [("with_image", "With image"), ("without_image", "Without image (blank)")]:
        cond_data = va.get(cond_key, {})
        zh_base    = cond_data.get("zh", {}).get("baseline")
        zh_steered = cond_data.get("zh", {}).get("steered")
        ar_base    = cond_data.get("ar", {}).get("baseline")
        ar_steered = cond_data.get("ar", {}).get("steered")
        def fs(v): return f"{v:.3f}" if v is not None else "n/a"
        print(f"{cond_label} & {fs(zh_base)} & {fs(zh_steered)} & {fs(ar_base)} & {fs(ar_steered)} \\\\")
else:
    print("\\multicolumn{5}{c}{Results not yet available} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
print()

# ─── Table 4: Cross-Lingual Transfer ─────────────────────────────────────────

print("% Table 4: Cross-Lingual Transfer")
print("\\begin{table}[t]")
print("\\centering")
print("\\caption{Cross-lingual transfer accuracy on MMMB (\\alpha=1.0). Rows = steering vector source, Cols = evaluation language.}")
print("\\label{tab:cross_lingual}")

cl_path = "outputs/ablations/cross_lingual_transfer.json"
if os.path.exists(cl_path):
    cl = json.load(open(cl_path))
    # Determine which languages appear
    src_langs = []
    tgt_langs = []
    for key in cl:
        parts = key.split("__")
        src = parts[0].replace("vec_from_", "")
        tgt = parts[1].replace("eval_", "")
        if src not in src_langs:
            src_langs.append(src)
        if tgt not in tgt_langs:
            tgt_langs.append(tgt)

    if src_langs and tgt_langs:
        print(f"\\begin{{tabular}}{{l{'c' * len(tgt_langs)}}}")
        print("\\toprule")
        header = "Source $\\backslash$ Target & " + " & ".join(tgt_langs) + " \\\\"
        print(header)
        print("\\midrule")
        for src in src_langs:
            row = [src]
            for tgt in tgt_langs:
                if src == tgt:
                    row.append("---")
                else:
                    key = f"vec_from_{src}__eval_{tgt}"
                    val = cl.get(key)
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
