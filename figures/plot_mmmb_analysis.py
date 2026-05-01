import os
import sys
import json
import textwrap
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.constants import LANG_NAMES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

ALPHA = 0.2
ANALYSIS_DIR = "mmmb_analysis"
OUT_DIR = "outputs/figures/mmmb_analysis"

OPT_COLORS = {"A": "#2166ac", "B": "#d6604d", "C": "#4dac26", "D": "#762a83"}
OPT_STYLES = {"A": "-",       "B": "--",       "C": "-.",      "D": ":"}

pivot_layers = None  # set in either mode below


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_curves(ax, curves, options, gold, pred, title):
    layers = list(range(32))
    ax.axvspan(min(pivot_layers) - 0.5, max(pivot_layers) + 0.5,
               alpha=0.08, color="gray", label="Pivot (5–17)")
    available = [k for k in ["A", "B", "C", "D"] if k in options]
    for opt in available:
        lw = 2.5 if opt == gold else 1.4
        label = f"{opt}. {options[opt]}"
        if opt == gold:
            label += "  (correct)"
        ax.plot(layers, curves[opt],
                color=OPT_COLORS[opt], linestyle=OPT_STYLES[opt],
                linewidth=lw, label=label)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlim(0, 31)
    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel("Log PMI", fontsize=9)
    ax.tick_params(labelsize=8)
    marker = "(correct)" if pred == gold else "(wrong)"
    ax.set_title(f"{title}  —  predicted: {pred} {marker}", fontsize=9, pad=3)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.7,
              handlelength=1.8, labelspacing=0.3)
    if "C" in curves and "D" in curves:
        hl = list(range(13, 20))
        c_h = [curves["C"][l] for l in hl]
        d_h = [curves["D"][l] for l in hl]
        ax.fill_between(hl, c_h, d_h,
                        where=[c > d for c, d in zip(c_h, d_h)],
                        alpha=0.22, color=OPT_COLORS["C"], zorder=1, interpolate=True)
        ax.fill_between(hl, c_h, d_h,
                        where=[d >= c for c, d in zip(c_h, d_h)],
                        alpha=0.22, color=OPT_COLORS["D"], zorder=1, interpolate=True)
        winner = "C" if curves["C"][31] > curves["D"][31] else "D"
        winner_vals = [curves[winner][l] for l in hl]
        mid_x = hl[len(hl) // 2]                        # middle of fill region
        top_y = max(max(c_h), max(d_h))                  # top of fill region
        slope = curves[winner][hl[-1]] - curves[winner][hl[0]]
        trend_arrow = "↑" if slope >= 0 else "↓"
        ax.text(
            mid_x, top_y - 0.4,
            f"Log PMI({winner}) ",
            color=OPT_COLORS[winner],
            fontsize=8.5, fontweight="bold", ha="right", va="bottom",
        )
        ax.text(
            mid_x, top_y - 0.4,
            trend_arrow,
            color=OPT_COLORS[winner],
            fontsize=11, fontweight="bold", ha="left", va="bottom",
        )



def draw_case_figure(d, base_curves, steer_curves, out_png):
    """Render and save one MMMB case figure from pre-computed curve dicts."""
    lang       = d["lang"]
    lang_label = LANG_NAMES.get(lang, lang.upper())
    options    = d["options"]
    gold       = d["answer"]
    base_pred  = d["baseline_predicted"]
    steer_pred = d["steered_predicted"]
    alpha      = d.get("alpha", ALPHA)
    stem       = os.path.splitext(os.path.basename(d["image_path"]))[0]

    img = Image.open(d["image_path"]).convert("RGB")

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 2.5], wspace=0.3)

    left_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[0], height_ratios=[3, 2], hspace=0.15
    )
    right_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[1], hspace=0.5
    )

    ax_img = fig.add_subplot(left_gs[0])
    ax_img.imshow(np.array(img))
    ax_img.axis("off")

    ax_base = fig.add_subplot(right_gs[0])
    plot_curves(ax_base, base_curves, options, gold, base_pred,
                "Baseline (no steering)")

    ax_steer = fig.add_subplot(right_gs[1])
    plot_curves(ax_steer, steer_curves, options, gold, steer_pred,
                f"Steered  (α = {alpha})")

    ax_txt = fig.add_subplot(left_gs[1])
    ax_txt.axis("off")
    wrapped_q = textwrap.fill(d["question"], width=42)
    opt_lines = "\n".join(
        f"  {k}. {options[k]}" for k in ["A", "B", "C", "D"] if k in options
    )
    caption = f"{wrapped_q}\n\n{opt_lines}\n\nCorrect answer: {gold}"
    ax_txt.text(0.5, 0.5, caption, transform=ax_txt.transAxes,
                fontsize=8, va="center", ha="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                          edgecolor="#cccccc", linewidth=0.6))

    plt.savefig(out_png, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  Saved {out_png}")


# ── JSON replay mode ──────────────────────────────────────────────────────────
# Usage: python figures/plot_mmmb_analysis.py <path/to/{stem}_data.json>

if len(sys.argv) > 1:
    json_path = sys.argv[1]
    print(f"Regenerating figure from {json_path}")
    d = json.load(open(json_path, encoding="utf-8"))

    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(d["image_path"]):
        d["image_path"] = os.path.join(_project_root, d["image_path"])

    pivot_layers = d["pivot_layers"]

    base_curves  = {o: np.array(v) for o, v in d["curves"]["baseline"].items()}
    steer_curves = {o: np.array(v) for o, v in d["curves"]["steered"].items()}

    out_png = os.path.splitext(json_path)[0].removesuffix("_data") + ".png"
    draw_case_figure(d, base_curves, steer_curves, out_png)
    sys.exit(0)


# ── Normal mode: run model and compute curves ─────────────────────────────────

import torch
from models.llava_wrapper import (load_model, get_image_text_inputs,
                                   extract_residual_streams, apply_logit_lens,
                                   compute_pmi_baseline)

# ── Gather all cases ──────────────────────────────────────────────────────────

all_cases = []
for lang in ["pt", "ru"]:
    for subset in ["steered_wins", "baseline_wins"]:
        qfile = os.path.join(ANALYSIS_DIR, lang, subset, "questions.json")
        if not os.path.exists(qfile):
            continue
        qdir = os.path.dirname(qfile)
        for q in json.load(open(qfile, encoding="utf-8")):
            stem = os.path.splitext(os.path.basename(q["image"]))[0]
            out_subdir = os.path.join(OUT_DIR, lang, subset)
            all_cases.append({
                "lang":       lang,
                "subset":     subset,
                "question":   q["question"],
                "options":    q["options"],
                "answer":     q["answer"].upper(),
                "image_path": os.path.join(qdir, q["image"]),
                "stem":       stem,
                "out_subdir": out_subdir,
                "out_png":    os.path.join(out_subdir, f"{stem}.png"),
                "out_json":   os.path.join(out_subdir, f"{stem}_data.json"),
            })

for c in all_cases:
    os.makedirs(c["out_subdir"], exist_ok=True)

pending = [c for c in all_cases if not os.path.exists(c["out_png"])]
print(f"{len(pending)} / {len(all_cases)} figures to generate.")
if not pending:
    print("All done.")
    sys.exit(0)

# ── Load model ────────────────────────────────────────────────────────────────

model, processor = load_model()
baseline_log_probs = compute_pmi_baseline(model)
pivot_layers = json.load(open("outputs/pivot_layers.json"))["pivot_layers"]

# Option token IDs via contextual subtraction (same as get_token_id)
_prefix     = "ASSISTANT:"
_prefix_ids = processor.tokenizer.encode(_prefix, add_special_tokens=False)
OPTION_TIDS = {}
for opt in ["A", "B", "C", "D"]:
    full_ids = processor.tokenizer.encode(f"{_prefix} {opt}", add_special_tokens=False)
    word_ids = full_ids[len(_prefix_ids):]
    ids = [i for i in word_ids if i != 29871]   # strip LLaMA leading-space token
    OPTION_TIDS[opt] = ids[0] if ids else None
print(f"Option token IDs: {OPTION_TIDS}")

# Load steering vectors
steering_vecs = {}
for lang in ["pt", "ru"]:
    sv_path = f"outputs/steering_vectors/steering_vec_{lang}.pt"
    if os.path.exists(sv_path):
        steering_vecs[lang] = torch.load(sv_path, map_location="cuda")
        print(f"[SV] Loaded steering vector for {lang}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_prompt(question, options):
    lines = [question, "\nOptions:"]
    for k in ["A", "B", "C", "D"]:
        if k in options:
            lines.append(f"\n{k}. {options[k]}")
    lines.append("\nAnswer with only the correct option letter (A, B, C, or D).")
    return "".join(lines)


def compute_option_curves(inputs, answer_pos, sv=None):
    """
    Run one forward pass (optionally steered) and return PMI curves for each
    option token across all 32 layers.
    Returns dict {opt: np.array(32,)}.
    """
    hooks = []
    if sv is not None:
        def make_hook(L, vec, alpha):
            def fn(module, inp, output):
                t = output[0] if isinstance(output, tuple) else output
                v = vec[L].to(t.device).to(t.dtype)
                if t.dim() == 3:
                    t.data[:, -1:, :] += alpha * v
                else:
                    t.data[-1:, :] += alpha * v
            return fn
        for L in pivot_layers:
            if L in sv:
                hooks.append(
                    model.language_model.model.layers[L].register_forward_hook(
                        make_hook(L, sv, ALPHA)
                    )
                )
    try:
        residuals = extract_residual_streams(model, inputs, list(range(32)))
    finally:
        for h in hooks:
            h.remove()

    curves = {}
    for opt, tid in OPTION_TIDS.items():
        if tid is None:
            curves[opt] = np.zeros(32)
            continue
        curves[opt] = np.array([
            np.log(apply_logit_lens(model, residuals[l][answer_pos])[tid] + 1e-10)
            - baseline_log_probs[tid]
            for l in range(32)
        ])
    return curves


def predict_from_curves(curves, options):
    """Argmax PMI at layer 31 among the options that appear in this question."""
    available = [k for k in ["A", "B", "C", "D"] if k in options]
    return max(available, key=lambda o: curves[o][31])


# ── Main loop ─────────────────────────────────────────────────────────────────

for case in pending:
    if os.path.exists(case["out_png"]):
        print(f"[SKIP] {case['out_png']}")
        continue

    lang    = case["lang"]
    options = case["options"]
    gold    = case["answer"]
    print(f"\n[{lang}/{case['subset']}] {case['stem']}")

    img    = Image.open(case["image_path"]).convert("RGB")
    prompt = build_prompt(case["question"], options)
    inputs, answer_pos = get_image_text_inputs(processor, img, prompt)
    sv = steering_vecs.get(lang)

    base_curves  = compute_option_curves(inputs, answer_pos, sv=None)
    steer_curves = compute_option_curves(inputs, answer_pos, sv=sv)
    base_pred    = predict_from_curves(base_curves,  options)
    steer_pred   = predict_from_curves(steer_curves, options)

    print(f"  gold={gold}  baseline={base_pred}  steered={steer_pred}")

    # ── Save plot data ────────────────────────────────────────────────────
    plot_data = {
        "lang":              lang,
        "lang_name":         LANG_NAMES.get(lang, lang),
        "subset":            case["subset"],
        "image_path":        case["image_path"],
        "question":          case["question"],
        "options":           options,
        "answer":            gold,
        "option_token_ids":  OPTION_TIDS,
        "baseline_predicted":  base_pred,
        "steered_predicted":   steer_pred,
        "alpha":             ALPHA,
        "pivot_layers":      pivot_layers,
        "curves": {
            "baseline": {o: base_curves[o].tolist()  for o in "ABCD"},
            "steered":  {o: steer_curves[o].tolist() for o in "ABCD"},
        },
    }
    with open(case["out_json"], "w", encoding="utf-8") as f:
        json.dump(plot_data, f, ensure_ascii=False, indent=2)

    draw_case_figure(plot_data, base_curves, steer_curves, case["out_png"])

print(f"\nDONE. Figures saved to {OUT_DIR}/")