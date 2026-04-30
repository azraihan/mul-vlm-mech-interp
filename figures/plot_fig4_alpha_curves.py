import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.constants import LANG_NAMES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})
os.makedirs("outputs/figures", exist_ok=True)

alpha_sensitivity = json.load(open("../outputs/ablations/alpha_sensitivity.json"))

colors = {"pt": "#648FFF", "de": "#FE6100", "ru": "#DC267F", "en": "#FFB000"}
markers = {"pt": "o",       "de": "s",       "ru": "^",       "en": "x"}
# Drop α=0.0 (baseline, already in Table 1) so log scale is clean
alphas_x = [0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5]
fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

# x positions for gap arrows (one per lang to avoid overlap)
arrow_x  = {"pt": 1.15, "ru": 1.10, "de": 0.28, "en": None}
LABEL_X  = 1.42   # fixed right-margin x for all +Xpp labels

dataset_data = alpha_sensitivity.get("mmmb", {})
peak_added     = False
baseline_added = False

for lang, acc_dict in dataset_data.items():
    y = [acc_dict.get(str(a), None) for a in alphas_x]
    x_valid = [x for x, yv in zip(alphas_x, y) if yv is not None]
    y_valid = [yv for yv in y if yv is not None]
    if not y_valid:
        continue
    color = colors.get(lang, "#333333")
    lang_label = LANG_NAMES.get(lang, lang)

    ax.plot(x_valid, y_valid,
            color=color, marker=markers.get(lang, "o"),
            linewidth=2, markersize=6, label=lang_label)

    peak_val = max(y_valid)
    ax.axhline(peak_val, color=color, linewidth=1.5, linestyle="--", alpha=0.85,
               label="Peak" if not peak_added else "_nolegend_")
    peak_added = True

    baseline_val = acc_dict.get("0.0")
    if baseline_val is None:
        continue
    ax.axhline(baseline_val, color=color, linewidth=1.5, linestyle=":", alpha=0.85,
               label="Baseline" if not baseline_added else "_nolegend_")
    baseline_added = True

    x_arr = arrow_x.get(lang)
    if x_arr is None:
        continue

    gap = peak_val - baseline_val
    ap = dict(color=color, lw=1.5, mutation_scale=12)
    mid = (peak_val + baseline_val) / 2

    if gap >= 0.02:
        ax.annotate("", xy=(x_arr, peak_val), xytext=(x_arr, baseline_val),
                    arrowprops=dict(arrowstyle="<->", **ap))
    else:
        tick_w = 0.015
        for y_tick in [peak_val, baseline_val]:
            ax.plot([x_arr - tick_w, x_arr + tick_w], [y_tick, y_tick],
                    color=color, lw=1.5, solid_capstyle="round")
        ax.plot([x_arr, x_arr], [baseline_val, peak_val], color=color, lw=1.5)

    ax.text(LABEL_X, mid,
            f"+{gap*100:.1f}pp", color=color, fontsize=9,
            va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))

ax.set_xscale("log")
ax.set_xlim(left=0.01, right=1.8)
ax.set_xticks(alphas_x)
ax.set_xticklabels([str(a) for a in alphas_x], rotation=45, ha="right")
ax.set_xlabel("Steering coefficient α (log scale)", labelpad=10)
ax.set_ylabel("Accuracy", labelpad=10)
ax.legend(fontsize=9)

fig.tight_layout()

plt.savefig("../outputs/figures/fig4_alpha_curves.pdf", bbox_inches="tight")
plt.savefig("../outputs/figures/fig4_alpha_curves.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved fig4_alpha_curves.pdf/png")
