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

alpha_sensitivity = json.load(open("outputs/ablations/alpha_sensitivity.json"))

colors  = {"pt": "#2166ac", "de": "#d6604d", "ru": "#4dac26", "en": "#808080"}
markers = {"pt": "o",       "de": "s",       "ru": "^",       "en": "x"}
alphas_x = [0.0, 0.5, 1.0, 1.5, 2.0]
datasets = ["mmmb", "xgqa"]
titles   = ["Dataset: MMMB", "Dataset: xGQA"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, dataset, title in zip(axes, datasets, titles):
    dataset_data = alpha_sensitivity.get(dataset, {})
    for lang, acc_dict in dataset_data.items():
        y = [acc_dict.get(str(a), None) for a in alphas_x]
        # Filter out None values
        x_valid = [x for x, yv in zip(alphas_x, y) if yv is not None]
        y_valid = [yv for yv in y if yv is not None]
        if not y_valid:
            continue
        lang_label = LANG_NAMES.get(lang, lang)
        ax.plot(x_valid, y_valid,
                color=colors.get(lang, "#333333"),
                marker=markers.get(lang, "o"),
                linewidth=2, markersize=6,
                label=lang_label)
    ax.set_xlabel("Steering coefficient α")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.set_xticks(alphas_x)

fig.tight_layout()

plt.savefig("outputs/figures/fig4_alpha_curves.pdf", bbox_inches="tight")
plt.savefig("outputs/figures/fig4_alpha_curves.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved fig4_alpha_curves.pdf/png")
