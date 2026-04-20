import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.constants import LANGUAGES, LANG_NAMES

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

residual = np.load("outputs/patching/patching_residual.npy")   # (4, 32)
visual   = np.load("outputs/patching/patching_visual.npy")     # (4, 32)

fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
layers = list(range(32))

for lang_idx in range(4):
    ax = axes[lang_idx]
    ax.plot(layers, residual[lang_idx], color="#2166ac", linewidth=2, label="Full patch")
    ax.plot(layers, visual[lang_idx],   color="#4dac26", linewidth=2, linestyle="--",
            label="Visual tokens only")
    ax.fill_between(layers, visual[lang_idx], residual[lang_idx],
                    alpha=0.15, color="#7b3294", label="Text token contribution")
    ax.set_xlabel("Layer")
    if lang_idx == 0:
        ax.set_ylabel("AIE")
    ax.set_title(LANG_NAMES[LANGUAGES[lang_idx]])
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()

plt.savefig("outputs/figures/fig3_visual_contribution.pdf", bbox_inches="tight")
plt.savefig("outputs/figures/fig3_visual_contribution.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved fig3_visual_contribution.pdf/png")
