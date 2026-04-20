import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.constants import LANGUAGES, LANG_NAMES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})
os.makedirs("outputs/figures", exist_ok=True)

patching_residual = np.load("outputs/patching/patching_residual.npy")   # (4, 32)
patching_attn     = np.load("outputs/patching/patching_attn.npy")       # (4, 32)
patching_mlp      = np.load("outputs/patching/patching_mlp.npy")        # (4, 32)

arrays = [patching_residual, patching_attn, patching_mlp]
col_titles = ["Residual Stream", "Attention Output", "MLP Output"]

fig, axes = plt.subplots(4, 3, figsize=(16, 10))

# Colorbar axes: one per column
cbar_axes = [fig.add_axes([0.92 + 0.025 * c, 0.11, 0.01, 0.77]) for c in range(1)]

for col, (arr, col_title) in enumerate(zip(arrays, col_titles)):
    for lang_idx in range(4):
        ax = axes[lang_idx, col]
        data = arr[lang_idx, :].reshape(1, 32)

        # Only draw the colorbar once — on the last column, last row.
        draw_cbar = (col == 2 and lang_idx == 3)
        sns.heatmap(
            data, ax=ax, cmap="RdBu_r", center=0, vmin=-0.05, vmax=0.3,
            cbar=draw_cbar,
            cbar_ax=cbar_axes[0] if draw_cbar else None,
            xticklabels=[str(i) if i % 8 == 0 else "" for i in range(32)],
            yticklabels=[LANG_NAMES[LANGUAGES[lang_idx]]],
        )
        ax.set_xlabel("Layer" if lang_idx == 3 else "")
        if lang_idx == 0:
            ax.set_title(col_title)

fig.tight_layout(rect=[0, 0, 0.91, 1])

plt.savefig("outputs/figures/fig2_patching_heatmap.pdf", bbox_inches="tight")
plt.savefig("outputs/figures/fig2_patching_heatmap.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved fig2_patching_heatmap.pdf/png")
