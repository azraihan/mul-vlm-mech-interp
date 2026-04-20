#!/bin/bash
# SESSION 2 — Run this in your second Kaggle notebook after Session 1 is committed.
# Covers: Exp 3 (Steering), Exp 4 (Ablations), all figures, tables, final verification.
# Expected runtime on H100: 5–8 hours.
#
# HOW TO USE IN KAGGLE:
#   1. Create a new notebook (or re-open the same one with a fresh session).
#   2. Add your Session 1 committed notebook as a Data source
#      (Add data → Your Work → select the committed Session 1 notebook).
#      Its outputs will appear at /kaggle/input/<notebook-slug>/
#   3. Enable Internet Access. Add HF_TOKEN secret (same as Session 1).
#   4. In a notebook cell, run:
#        import os, shutil
#        from kaggle_secrets import UserSecretsClient
#        os.environ["HUGGING_FACE_HUB_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
#        # Copy Session 1 outputs into the working repo
#        src = "/kaggle/input/<your-session1-notebook-slug>"   # ← update this slug
#        dst = "/kaggle/working/mul-vlm-mech-interp"
#        for folder in ["outputs", "data"]:
#            s = os.path.join(src, folder)
#            d = os.path.join(dst, folder)
#            if os.path.exists(s):
#                shutil.copytree(s, d, dirs_exist_ok=True)
#        print("Copied Session 1 outputs.")
#   5. In the next cell: !cd /kaggle/working/mul-vlm-mech-interp && bash session2.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "[0/5] Verifying Session 1 outputs are present..."
python - <<'EOF'
import os, sys
required = [
    "outputs/logit_lens/logit_lens_mean_en.npy",
    "outputs/logit_lens/logit_lens_mean_tl.npy",
    "outputs/logit_lens/logit_lens_en_prob.npy",
    "outputs/logit_lens/logit_lens_tl_prob.npy",
    "outputs/logit_lens/exp1_metadata.json",
    "outputs/patching/patching_residual.npy",
    "outputs/patching/patching_attn.npy",
    "outputs/patching/patching_mlp.npy",
    "outputs/patching/patching_visual.npy",
    "outputs/pivot_layers.json",
    "data/probing_set/probing_set.json",
]
missing = [f for f in required if not os.path.exists(f)]
if missing:
    print("MISSING — copy Session 1 outputs first (see instructions at top of session2.sh):")
    for f in missing:
        print(f"  {f}")
    sys.exit(1)
print("Session 1 outputs verified. Proceeding.")
EOF

echo "[1/5] Installing dependencies (only packages not already in Kaggle)..."
# Only install what Kaggle doesn't provide — do NOT touch transformers, numpy,
# protobuf, huggingface_hub etc. as downgrading them breaks the environment.
pip install -q pycocotools einops sentencepiece

echo "[1b/5] Re-downloading MMMB (source moved to AIDC-AI/Parrot-dataset)..."
rm -rf data/mmmb/mmmb_*.json data/mmmb/images/
python data/download_mmmb.py

echo "[1c/5] Re-downloading xGQA with images (source: floschne/xgqa)..."
rm -rf data/xgqa/xgqa_*.json data/gqa_images/
python data/download_xgqa.py || true  # HF streaming GIL crash on exit is harmless
python - <<'PYEOF'
import json, os, sys
for lang in ["en", "zh", "bn"]:
    p = f"data/xgqa/xgqa_{lang}.json"
    n = len(json.load(open(p))) if os.path.exists(p) else 0
    print(f"  xgqa_{lang}: {n} examples")
    if n == 0:
        print(f"  ERROR: xgqa_{lang} is empty — download failed"); sys.exit(1)
cached = len([f for f in os.listdir("data/gqa_images") if f.endswith(".jpg")])
print(f"  gqa_images cache: {cached} images")
if cached == 0:
    print("  ERROR: no images cached"); sys.exit(1)
print("  xGQA data verified.")
PYEOF

echo "[2/5] Experiment 3: Steering Vectors + Evaluation (~3–4h on H100)..."
python experiments/exp3_steering.py

echo "[3/5] Experiment 4: Ablations (~2–3h on H100)..."
python experiments/exp4_ablations.py

echo "[4/5] Generating figures..."
python figures/plot_fig1_logit_lens.py
python figures/plot_fig2_patching_heatmap.py
python figures/plot_fig3_visual_contribution.py
python figures/plot_fig4_alpha_curves.py
python figures/plot_fig5_qualitative.py

echo "[5/5] Generating tables..."
python tables/generate_tables.py > tables/tables.tex

echo ""
echo "=== SESSION 2 COMPLETE — Final verification ==="
python - <<'EOF'
import os, sys
required = [
    "outputs/logit_lens/logit_lens_mean_en.npy",
    "outputs/logit_lens/logit_lens_mean_tl.npy",
    "outputs/patching/patching_residual.npy",
    "outputs/patching/patching_attn.npy",
    "outputs/patching/patching_mlp.npy",
    "outputs/patching/patching_visual.npy",
    "outputs/pivot_layers.json",
    "outputs/steering_vectors/steering_vec_fr.pt",
    "outputs/steering_vectors/steering_vec_ar.pt",
    "outputs/steering_vectors/steering_vec_zh.pt",
    "outputs/steering_vectors/steering_vec_bn.pt",
    "outputs/ablations/layer_range_ablation.json",
    "outputs/ablations/visual_token_ablation.json",
    "outputs/ablations/alpha_sensitivity.json",
    "outputs/ablations/cross_lingual_transfer.json",
    "outputs/figures/fig1_logit_lens.pdf",
    "outputs/figures/fig2_patching_heatmap.pdf",
    "outputs/figures/fig3_visual_contribution.pdf",
    "outputs/figures/fig4_alpha_curves.pdf",
    "outputs/figures/fig5_qualitative.pdf",
    "tables/tables.tex",
]
missing = [f for f in required if not os.path.exists(f)]
if missing:
    print("MISSING FILES:")
    for f in missing:
        print(f"  {f}")
    sys.exit(1)
print("ALL REQUIRED OUTPUTS PRESENT.")
print("Pipeline complete. Commit this notebook to save all outputs.")
EOF
