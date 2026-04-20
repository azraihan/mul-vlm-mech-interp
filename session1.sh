#!/bin/bash
# SESSION 1 — Run this in your first Kaggle notebook.
# Covers: data download, Exp 1 (Logit Lens), Exp 2 (Activation Patching).
# Expected runtime on H100: 7–11 hours.
#
# HOW TO USE IN KAGGLE:
#   1. Enable Internet Access in notebook settings.
#   2. Add your HF token: Notebook settings → Secrets → add HF_TOKEN.
#   3. In a notebook cell, run:
#        import os
#        from kaggle_secrets import UserSecretsClient
#        os.environ["HUGGING_FACE_HUB_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
#   4. In the next cell: !cd /kaggle/working/mul-vlm-mech-interp && bash session1.sh
#   5. When done: commit the notebook so outputs/ is saved as notebook output.
#      In Session 2, add this committed notebook as a Data source.

set -euo pipefail
cd "$(dirname "$0")"

echo "[1/4] Installing dependencies (skipping torch/torchvision — Kaggle provides them)..."
grep -v "^torch" setup/requirements.txt \
  | grep -v "^torchvision" \
  | pip install -q -r /dev/stdin

echo "[2/4] Downloading and preparing data..."
python data/build_probing_set.py
python data/download_xgqa.py
python data/download_mmmb.py

echo "[3/4] Experiment 1: Logit Lens (~2–3h on H100)..."
python experiments/exp1_logit_lens.py

echo "[4/4] Experiment 2: Activation Patching (~5–8h on H100)..."
python experiments/exp2_activation_patching.py

echo ""
echo "=== SESSION 1 COMPLETE ==="
echo "Verifying Session 1 outputs..."
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
]
missing = [f for f in required if not os.path.exists(f)]
if missing:
    print("MISSING FILES — something went wrong:")
    for f in missing:
        print(f"  {f}")
    sys.exit(1)
print("All Session 1 outputs present.")
print("Next step: commit this notebook so outputs/ is preserved for Session 2.")
EOF
