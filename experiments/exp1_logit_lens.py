import os
import json
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.llava_wrapper import load_model, get_image_text_inputs, get_blank_image_inputs
from models.llava_wrapper import extract_residual_streams, apply_logit_lens
from models.constants import LANGUAGES, NUM_LAYERS

os.makedirs("outputs/logit_lens", exist_ok=True)
model, processor = load_model()
examples = json.load(open("data/probing_set/probing_set.json"))
N = len(examples)

en_probs = np.full((2, 4, N, NUM_LAYERS), np.nan, dtype=np.float32)
tl_probs = np.full((2, 4, N, NUM_LAYERS), np.nan, dtype=np.float32)

for cond_idx, use_image in enumerate([True, False]):
    for lang_idx, lang in enumerate(LANGUAGES):
        for ex_idx, ex in enumerate(tqdm(examples, desc=f"cond={cond_idx} lang={lang}")):
            en_tid = ex["token_ids"].get("en")
            tl_tid = ex["token_ids"].get(lang)
            if en_tid is None or tl_tid is None:
                continue  # leaves NaN in the array; nanmean handles it
            img = Image.open(ex["image_path"]).convert("RGB")
            question = ex["questions"][lang]

            if use_image:
                inputs, answer_pos = get_image_text_inputs(processor, img, question)
            else:
                inputs, answer_pos = get_blank_image_inputs(processor, question)

            residuals = extract_residual_streams(model, inputs, layers=list(range(NUM_LAYERS)))

            for layer in range(NUM_LAYERS):
                h = residuals[layer][answer_pos]        # shape (4096,)
                probs = apply_logit_lens(model, h)      # shape (32000,)
                en_probs[cond_idx, lang_idx, ex_idx, layer] = probs[en_tid]
                tl_probs[cond_idx, lang_idx, ex_idx, layer] = probs[tl_tid]

np.save("outputs/logit_lens/logit_lens_en_prob.npy", en_probs)
np.save("outputs/logit_lens/logit_lens_tl_prob.npy", tl_probs)
np.save("outputs/logit_lens/logit_lens_mean_en.npy", np.nanmean(en_probs, axis=2))
np.save("outputs/logit_lens/logit_lens_mean_tl.npy", np.nanmean(tl_probs, axis=2))
json.dump(
    {"N": N, "languages": LANGUAGES, "conditions": ["image", "blank"]},
    open("outputs/logit_lens/exp1_metadata.json", "w")
)
print("DONE: Experiment 1")
