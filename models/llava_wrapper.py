import torch
import torch.nn.functional as F
import PIL.Image
import numpy as np
from transformers import LlavaForConditionalGeneration, AutoProcessor
from models.constants import MODEL_ID


def load_model(device="cuda"):
    """
    Returns: (model, processor)
    - model: LlavaForConditionalGeneration, float16, on `device`
    - processor: AutoProcessor for llava-hf/llava-1.5-7b-hf
    """
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=device
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def get_image_text_inputs(processor, image_pil, question_text, device="cuda"):
    """
    Formats a single (image, question) pair using the LLaVA-1.5 prompt template.
    Returns (inputs_dict, answer_position).
    answer_position = inputs["input_ids"].shape[1] - 1
    """
    prompt = f"USER: <image>\n{question_text}\nASSISTANT:"
    inputs = processor(
        text=prompt,
        images=image_pil,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    answer_position = inputs["input_ids"].shape[1] - 1
    return inputs, answer_position


def get_blank_image_inputs(processor, question_text, device="cuda"):
    """
    Same as get_image_text_inputs but uses a blank white 336×336 RGB PIL image.
    Keeps <image> in the prompt so visual token positions 1..576 are identical.
    """
    blank = PIL.Image.new("RGB", (336, 336), color=(255, 255, 255))
    return get_image_text_inputs(processor, blank, question_text, device)


def extract_residual_streams(model, inputs, layers):
    """
    Returns dict: layer_idx → numpy array (seq_len, 4096) float32.
    Captured from OUTPUT[0] of each LlamaDecoderLayer (residual stream after attn+MLP).
    """
    residuals = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, inp, output):
            residuals[layer_idx] = output[0][0].detach().cpu().float().numpy()
        return hook_fn

    try:
        for i in layers:
            h = model.language_model.model.layers[i].register_forward_hook(make_hook(i))
            hooks.append(h)
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    return residuals


def extract_submodule_streams(model, inputs, layers, submodule):
    """
    Captures submodule outputs for all specified layers in ONE forward pass.
    submodule: "attn" → self_attn  (output tuple; element 0 is (batch, seq_len, 4096))
               "mlp"  → mlp        (output is a plain tensor (batch, seq_len, 4096))
    Returns dict: layer_idx → numpy array (seq_len, 4096) float32.
    """
    cache = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, inp, output):
            tensor = output[0] if isinstance(output, tuple) else output
            cache[layer_idx] = tensor[0].detach().cpu().float().numpy()
        return hook_fn

    try:
        for i in layers:
            if submodule == "attn":
                sub = model.language_model.model.layers[i].self_attn
            elif submodule == "mlp":
                sub = model.language_model.model.layers[i].mlp
            else:
                raise ValueError(f"Unknown submodule: {submodule}")
            hooks.append(sub.register_forward_hook(make_hook(i)))
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    return cache


def apply_logit_lens(model, hidden_1d_numpy):
    """
    Args:
        hidden_1d_numpy: numpy array (4096,) — residual stream at ONE position
    Returns:
        probs: numpy array (32000,) — token probabilities
    """
    device = model.language_model.lm_head.weight.device
    h = torch.tensor(hidden_1d_numpy, dtype=torch.float16, device=device)
    h = h.unsqueeze(0).unsqueeze(0)                        # (1, 1, 4096)
    with torch.no_grad():
        normed = model.language_model.model.norm(h)         # (1, 1, 4096)
        logits = model.language_model.lm_head(normed)       # (1, 1, 32000)
    probs = torch.softmax(logits.squeeze(), dim=-1)         # (32000,)
    return probs.float().cpu().numpy()


def patch_and_run(model, inputs_clean, inputs_corrupted, patch_layer, patch_positions,
                  corrupted_cache=None):
    """
    Causal activation patching: run clean forward pass but replace residual stream
    at patch_layer for patch_positions with values from the corrupted forward pass.

    Returns probs (32000,) numpy at the last sequence position.
    """
    # Step 1: Get corrupted hidden state at patch_layer
    if corrupted_cache is not None:
        corrupted_h = corrupted_cache[patch_layer]   # numpy (seq_len, 4096)
    else:
        cache = extract_residual_streams(model, inputs_corrupted, [patch_layer])
        corrupted_h = cache[patch_layer]             # numpy (seq_len, 4096)

    # Step 2 & 3: Patch and run clean pass
    final_hidden = {}

    def patch_hook(module, inp, output):
        h = output[0].clone()   # (batch=1, seq_len, 4096)
        device = h.device
        patch_vals = torch.tensor(corrupted_h[patch_positions], dtype=h.dtype, device=device)
        h[0, patch_positions, :] = patch_vals
        return (h,) + output[1:]

    def capture_final_hook(module, inp, output):
        final_hidden["layer31"] = output[0][0].detach().cpu().float().numpy()

    hooks = []
    try:
        h1 = model.language_model.model.layers[patch_layer].register_forward_hook(patch_hook)
        h2 = model.language_model.model.layers[31].register_forward_hook(capture_final_hook)
        hooks = [h1, h2]
        with torch.no_grad():
            model(**inputs_clean)
    finally:
        for h in hooks:
            h.remove()

    # Step 4: Apply logit lens at the last position
    last_pos = inputs_clean["input_ids"].shape[1] - 1
    probs = apply_logit_lens(model, final_hidden["layer31"][last_pos])
    return probs


def patch_submodule_and_run(model, inputs_clean, inputs_corrupted, patch_layer, submodule,
                            patch_positions, corrupted_cache=None):
    """
    Like patch_and_run but patches only a submodule output.
    submodule: "attn" → self_attn, "mlp" → mlp
    corrupted_cache: optional dict {layer_idx: numpy array (seq_len, 4096)} pre-computed by
                     extract_submodule_streams(model, inputs_corrupted, layers, submodule).
                     If provided, skips the corrupted forward pass entirely.
    Returns probs (32000,) numpy at the last sequence position.

    NOTE: LlamaAttention returns a tuple (attn_out, weights, past_kv);
          LlamaMLP returns a plain tensor. Both cases are handled below.
    """
    if submodule == "attn":
        sub = model.language_model.model.layers[patch_layer].self_attn
    elif submodule == "mlp":
        sub = model.language_model.model.layers[patch_layer].mlp
    else:
        raise ValueError(f"Unknown submodule: {submodule}")

    # Step 1: get corrupted submodule output — shape (seq_len, 4096)
    if corrupted_cache is not None:
        corrupted_val = corrupted_cache[patch_layer]
    else:
        corrupted_sub_out = {}

        def capture_hook(module, inp, output):
            tensor = output[0] if isinstance(output, tuple) else output
            corrupted_sub_out["val"] = tensor[0].detach().cpu().float().numpy()

        h_cap = sub.register_forward_hook(capture_hook)
        try:
            with torch.no_grad():
                model(**inputs_corrupted)
        finally:
            h_cap.remove()
        corrupted_val = corrupted_sub_out["val"]

    # Step 2: patch clean run
    final_hidden = {}

    def patch_hook(module, inp, output):
        is_tuple = isinstance(output, tuple)
        tensor = output[0] if is_tuple else output          # (batch, seq_len, 4096)
        h = tensor.clone()
        device = h.device
        patch_vals = torch.tensor(corrupted_val[patch_positions],
                                  dtype=h.dtype, device=device)
        h[0, patch_positions, :] = patch_vals
        return (h,) + output[1:] if is_tuple else h

    def capture_final(module, inp, output):
        # layer 31 is always a LlamaDecoderLayer → tuple; output[0] is (batch, seq_len, 4096)
        final_hidden["val"] = output[0][0].detach().cpu().float().numpy()

    hooks = []
    try:
        hooks.append(sub.register_forward_hook(patch_hook))
        hooks.append(model.language_model.model.layers[31].register_forward_hook(capture_final))
        with torch.no_grad():
            model(**inputs_clean)
    finally:
        for h in hooks:
            h.remove()

    last_pos = inputs_clean["input_ids"].shape[1] - 1
    return apply_logit_lens(model, final_hidden["val"][last_pos])


def get_token_id(processor, word):
    """
    Returns the FIRST token ID for `word` in generation context, regardless of how many
    tokens the word spans. Uses contextual subtraction ("ASSISTANT: {word}" minus the
    "ASSISTANT:" prefix) to match actual generation-time tokenization.
    Returns None only if the word produces zero tokens (should never happen).
    """
    prefix = "ASSISTANT:"
    prefix_ids = processor.tokenizer.encode(prefix, add_special_tokens=False)
    full_ids   = processor.tokenizer.encode(f"{prefix} {word}", add_special_tokens=False)
    word_ids   = full_ids[len(prefix_ids):]
    ids = word_ids
    if len(ids) >= 1:
        return ids[0]
    return None


def compute_pmi_baseline(model):
    """
    Returns log P(token) for each vocabulary token using a zero hidden state as the
    model's prior. Shape: (VOCAB_SIZE,) float32 numpy array.
    Used to PMI-normalize logit lens scores: score = log P(token|h) - log P(token|0).
    """
    device = model.language_model.lm_head.weight.device
    h = torch.zeros(1, 1, 4096, dtype=torch.float16, device=device)
    with torch.no_grad():
        normed = model.language_model.model.norm(h)
        logits = model.language_model.lm_head(normed)
    log_probs = torch.log_softmax(logits.squeeze(), dim=-1)
    return log_probs.float().cpu().numpy()


def generate_with_steering(model, inputs, steering_vec, alpha, pivot_layers, max_new_tokens=5):
    """
    Generates up to max_new_tokens new tokens, steering the residual stream at each pivot layer.
    steering_vec: dict {layer_idx: tensor(4096,)}
    alpha: float — scaling factor
    Returns: token tensor of newly generated tokens.
    """
    def make_hook(L):
        def hook_fn(module, inp, output):
            h = output[0].clone()
            v = steering_vec[L].to(h.device).to(h.dtype)
            # Steer the last position only — works for both prefill (seq_len > 1)
            # and KV-cache decode steps (seq_len = 1, where answer_pos would be OOB).
            h[:, -1:, :] += alpha * v
            return (h,) + output[1:]
        return hook_fn

    hooks = []
    try:
        for L in pivot_layers:
            hooks.append(
                model.language_model.model.layers[L].register_forward_hook(make_hook(L))
            )
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        for h in hooks:
            h.remove()

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return new_tokens
