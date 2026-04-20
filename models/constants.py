# models/constants.py

# ── Model identity ─────────────────────────────────────────────────────────
MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# ── LLM backbone: Vicuna-7B-v1.5 (Llama-2 architecture) ───────────────────
NUM_LAYERS  = 32      # decoder layers: model.language_model.model.layers[0..31]
NUM_HEADS   = 32      # attention heads per layer
D_MODEL     = 4096    # residual stream / hidden state dimension
D_HEAD      = 128     # D_MODEL // NUM_HEADS
VOCAB_SIZE  = 32000   # Vicuna tokenizer vocabulary size

# ── Visual encoder: CLIP ViT-L/14 at 336px ────────────────────────────────
NUM_VISUAL_TOKENS = 576   # 24×24 patch grid; always positions 1..576 in the sequence
CLIP_DIM          = 1024  # CLIP output dim (before MLP projector)
PROJ_DIM          = 4096  # after MLP projector; equals D_MODEL

# ── Sequence layout ────────────────────────────────────────────────────────
# Full input sequence to the LLM decoder:
#   pos 0            : BOS token
#   pos 1..576       : visual tokens (NUM_VISUAL_TOKENS = 576)
#   pos 577..577+S-1 : system + question tokens (length S varies per example)
#   pos 577+S        : generation starts here
# answer_position is computed dynamically as inputs["input_ids"].shape[1] - 1
VISUAL_START = 1
VISUAL_END   = 576   # inclusive

# ── Languages ──────────────────────────────────────────────────────────────
LANGUAGES  = ["fr", "ar", "zh", "bn"]
LANG_NAMES = {"fr": "French", "ar": "Arabic", "zh": "Chinese", "bn": "Bengali"}

# ── Layer range definitions for ablations ─────────────────────────────────
LAYER_RANGES = {
    "early": list(range(0,  11)),   # layers 0–10
    "mid":   list(range(11, 22)),   # layers 11–21
    "late":  list(range(22, 32)),   # layers 22–31
}

# ── Steering α values to test ──────────────────────────────────────────────
ALPHA_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0]

# ── 15 object categories for probing set ──────────────────────────────────
CATEGORIES = [
    "cat", "dog", "bird", "horse", "car",
    "bus", "chair", "bed", "apple", "banana",
    "cup", "book", "clock", "handbag", "potted plant",
]

# Translations: for each category, one word per language.
TRANSLATIONS = {
    "cat":    {"en": "cat",    "fr": "chat",    "ar": "قطة",    "zh": "猫",    "bn": "বিড়াল"},
    "dog":    {"en": "dog",    "fr": "chien",   "ar": "كلب",    "zh": "狗",    "bn": "কুকুর"},
    "bird":   {"en": "bird",   "fr": "oiseau",  "ar": "طائر",   "zh": "鸟",    "bn": "পাখি"},
    "horse":  {"en": "horse",  "fr": "cheval",  "ar": "حصان",   "zh": "马",    "bn": "ঘোড়া"},
    "car":    {"en": "car",    "fr": "voiture", "ar": "سيارة",  "zh": "汽车",  "bn": "গাড়ি"},
    "bus":    {"en": "bus",    "fr": "bus",     "ar": "حافلة",  "zh": "公共汽车","bn": "বাস"},
    "chair":  {"en": "chair",  "fr": "chaise",  "ar": "كرسي",   "zh": "椅子",  "bn": "চেয়ার"},
    "bed":    {"en": "bed",    "fr": "lit",     "ar": "سرير",   "zh": "床",    "bn": "বিছানা"},
    "apple":  {"en": "apple",  "fr": "pomme",   "ar": "تفاحة",  "zh": "苹果",  "bn": "আপেল"},
    "banana": {"en": "banana", "fr": "banane",  "ar": "موزة",   "zh": "香蕉",  "bn": "কলা"},
    "cup":    {"en": "cup",    "fr": "tasse",   "ar": "كوب",    "zh": "杯子",  "bn": "কাপ"},
    "book":   {"en": "book",   "fr": "livre",   "ar": "كتاب",   "zh": "书",    "bn": "বই"},
    "clock":  {"en": "clock",  "fr": "horloge", "ar": "ساعة",   "zh": "时钟",  "bn": "ঘড়ি"},
    "handbag":      {"en": "bag",    "fr": "sac",     "ar": "حقيبة",  "zh": "包",    "bn": "ব্যাগ"},
    "potted plant": {"en": "plant",  "fr": "plante",  "ar": "نبتة",   "zh": "植物",  "bn": "গাছ"},
}

# Prompt template per language
QUESTIONS = {
    "en": "What object is shown in this image?",
    "fr": "Quel objet est montré dans cette image ?",
    "ar": "ما الجسم الموضح في هذه الصورة؟",
    "zh": "这张图片中显示的是什么物体？",
    "bn": "এই ছবিতে কোন বস্তুটি দেখানো হয়েছে?",
}
