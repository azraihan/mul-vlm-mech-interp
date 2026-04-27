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
LANGUAGES  = ['pt', 'de', 'ru']
LANG_NAMES = {"en": "English", "pt": "Portuguese", "de": "German", "ru": "Russian"}

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
    'person','car','bus','train','boat','orange',
    'book','clock','knife','spoon','umbrella','bird','cat','dog','apple'
]


# Translations: for each category, one word per language.
TRANSLATIONS = {
    'person':    {'en':'person',   'de':'Person',       'ru':'человек', 'pt':'pessoa'},
    'car':       {'en':'car',      'de':'Auto',         'ru':'машина',  'pt':'carro'},
    'bus':       {'en':'bus',      'de':'Bus',          'ru':'автобус', 'pt':'ônibus'},
    'train':     {'en':'train',    'de':'Zug',          'ru':'поезд',   'pt':'trem'},
    'boat':      {'en':'boat',     'de':'Boot',         'ru':'лодка',   'pt':'barco'},
    'orange':    {'en':'orange',   'de':'Orange',       'ru':'апельсин','pt':'laranja'},
    'book':      {'en':'book',     'de':'Buch',         'ru':'книга',   'pt':'livro'},
    'clock':     {'en':'clock',    'de':'Uhr',          'ru':'часы',    'pt':'relógio'},
    'knife':     {'en':'knife',    'de':'Messer',       'ru':'нож',     'pt':'faca'},
    'spoon':     {'en':'spoon',    'de':'Löffel',       'ru':'ложка',   'pt':'colher'},
    'umbrella':  {'en':'umbrella', 'de':'Regenschirm',  'ru':'зонт',    'pt':'guarda-chuva'},
    'bird':      {'en':'bird',     'de':'Vogel',        'ru':'птица',   'pt':'pássaro'},
    'cat':       {'en':'cat',      'de':'Katze',        'ru':'кошка',   'pt':'gato'},
    'dog':       {'en':'dog',      'de':'Hund',         'ru':'собака',  'pt':'cachorro'},
    'apple':     {'en':'apple',    'de':'Apfel',        'ru':'яблоко',  'pt':'maçã'},
}

# Prompt template per language
CATEGORY_OPTIONS = {
    'en': ", ".join(["person","car","bus","train","boat","orange","book","clock","knife","spoon","umbrella","bird","cat","dog","apple"]),
    'de': ", ".join(["Person","Auto","Bus","Zug","Boot","Orange","Buch","Uhr","Messer","Löffel","Regenschirm","Vogel","Katze","Hund","Apfel"]),
    'ru': ", ".join(["человек","машина","автобус","поезд","лодка","апельсин","книга","часы","нож","ложка","зонт","птица","кошка","собака","яблоко"]),
    'pt': ", ".join(["pessoa","carro","ônibus","trem","barco","laranja","livro","relógio","faca","colher","guarda-chuva","pássaro","gato","cachorro","maçã"]),
}

QUESTIONS = {
    'en': f"What's shown in the image? Answer with exactly one word from: {CATEGORY_OPTIONS['en']}.",
    'de': f"Was ist auf dem Bild zu sehen? Antworten Sie mit genau einem Wort aus: {CATEGORY_OPTIONS['de']}.",
    'ru': f"Что изображено на картинке? Ответьте ровно одним словом из: {CATEGORY_OPTIONS['ru']}.",
    'pt': f"O que está mostrado na imagem? Responda com exatamente uma palavra de: {CATEGORY_OPTIONS['pt']}.",
}