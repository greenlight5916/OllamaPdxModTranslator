import os, sys, json, re

from ollama_translator.engine import OllamaTranslator

_PREFIX_TO_LANG = {f"l_{v}": k for k, v in OllamaTranslator._LANG_CODE.items()}

PLATFORM_PATTERNS = [
    r'steamapps[\\/]common[\\/]([^\\/]+)',
    r'GOG Games[\\/]([^\\/]+)',
    r'GOG Galaxy[\\/]Games[\\/]([^\\/]+)',
]

_stopwords_cache = {}

def _app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))

def _glossary_dir():
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gd = os.path.join(base, "glossary")
    os.makedirs(gd, exist_ok=True)
    return gd

def load_stopwords(src_lang="en"):
    if src_lang in _stopwords_cache:
        return _stopwords_cache[src_lang]
    ext = os.path.join(_glossary_dir(), "glossary_stopwords.json")
    if os.path.isfile(ext):
        try:
            with open(ext, "r", encoding="utf-8") as f:
                data = json.load(f)
                s = set(data.get(src_lang, []))
                _stopwords_cache[src_lang] = s
                return s
        except Exception:
            pass
    try:
        if getattr(sys, 'frozen', False):
            path = os.path.join(sys._MEIPASS, "glossary", "glossary_stopwords.json")
        else:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base, "glossary", "glossary_stopwords.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            s = set(data.get(src_lang, []))
            _stopwords_cache[src_lang] = s
            return s
    except Exception:
        _stopwords_cache[src_lang] = set()
        return set()

def stem_en(word):
    w = word
    if len(w) > 5 and w.endswith('isation'):
        w = w[:-5]
    elif len(w) > 5 and w.endswith('ation'):
        w = w[:-3]
    elif len(w) > 4 and w.endswith('ment'):
        w = w[:-4]
    elif len(w) > 4 and w.endswith('ness'):
        w = w[:-4]
    elif len(w) > 4 and w.endswith('able'):
        w = w[:-4]
    elif len(w) > 4 and w.endswith('ible'):
        w = w[:-4]
    elif len(w) > 4 and w.endswith('ful'):
        w = w[:-3]
    elif len(w) > 4 and w.endswith('less'):
        w = w[:-4]
    elif len(w) > 3 and w.endswith('ing'):
        w = w[:-3]
    elif len(w) > 3 and w.endswith('ed'):
        w = w[:-2]
    elif len(w) > 3 and w.endswith('ly'):
        w = w[:-2]
    elif len(w) > 2 and w.endswith('s') and not w.endswith('ss'):
        w = w[:-1]
    return w

def tokenize_game_en(val, src_prefix="l_english"):
    tokens = []
    for m in re.finditer(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", val.lower()):
        t = m.group()
        for p in re.split(r'[_-]', t):
            p = p.strip("'")
            if len(p) >= 3 and p not in load_stopwords(src_prefix):
                p = stem_en(p)
                tokens.append(p)
    return tokens

def tokenize_tgt_pure(val):
    tokens = []
    for m in re.finditer(r'[\uAC00-\uD7AF]+', val):
        tokens.append(m.group())
    return tokens

def tokenize_en(val):
    return re.findall(r'[a-zA-Z]+', val.lower())

def tokenize_tgt(val, src_prefix="l_korean"):
    tokens = []
    for m in re.finditer(r'[\uAC00-\uD7AF]+', val):
        t = m.group()
        if t not in load_stopwords(src_prefix):
            tokens.append(t)
    return tokens

def parse_yml(val):
    data = {}
    for line in val.split("\n"):
        line = line.strip()
        if ":" in line:
            parts = line.split(":", 1)
            key = parts[0].strip().lower()
            raw = parts[1].strip()
            if raw.startswith('"') and raw.endswith('"'):
                raw = raw[1:-1]
            data[key] = raw
    return data
