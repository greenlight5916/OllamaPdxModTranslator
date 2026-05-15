import os, sys, json, time, re, threading, subprocess, concurrent.futures
import requests, customtkinter as ctk
from tkinter import filedialog, messagebox

def _app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(_app_dir(), "ollama_translator_config.json")
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

# ── Game-specific prompt templates ──

GAME_PROMPTS = {
    "Crusader Kings 3": """You are translating text from the medieval grand strategy game 'Crusader Kings 3'.
Use a majestic, epic tone appropriate for medieval nobility and court intrigue.""",
    "Hearts of Iron 4": """You are translating text from the WWII grand strategy game 'Hearts of Iron 4'.
Use a concise, military report style with professional terminology.""",
    "Stellaris": """You are translating text from the sci-fi grand strategy game 'Stellaris'.
Use futuristic, scientific terminology and a tone suitable for space exploration and diplomacy.""",
    "Europa Universalis IV": """You are translating text from the historical grand strategy game 'Europa Universalis IV' (1444-1821 period).
Use formal diplomatic language appropriate for the Early Modern period.""",
    "Victoria 3": """You are translating text from the industrial era grand strategy game 'Victoria 3' (19th century).
Use terminology appropriate for the Industrial Revolution era, including political movements, economic systems, and social reforms.""",
    "Imperator: Rome": """You are translating text from the ancient grand strategy game 'Imperator: Rome'.
Use classical, dignified language appropriate for the Roman Republic period."""
}

def get_enhanced_prompt(game_name, base_prompt):
    if game_name in GAME_PROMPTS:
        return f"""[GAME CONTEXT]\n{GAME_PROMPTS[game_name]}\n\n[GENERAL INSTRUCTIONS]\n{base_prompt}"""
    return base_prompt

# ── Translation Engine ──

class OllamaTranslator:

    def __init__(self, log_callback, progress_callback, status_callback, stop_event, live_callback=None):
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        self.stop_event = stop_event
        self.live_callback = live_callback
        self.thread = None
        self.base_url = "http://localhost:11434"
        self.prompt_template = None
        self.max_retries = 3
        self.checkpoint_enabled = True
        self.debug_mode = False
        self._consecutive_errors = 0
        self.busy = False
        self._ollama_process = None

    def set_base_url(self, url):
        self.base_url = url.rstrip("/")

    def start_server(self):
        try:
            self._ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            self.log_callback("[OLLAMA] Starting Ollama server...")
            for i in range(30):
                time.sleep(1)
                try:
                    resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
                    if resp.status_code == 200:
                        models = [m["name"] for m in resp.json().get("models", [])]
                        self.log_callback(f"[OLLAMA] Server ready ({len(models)} model(s) found)")
                        return models
                except requests.exceptions.ConnectionError:
                    continue
            self.log_callback("[OLLAMA] Server start timed out after 30s")
            return None
        except Exception as e:
            self.log_callback(f"[OLLAMA] Failed to start server: {e}")
            return None

    def kill_server(self):
        if self._ollama_process:
            try:
                self._ollama_process.terminate()
                self._ollama_process.wait(timeout=5)
                self.log_callback("[OLLAMA] Server stopped")
            except Exception:
                try:
                    self._ollama_process.kill()
                except Exception:
                    pass
            self._ollama_process = None

    def fetch_models(self):
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=30)
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            pass
        return None

    def get_running_models(self):
        try:
            resp = requests.get(f"{self.base_url}/api/ps", timeout=10)
            if resp.status_code == 200:
                return resp.json().get("models", [])
        except Exception:
            pass
        return None

    def test_model(self, model, target_lang, game="None"):
        raw = self.prompt_template or (
            "Translate from English to {target_lang}.\nRules:\n"
            "1. Only translate after ': ' in quotes.\n"
            "2. Keep $vars$, [brackets], sectX intact.\n"
            "3. Output exactly one line.\n\n{batch_text}")
        test_text = 'key: "Hello World"'
        bp = raw.replace("{source_lang}", "English").replace("{target_lang}", target_lang).replace("{batch_text}", test_text)
        prompt = get_enhanced_prompt(game, bp)
        return self._call_ollama(model, prompt, temperature=0.1, max_tokens=512)

    def _check_fatal(self):
        self._consecutive_errors += 1
        if self._consecutive_errors >= self.max_retries + 2:
            self.log_callback("[FATAL] Consecutive LLM failures - stopping translation")
            self.stop_event.set()
            return True
        return False

    def _call_ollama(self, model, prompt, temperature=0.5, max_tokens=4096):
        try:
            resp = requests.post(f"{self.base_url}/api/chat", json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "stream": False
            }, timeout=120)
            resp.raise_for_status()
            self._consecutive_errors = 0
            return resp.json().get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            self._check_fatal()
            return "[OLLAMA_CONNECTION_ERROR]"
        except requests.exceptions.Timeout:
            self.log_callback("[TIMEOUT] LLM request timed out")
            if self._check_fatal():
                return "[OLLAMA_FATAL]"
            return "[OLLAMA_TIMEOUT]"
        except Exception as e:
            self._check_fatal()
            return f"[OLLAMA_ERROR: {e}]"

    def _save_checkpoint(self, result_lines, output_path):
        if not self.checkpoint_enabled:
            return
        cp_dir = os.path.join(os.path.dirname(CONFIG_FILE), "checkpoint")
        os.makedirs(cp_dir, exist_ok=True)
        cp_path = os.path.join(cp_dir, os.path.basename(output_path))
        try:
            with open(cp_path, "w", encoding="utf-8-sig") as f:
                f.writelines(result_lines)
            self.log_callback(f"[CHECKPOINT] Saved: {cp_path}")
        except Exception as e:
            self.log_callback(f"[CHECKPOINT] Failed: {e}")

    _LANG_CODE = {
        "English":"english","Korean":"korean","Simplified Chinese":"simp_chinese",
        "French":"french","German":"german","Spanish":"spanish",
        "Japanese":"japanese","Russian":"russian","Polish":"polish",
        "Brazilian Portuguese":"braz_por",
    }

    # ── Core batch translation ──

    def _translate_batch(self, lines, source_lang, target_lang, model, temperature, max_tokens, game="None", retry_count=0):
        header_pat = re.compile(r'^l_[a-z_]+:\s*$')
        comment_indices = {i for i, l in enumerate(lines) if not l.strip() or l.strip().startswith('#') or header_pat.match(l)}
        if comment_indices:
            if len(comment_indices) == len(lines):
                return lines
            actual_lines = [l for i, l in enumerate(lines) if i not in comment_indices]
            if self.debug_mode: self.log_callback(f"[DEBUG] batch:{len(lines)}lines filter->{len(actual_lines)}content")
            translated_actual = self._translate_batch(actual_lines, source_lang, target_lang, model, temperature, max_tokens, game, 0)
            if self.stop_event.is_set():
                return lines
            merged = []
            ai = 0
            tgt_code = self._LANG_CODE.get(target_lang, target_lang.lower())
            for i in range(len(lines)):
                if i in comment_indices:
                    l = lines[i]
                    if header_pat.match(l):
                        l = f"l_{tgt_code}:"
                    merged.append(l)
                else:
                    merged.append(translated_actual[ai])
                    ai += 1
            return merged

        ph_counter = 0
        send_data = []
        all_info = []

        for line in lines:
            stripped = line.strip()
            m = re.match(r'^([\w.]+):\s*', stripped)
            if not m:
                all_info.append({"type": "passthrough", "line": line})
                continue
            key = m.group(1)
            ws = line[:len(line) - len(line.lstrip())]
            value_part = stripped[len(m.group(0)):]
            raw_val = value_part
            if raw_val.startswith('"') and raw_val.endswith('"'):
                raw_val = raw_val[1:-1]
            line_phs = []
            def _ph_replacer(text):
                nonlocal ph_counter
                ph = f"{{PH{ph_counter}}}"
                ph_counter += 1
                line_phs.append((ph, text))
                return ph
            cleaned = re.sub(r'\$[^$]+\$', lambda m: _ph_replacer(m.group(0)), raw_val)
            cleaned = re.sub(r'\[[^\]]*\]', lambda m: _ph_replacer(m.group(0)), cleaned)
            cleaned = re.sub(r'\u00a7.', lambda m: _ph_replacer(m.group(0)), cleaned)
            cleaned = re.sub(r'\u00a3[^\u00a3]+\u00a3', lambda m: _ph_replacer(m.group(0)), cleaned)
            only_phs = not cleaned.strip() or re.match(r'^(\{PH\d+\}\s*)+\s*$', cleaned.strip())
            if only_phs:
                all_info.append({"type": "keep", "key": key, "ws": ws, "val": value_part, "phs": line_phs})
            else:
                idx = len(send_data)
                all_info.append({"type": "send", "midx": idx, "key": key, "ws": ws})
                send_data.append((f"\u27e8{idx}\u27e9 {cleaned}", line_phs, key, ws))

        if not send_data:
            return lines

        batch_text = "\n".join(s[0] for s in send_data)
        if self.debug_mode: self.log_callback(f"[DEBUG] sending {len(send_data)} lines to LLM: {repr(batch_text[:200])}")
        if self.prompt_template:
            base_prompt = self.prompt_template.replace("{source_lang}", source_lang)
            base_prompt = base_prompt.replace("{target_lang}", target_lang)
            base_prompt = base_prompt.replace("{batch_text}", batch_text)
        else:
            base_prompt = (
                f"Translate the following text from '{source_lang}' to '{target_lang}'.\n"
                f"Rules:\n1. Preserve all {{PH0}}, {{PH1}}, etc. placeholders exactly as-is.\n"
                f"2. Preserve line markers like \u27e80\u27e9 \u27e81\u27e9 exactly as-is.\n"
                f"3. Do NOT wrap in code blocks or add explanations.\n\n{batch_text}")

        glossary = self._get_glossary_text(target_lang, game)
        if glossary:
            cnt = glossary.count("\n") - 4
            self.log_callback(f"[GLOSSARY] Applied {game} glossary ({max(0,cnt)} terms)")
        prompt = get_enhanced_prompt(game, base_prompt + glossary)
        result = self._call_ollama(model, prompt, temperature, max_tokens)

        if result.startswith("[OLLAMA_"):
            if len(lines) > 1:
                self.log_callback(f"[SPLIT] {result} - splitting batch of {len(lines)} lines")
                mid = len(lines) // 2
                t = min(temperature + 0.05, 1.0)
                first = self._translate_batch(lines[:mid], source_lang, target_lang, model, t, max_tokens, game)
                if self.stop_event.is_set():
                    return lines
                second = self._translate_batch(lines[mid:], source_lang, target_lang, model, t, max_tokens, game)
                return first + second
            else:
                if retry_count < self.max_retries:
                    self.log_callback(f"[RETRY {retry_count+1}/{self.max_retries}] {result} - retrying single line")
                    t = min(temperature + 0.1, 1.0)
                    return self._translate_batch(lines, source_lang, target_lang, model, t, max_tokens, game, retry_count + 1)
                self.log_callback(f"[FAIL] {result} - returning original after {self.max_retries} retries")
                if self.live_callback:
                    self.live_callback(lines, lines)
                return lines

        result = result.strip()
        result = re.sub(r"```(?:yaml|yml)?\s*\n?", "", result, flags=re.IGNORECASE)
        result = re.sub(r"\n?```", "", result)
        if self.debug_mode: self.log_callback(f"[DEBUG] raw response ({len(result)} chars): {repr(result[:300])}")

        translated_values = re.split(r'\u27e8\d+\u27e9\s*', result)
        if translated_values and translated_values[0].strip() == '':
            translated_values = translated_values[1:]
        if self.debug_mode: self.log_callback(f"[DEBUG] marker split -> {len(translated_values)} values")

        if len(translated_values) != len(send_data):
            self.log_callback(f"[WARN] LLM returned {len(translated_values)} markers (expected {len(send_data)}), keeping original")
            for i, tv in enumerate(translated_values):
                self.log_callback(f"[WARN]   out[{i}]: {repr(tv[:100])}")
            return lines

        reconstructed = list(lines)
        for i, info in enumerate(all_info):
            if info["type"] == "passthrough":
                continue
            elif info["type"] == "keep":
                key, ws, val, phs = info["key"], info["ws"], info["val"], info["phs"]
                restored = val
                if restored.startswith('"') and restored.endswith('"'):
                    restored = restored[1:-1]
                for ph, orig in phs:
                    restored = restored.replace(ph, orig)
                restored = restored.replace('\n', '\\n')
                q = '"' if val.startswith('"') else ''
                reconstructed[i] = f'{ws}{key}: {q}{restored}{q}\n'
            else:
                _, phs, send_key, send_ws = send_data[info["midx"]]
                raw = translated_values[info["midx"]].strip()
                if raw and raw != '""':
                    val = raw.strip('"')
                    for ph, orig in phs:
                        val = val.replace(ph, orig)
                    val = val.replace('\n', '\\n')
                    reconstructed[i] = f'{send_ws}{send_key}: "{val}"\n'

        reconstructed = [re.sub(r'^l_([a-z]+)::\s*$', r'l_\1:', t) for t in reconstructed]
        tgt_code = self._LANG_CODE.get(target_lang, target_lang.lower())
        header_pat = re.compile(r'^l_[a-z_]+:\s*$')
        reconstructed = [f"l_{tgt_code}:\n" if header_pat.match(t) else t for t in reconstructed]
        if self.live_callback:
            self.live_callback(lines, reconstructed)
        return reconstructed

    # ── File processing ──

    def _process_file(self, input_path, output_path, source_lang, target_lang, model, temperature, max_tokens, batch_size, game="None"):
        with open(input_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
        if not lines:
            return
        total = len(lines)
        result = []
        first = lines[0]
        id_match = re.match(r"^(l_[a-z]+:)", first)
        if id_match:
            target_code = {
                "English": "english", "Korean": "korean", "Simplified Chinese": "simp_chinese",
                "French": "french", "German": "german", "Spanish": "spanish", "Japanese": "japanese",
                "Brazilian Portuguese": "braz_por", "Russian": "russian", "Polish": "polish"
            }.get(target_lang, target_lang.lower())
            result.append(f"l_{target_code}:{first[first.index(':') + 1:]}")
            content = lines[1:]
        else:
            result.append(first)
            content = lines[1:]
        for i in range(0, len(content), batch_size):
            if self.stop_event.is_set():
                self.log_callback("[STOPPED] Translation interrupted")
                if len(result) > 1:
                    self._save_checkpoint(result, output_path)
                return
            batch = content[i:i + batch_size]
            self.log_callback(f"  Translating lines {len(result)+1}-{len(result)+len(batch)}/{total}")
            translated = self._translate_batch(batch, source_lang, target_lang, model, temperature, max_tokens, game)
            result.extend(translated)
            self.progress_callback(min(len(result), total), total)
        with open(output_path, "w", encoding="utf-8-sig") as f:
            f.writelines(result)
        self.log_callback(f"  Saved: {output_path}")

    def _worker(self, input_dir, output_dir, source_lang, target_lang, model, temperature, max_tokens, batch_size, game="None"):
        _codes = {"English": "english", "Korean": "korean", "Simplified Chinese": "simp_chinese",
                  "French": "french", "German": "german", "Spanish": "spanish", "Japanese": "japanese",
                  "Russian": "russian", "Polish": "polish", "Brazilian Portuguese": "braz_por"}
        source_code = _codes.get(source_lang, source_lang.lower())
        target_code = _codes.get(target_lang, target_lang.lower())
        files = []
        for root, dirs, fnames in os.walk(input_dir):
            if root[len(input_dir):].count(os.sep) >= 1:
                dirs[:] = []
            for fn in fnames:
                if f"l_{source_code}" in fn.lower() and fn.lower().endswith((".yml", ".yaml")):
                    files.append(os.path.join(root, fn))
        if not files:
            self.log_callback(f"No files found with language identifier 'l_{source_code}'")
            self.busy = False
            self.status_callback("idle")
            return
        self.log_callback(f"Found {len(files)} files. Starting translation...")
        self.status_callback("translating")
        test = self._call_ollama(model, "test", temperature=0.1, max_tokens=1)
        if test.startswith("[OLLAMA_"):
            self.log_callback(f"[ERROR] Cannot connect to Ollama at {self.base_url}")
            self.busy = False
            self.status_callback("idle")
            return
        self.log_callback(f"Ollama connected. Using model: {model}")

        def _translate_one(fp):
            if self.stop_event.is_set():
                return
            base = os.path.basename(fp)
            rel = os.path.relpath(os.path.dirname(fp), input_dir)
            new_base = re.sub(f"l_{source_code}", f"l_{target_code}", base, flags=re.IGNORECASE)
            if rel == ".":
                out_path = os.path.join(output_dir, new_base)
            else:
                out_path = os.path.join(output_dir, rel, new_base)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
            self.log_callback(f"  Processing: {os.path.basename(fp)}")
            self._process_file(fp, out_path, source_lang, target_lang, model, temperature, max_tokens, batch_size, game)

        stopped = False
        for fp in files:
            if self.stop_event.is_set():
                stopped = True
                break
            _translate_one(fp)
        self.busy = False
        if stopped:
            self.log_callback("[STOPPED] Translation stopped. Checkpoints saved")
        else:
            self.log_callback("All done!")
            for fp in files:
                base = os.path.basename(fp)
                rel = os.path.relpath(os.path.dirname(fp), input_dir)
                new_base = re.sub(f"l_{source_code}", f"l_{target_code}", base, flags=re.IGNORECASE)
                out_path = os.path.join(output_dir, new_base) if rel == "." else os.path.join(output_dir, rel, new_base)
                if os.path.exists(out_path):
                    issues = self.check_quality(fp, out_path, source_lang, target_lang)
                    if issues:
                        counts = {"UNTRANSLATED": 0, "FOREIGN": 0, "DUPLICATE": 0}
                        for _, _, _, typ, _ in issues:
                            if typ in counts:
                                counts[typ] += 1
                        parts = [f"{k.lower()} {v}" for k, v in counts.items() if v > 0]
                        self.log_callback(f"  [VALIDATE] {new_base}: {', '.join(parts)}")
        self.status_callback("idle")

    # ── Glossary / Validation helpers ──

    @staticmethod
    def _glossary_dir():
        d = os.path.join(os.path.dirname(CONFIG_FILE), "glossary")
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def _get_glossary_text(target_lang, game_name="None"):
        combined = {}
        game_dir = os.path.join(OllamaTranslator._glossary_dir(), game_name)
        if not os.path.isdir(game_dir):
            return ""
        try:
            files = sorted([f for f in os.listdir(game_dir) if f.endswith(f"_{target_lang.lower()}.txt")], reverse=True)
            for fn in files:
                with open(os.path.join(game_dir, fn), "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or ":" not in line:
                            continue
                        src, tgt = line.split(":", 1)
                        combined[src.strip()] = tgt.strip()
        except Exception:
            pass
        if not combined:
            return ""
        text = "\n".join(f"  {s} -> {t}" for s, t in combined.items())
        return (
            f"\n[REFERENCE GLOSSARY]\n{text}\n\n"
            "The terms above are base forms. Both the source and the translation may "
            "appear in different conjugated forms depending on context.\n"
            "- Use this glossary ONLY when the source term is contextually relevant.\n"
            "- If the glossary term does NOT fit the context, translate naturally instead.\n"
            "- Do NOT force-match glossary entries to unrelated text.\n"
        )

    @staticmethod
    def _strip_codes(text):
        return re.sub(r'\[.*?\]|\$.*?\$|§.', '', text)

    @staticmethod
    def _find_duplicate_keys(lines):
        keys = {}
        for i, line in enumerate(lines, 1):
            m = re.match(r'^([\w.]+):\s*["\[]', line)
            if m:
                keys.setdefault(m.group(1), []).append(i)
        return {k: v for k, v in keys.items() if len(v) > 1}

    TEXT_GROUPS = {
        "CJK": re.compile(r'[\u4E00-\u9FFF\u3400-\u4DBF]'),
        "KOREAN": re.compile(r'[\uAC00-\uD7AF]'),
        "KANA": re.compile(r'[\u3040-\u309F\u30A0-\u30FF]'),
        "CYRILLIC": re.compile(r'[\u0400-\u04FF]'),
        "LATIN": re.compile(r'[a-zA-Z\u00C0-\u024F]'),
    }
    ALLOWED_GROUPS = {
        "Korean": {"KOREAN", "CJK", "KANA"}, "Japanese": {"KANA", "CJK", "KOREAN"},
        "Simplified Chinese": {"CJK"}, "Russian": {"CYRILLIC"}, "English": {"LATIN"},
        "French": {"LATIN"}, "German": {"LATIN"}, "Spanish": {"LATIN"},
        "Brazilian Portuguese": {"LATIN"}, "Polish": {"LATIN"},
    }

    def _has_foreign_chars(self, text, target_lang):
        clean = self._strip_codes(text)
        allowed = self.ALLOWED_GROUPS.get(target_lang, set())
        if not allowed:
            return False
        return any(pat.search(clean) for grp, pat in self.TEXT_GROUPS.items() if pat.search(clean) and grp not in allowed)

    def check_quality(self, input_path, output_path, source_lang, target_lang):
        try:
            with open(input_path, "r", encoding="utf-8-sig") as f:
                src_lines = [l.rstrip("\n") for l in f.readlines()]
            with open(output_path, "r", encoding="utf-8-sig") as f:
                tgt_lines = [l.rstrip("\n") for l in f.readlines()]
        except FileNotFoundError:
            return []
        issues = []
        dups = self._find_duplicate_keys(tgt_lines)
        min_len = min(len(src_lines), len(tgt_lines))
        if len(src_lines) != len(tgt_lines):
            issues.append((0, f"Line count: src={len(src_lines)} tgt={len(tgt_lines)}", "", "MISMATCH", ""))
        for i in range(1, min_len):
            s = src_lines[i]
            t = tgt_lines[i]
            if not s.strip() or s.strip().startswith('#'):
                continue
            m = re.match(r'^([\w.]+):\s*', s)
            if not m:
                continue
            key = m.group(1)
            sv = re.match(r'^([\w.]+):\s*"(.+)"', s)
            tv = re.match(r'^([\w.]+):\s*"(.+)"', t)
            if not sv or not tv:
                continue
            s_val, t_val = sv.group(2), tv.group(2)
            cs, ct = self._strip_codes(s_val), self._strip_codes(t_val)
            if not re.search(r'[\uAC00-\uD7AFa-zA-Z\u00C0-\u024F\u4E00-\u9FFF\u3040-\u30FF\u0400-\u04FF]', ct):
                continue
            dup_info = f"key '{key}' dup at {dups[key]}" if key in dups else ""
            if cs == ct:
                issues.append((i, s, t, "UNTRANSLATED", dup_info))
            elif self._has_foreign_chars(t_val, target_lang):
                issues.append((i, s, t, "FOREIGN", dup_info))
            elif dup_info:
                issues.append((i, s, t, "DUPLICATE", dup_info))
        return issues

    def start(self, input_dir, output_dir, source_lang, target_lang, model, temperature, max_tokens, batch_size, game="None", max_retries=3):
        self.stop_event.clear()
        self.max_retries = max_retries
        self.busy = True
        def _run():
            try:
                self._worker(input_dir, output_dir, source_lang, target_lang, model, temperature, max_tokens, batch_size, game)
            except Exception as e:
                self.log_callback(f"[FATAL] Unhandled error in translation thread: {e}")
                import traceback
                for line in traceback.format_exc().split("\n"):
                    self.log_callback(line)
                self.busy = False
                self.status_callback("idle")
        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

# ── GUI ──

class OllamaTranslatorGUI(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Ollama PDX Translator")
        self.geometry("800x700")
        self.stop_event = threading.Event()
        self.ollama_url = ctk.StringVar()
        self.input_dir = ctk.StringVar()
        self.output_dir = ctk.StringVar()
        self.source_lang = ctk.StringVar(value="English")
        self.target_lang = ctk.StringVar(value="Korean")
        self.ollama_model = ctk.StringVar()
        self.temperature = ctk.DoubleVar(value=0.2)
        self.max_tokens = ctk.IntVar(value=8192)
        self.batch_size = ctk.IntVar(value=20)
        self.max_retries = ctk.IntVar(value=3)
        self.selected_game = ctk.StringVar()
        self.available_games = list(GAME_PROMPTS.keys())
        self.show_prompt = ctk.BooleanVar(value=False)
        self.checkpoint_enabled = ctk.BooleanVar(value=True)
        self.debug_mode = ctk.BooleanVar(value=False)
        self.prompt_template_var = ctk.StringVar(value=self._default_prompt())
        self.live_visible = ctk.BooleanVar(value=False)
        self._connected = False
        self._m_modname = "mod"
        self._g_page = 0
        self._g_per_page = 20
        self._m_page = 0
        self._m_per_page = 20
        self._validate_file_pairs = []
        self._validate_all_data = []
        self._validate_modified = {}
        self._validate_page = 0
        self._validate_hide_filter = ctk.BooleanVar(value=False)
        self._validate_per_page = ctk.IntVar(value=20)
        self.available_langs = ["English", "Korean", "Simplified Chinese", "French", "German",
                                "Spanish", "Japanese", "Brazilian Portuguese", "Russian", "Polish"]
        self.engine = OllamaTranslator(
            log_callback=self.log, progress_callback=self.update_progress,
            status_callback=self.set_status, stop_event=self.stop_event,
            live_callback=self._on_live_result
        )
        self._build_ui()
        self._load_config()
        self._init_log_file()
        for sv in [self.ollama_url, self.ollama_model, self.input_dir, self.output_dir, self.source_lang, self.target_lang]:
            sv.trace_add("write", self._validate_fields)
        self._validate_fields()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Config ──

    def _config_path(self):
        return CONFIG_FILE

    def _load_config(self):
        try:
            if not os.path.exists(self._config_path()):
                return
            with open(self._config_path(), "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.ollama_url.set(cfg.get("ollama_url", ""))
            self.ollama_model.set(cfg.get("ollama_model", "llama3.1"))
            self.input_dir.set(cfg.get("input_dir", ""))
            self.output_dir.set(cfg.get("output_dir", ""))
            self.source_lang.set(cfg.get("source_lang", ""))
            self.target_lang.set(cfg.get("target_lang", ""))
            sg = cfg.get("selected_game", "")
            self.selected_game.set(sg if sg in self.available_games else self.available_games[0])
            self.temperature.set(cfg.get("temperature", 0.2))
            self.max_tokens.set(cfg.get("max_tokens", 8192))
            self.batch_size.set(cfg.get("batch_size", 1))
            self.max_retries.set(cfg.get("max_retries", 3))
            p = cfg.get("prompt_template", "")
            if p:
                self.prompt_template_var.set(p)
                self.prompt_textbox.delete("1.0", "end")
                self.prompt_textbox.insert("1.0", p)
            if cfg.get("show_prompt", False):
                self.show_prompt.set(True)
                self.prompt_frame.grid()
            if hasattr(self, '_g_game_var'):
                gg = cfg.get("glossary_game", "")
                if gg in self.available_games:
                    self._g_game_var.set(gg)
                self._g_src_var.set(cfg.get("glossary_src", "English"))
                self._g_tgt_var.set(cfg.get("glossary_tgt", "Korean"))
                self._g_min_var.set(str(cfg.get("glossary_min", 3)))
                self._g_folder_var.set(cfg.get("glossary_folder", ""))
            if hasattr(self, '_m_game_var'):
                mg = cfg.get("glossary_mod_game", "")
                if mg in self.available_games:
                    self._m_game_var.set(mg)
                self._m_tgt_var.set(cfg.get("glossary_mod_tgt", "Korean"))
                self._m_min_var.set(str(cfg.get("glossary_mod_min", 3)))
                self._m_folder_var.set(cfg.get("glossary_mod_folder", ""))
                self._m_modname = cfg.get("glossary_mod_name", "mod")
        except Exception:
            pass

    def _save_config(self):
        self._sync_prompt()
        cfg = {
            "ollama_url": self.ollama_url.get(),
            "ollama_model": self.ollama_model.get(),
            "input_dir": self.input_dir.get(),
            "output_dir": self.output_dir.get(),
            "source_lang": self.source_lang.get(),
            "target_lang": self.target_lang.get(),
            "selected_game": self.selected_game.get(),
            "temperature": self.temperature.get(),
            "max_tokens": self.max_tokens.get(),
            "batch_size": self.batch_size.get(),
            "max_retries": self.max_retries.get(),
            "prompt_template": self.prompt_template_var.get(),
            "show_prompt": self.show_prompt.get(),
            "glossary_game": self._g_game_var.get() if hasattr(self, '_g_game_var') else "",
            "glossary_src": self._g_src_var.get() if hasattr(self, '_g_src_var') else "",
            "glossary_tgt": self._g_tgt_var.get() if hasattr(self, '_g_tgt_var') else "",
            "glossary_min": int(self._g_min_var.get()) if hasattr(self, '_g_min_var') else 3,
            "glossary_folder": self._g_folder_var.get() if hasattr(self, '_g_folder_var') else "",
            "glossary_mod_game": self._m_game_var.get() if hasattr(self, '_m_game_var') else "",
            "glossary_mod_tgt": self._m_tgt_var.get() if hasattr(self, '_m_tgt_var') else "",
            "glossary_mod_min": int(self._m_min_var.get()) if hasattr(self, '_m_min_var') else 3,
            "glossary_mod_folder": self._m_folder_var.get() if hasattr(self, '_m_folder_var') else "",
            "glossary_mod_name": self._m_modname if hasattr(self, '_m_modname') else "",
        }
        try:
            with open(self._config_path(), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _on_close(self):
        self._g_running = False
        self._m_running = False
        if getattr(self, '_validate_modified', None):
            dlg = ctk.CTkToplevel(self, fg_color="white")
            dlg.title("Exit")
            dlg.transient(self)
            dlg.grab_set()
            dlg.grid_columnconfigure(0, weight=1)
            result = [None]
            ctk.CTkLabel(dlg, text="Save changes before exiting?", font=ctk.CTkFont(size=13), text_color="black").grid(row=0, column=0, pady=(15,10))
            bf = ctk.CTkFrame(dlg, fg_color="white")
            bf.grid(row=1, column=0, pady=5)
            for i, t in enumerate(["Save & Exit", "Exit", "Cancel"]):
                ctk.CTkButton(bf, text=t, width=100, fg_color="#D32F2F" if i == 1 else "#2E7D32" if i == 0 else "#757575",
                              text_color="white", hover_color="#E53935" if i == 1 else "#388E3C" if i == 0 else "#9E9E9E",
                              command=lambda v=i: [result.__setitem__(0, [True, False, None][v]), dlg.destroy()]).pack(side="left", padx=5)
            dlg.update_idletasks()
            pw, ph = self.winfo_width(), self.winfo_height()
            px, py = self.winfo_x(), self.winfo_y()
            dw, dh = dlg.winfo_width(), dlg.winfo_height()
            dlg.geometry(f"+{px + (pw - dw)//2}+{py + (ph - dh)//2}")
            dlg.wait_window()
            if result[0] is None:
                return
            if result[0]:
                self._validate_save()
        self.engine.kill_server()
        self._save_config()
        self.destroy()

    def _validate_fields(self, *args):
        ok = all([self.ollama_url.get(), self.ollama_model.get(),
                  self.input_dir.get(), self.output_dir.get(),
                  self.source_lang.get(), self.target_lang.get()])
        st = "normal" if (ok and self._connected) else "disabled"
        self.start_btn.configure(state=st)
        if hasattr(self, '_validate_retry_btn'):
            self._validate_retry_btn.configure(state=st)

    def _default_prompt(self):
        return ("Translate the following text from '{source_lang}' to '{target_lang}'.\n"
                "Rules:\n1. Preserve all {{PH0}}, {{PH1}}, etc. placeholders exactly as-is.\n"
                "2. Preserve line markers like \u27e80\u27e9 \u27e81\u27e9 exactly as-is.\n"
                "3. Do NOT wrap in code blocks or add explanations.\n\n{batch_text}")

    # ── UI Build ──

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, sticky="nsew")
        t_trans = self.tabview.add("Translate")
        t_val = self.tabview.add("Validate")
        self.tabview.set("Translate")
        t_trans.grid_columnconfigure(0, weight=1)
        t_val.grid_columnconfigure(0, weight=1)
        t_val.grid_rowconfigure(2, weight=1)
        topf = ctk.CTkFrame(t_val)
        topf.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        topf.grid_columnconfigure(1, weight=1)
        self._validate_combo = ctk.CTkComboBox(topf, values=["(no files)"], state="readonly", command=self._validate_load_file)
        self._validate_combo.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self._validate_msg = ctk.CTkLabel(topf, text="", font=ctk.CTkFont(size=12))
        self._validate_msg.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkButton(topf, text="Load Files", command=self._validate_load_files, width=90).grid(row=0, column=2, padx=5)
        cf = ctk.CTkFrame(t_val)
        cf.grid(row=1, column=0, padx=10, pady=2, sticky="ew")
        ctk.CTkCheckBox(cf, text="Show headers/comments/blanks", variable=self._validate_hide_filter,
                        font=ctk.CTkFont(size=12), command=self._validate_render).pack(side="left", padx=5)
        ctk.CTkLabel(cf, text="Lines/page:", font=ctk.CTkFont(size=11)).pack(side="right", padx=(5,2))
        self._validate_per_page_entry = ctk.CTkEntry(cf, textvariable=self._validate_per_page, width=50)
        self._validate_per_page_entry.pack(side="right")
        def _on_page_size_change(*a):
            val = self._validate_per_page.get()
            if val < 1:
                self._validate_per_page.set(1)
            self._validate_page = 0
            self._validate_render()
        self._validate_per_page.trace_add("write", _on_page_size_change)
        self._validate_frame = ctk.CTkScrollableFrame(t_val)
        self._validate_frame.grid(row=2, column=0, padx=5, pady=2, sticky="nsew")
        self._validate_frame.grid_columnconfigure(0, weight=1)
        pgf = ctk.CTkFrame(t_val)
        pgf.grid(row=3, column=0, padx=10, pady=(0,0), sticky="ew")
        self._validate_prev_btn = ctk.CTkButton(pgf, text="◀ Prev", width=60, command=self._validate_prev_page)
        self._validate_prev_btn.pack(side="left", padx=2)
        self._validate_page_label = ctk.CTkLabel(pgf, text="1/1", font=ctk.CTkFont(size=12))
        self._validate_page_label.pack(side="left", padx=5)
        self._validate_next_btn = ctk.CTkButton(pgf, text="Next ▶", width=60, command=self._validate_next_page)
        self._validate_next_btn.pack(side="left", padx=2)
        self._validate_info = ctk.CTkLabel(pgf, text="", font=ctk.CTkFont(size=12))
        self._validate_info.pack(side="right", padx=5)
        bf = ctk.CTkFrame(t_val)
        bf.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        self._validate_retry_btn = ctk.CTkButton(bf, text="Retry Selected", fg_color="#E65100", command=self._validate_retry_selected, state="disabled")
        self._validate_retry_btn.pack(side="left", padx=5)
        self._validate_save_btn = ctk.CTkButton(bf, text="Save Changes", command=self._validate_save)
        self._validate_save_btn.pack(side="left", padx=5)
        t_trans.grid_rowconfigure(7, weight=1)
        pf = ctk.CTkFrame(t_trans)
        pf.grid(row=5, column=0, padx=10, pady=0, sticky="ew")
        pf.grid_columnconfigure(0, weight=1)
        self.progress_bar = ctk.CTkProgressBar(pf)
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.progress_bar.set(0)
        self.progress_text = ctk.CTkLabel(pf, text="0 / 0 lines")
        self.progress_text.grid(row=0, column=1, padx=5)
        self.live_frame = ctk.CTkFrame(t_trans, height=120)
        self.live_frame.grid_propagate(False)
        self.live_frame.grid(row=6, column=0, padx=10, pady=0, sticky="ew")
        self.live_frame.grid_columnconfigure(0, weight=1)
        self.live_frame.grid_columnconfigure(2, weight=1)
        self.live_frame.grid_rowconfigure(1, weight=1)
        lh = ctk.CTkFrame(self.live_frame, fg_color="transparent")
        lh.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=(3, 0))
        lh.grid_columnconfigure(0, weight=1)
        lh.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(lh, text="Original", font=ctk.CTkFont(size=11, weight="bold"), anchor="w").grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(lh, text="Translated", font=ctk.CTkFont(size=11, weight="bold"), anchor="w").grid(row=0, column=1, sticky="w")
        self.live_orig = ctk.CTkTextbox(self.live_frame, wrap="none", font=ctk.CTkFont(size=11))
        self.live_orig.grid(row=1, column=0, sticky="nsew", padx=(3, 1), pady=3)
        self.live_trans = ctk.CTkTextbox(self.live_frame, wrap="none", font=ctk.CTkFont(size=11))
        self.live_trans.grid(row=1, column=2, sticky="nsew", padx=(1, 3), pady=3)
        self.live_frame.grid_remove()
        self.log_frame = ctk.CTkFrame(t_trans)
        self.log_frame.grid(row=7, column=0, padx=10, pady=0, sticky="nsew")
        self.log_frame.grid_columnconfigure(0, weight=1)
        self.log_frame.grid_rowconfigure(0, weight=1)
        self.log_text = ctk.CTkTextbox(self.log_frame, wrap="word", font=ctk.CTkFont(size=11))
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)
        ctk.CTkLabel(t_trans, text="Ollama Paradox Mod Translator",
                      font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, pady=(6, 2), sticky="n")
        sf = ctk.CTkFrame(t_trans)
        sf.grid(row=1, column=0, padx=10, pady=4, sticky="ew")
        for c in range(4):
            sf.grid_columnconfigure(c, weight=[0, 1, 0, 3][c])

        def _put(row, col, var, combo_vals=None, browse=None, extra_btns=None):
            f = ctk.CTkFrame(sf, fg_color="transparent")
            f.grid(row=row, column=col, sticky="ew", padx=(0, 10) if col == 1 else (0, 5), pady=3)
            f.grid_columnconfigure(0, weight=1)
            if combo_vals:
                w = ctk.CTkComboBox(f, variable=var, values=combo_vals, state="readonly")
                w.grid(row=0, column=0, sticky="ew")
            else:
                w = ctk.CTkEntry(f, textvariable=var)
                w.grid(row=0, column=0, sticky="ew")
            bc = 2
            if browse:
                ctk.CTkButton(f, text="Browse", width=70, command=browse).grid(row=0, column=bc, padx=(5, 0))
                bc += 1
            if extra_btns:
                for lbl, cmd in extra_btns:
                    ctk.CTkButton(f, text=lbl, width=70, command=cmd).grid(row=0, column=bc, padx=(5, 0))
                    bc += 1
            return w

        def _lb(row, col, text):
            ctk.CTkLabel(sf, text=text, anchor="w").grid(row=row, column=col, sticky="w", padx=5, pady=3)

        _lb(0, 0, "Model:")
        self.ollama_combo = ctk.CTkComboBox(sf, variable=self.ollama_model, values=["(none)"], state="readonly")
        self.ollama_combo.grid(row=0, column=1, padx=(0, 10), pady=3, sticky="ew")
        _lb(0, 2, "Ollama URL:")
        _put(0, 3, self.ollama_url, extra_btns=[("Start Ollama", self._start_ollama), ("Connect", self._connect_ollama)])
        _lb(1, 0, "Source:")
        _put(1, 1, self.source_lang, combo_vals=self.available_langs)
        _lb(1, 2, "Input Folder:")
        _put(1, 3, self.input_dir, browse=self._browse_input)
        _lb(2, 0, "Target:")
        _put(2, 1, self.target_lang, combo_vals=self.available_langs)
        _lb(2, 2, "Output Folder:")
        self.output_entry = _put(2, 3, self.output_dir, browse=self._browse_output)
        ctk.CTkLabel(sf, text="Game Preset:", anchor="w").grid(row=3, column=0, sticky="w", padx=5, pady=3)
        gf = ctk.CTkFrame(sf, fg_color="transparent")
        gf.grid(row=3, column=1, sticky="ew", padx=(0, 10), pady=3)
        gf.grid_columnconfigure(0, weight=1)
        ctk.CTkComboBox(gf, variable=self.selected_game, values=self.available_games, state="readonly").grid(row=0, column=0, sticky="ew")
        af = ctk.CTkFrame(sf, fg_color="transparent")
        af.grid(row=3, column=2, columnspan=2, sticky="ew", padx=5, pady=3)
        for lbl, var, w in [("Temperature:", self.temperature, 60), ("Tokens:", self.max_tokens, 70),
                            ("Batch:", self.batch_size, 50), ("Retries:", self.max_retries, 40)]:
            ctk.CTkLabel(af, text=lbl).pack(side="left", padx=(0, 5) if lbl == "Retries:" else (10, 5))
            ctk.CTkEntry(af, textvariable=var, width=w).pack(side="left", padx=(0, 10))
        cb_frame = ctk.CTkFrame(t_trans, fg_color="transparent")
        cb_frame.grid(row=2, column=0, padx=10, pady=0, sticky="ew")
        ctk.CTkCheckBox(cb_frame, text="Edit Prompt", variable=self.show_prompt,
                        font=ctk.CTkFont(size=12), command=self._toggle_prompt).pack(side="left", padx=0)
        ctk.CTkCheckBox(cb_frame, text="Checkpoint", variable=self.checkpoint_enabled,
                        font=ctk.CTkFont(size=12)).pack(side="left", padx=(15, 0))
        ctk.CTkCheckBox(cb_frame, text="Debug Log", variable=self.debug_mode,
                        font=ctk.CTkFont(size=12)).pack(side="left", padx=(15, 0))
        self.prompt_frame = ctk.CTkFrame(t_trans)
        self.prompt_frame.grid(row=3, column=0, padx=10, pady=0, sticky="ew")
        self.prompt_frame.grid_columnconfigure(0, weight=1)
        self.prompt_frame.grid_rowconfigure(1, weight=1)
        ptb = ctk.CTkFrame(self.prompt_frame, fg_color="transparent")
        ptb.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))
        ptb.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(ptb, text="Restore Default", width=120, command=self._restore_default_prompt).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(ptb, text="Load from .txt", width=120, command=self._load_prompt_from_file).grid(row=0, column=1, sticky="w", padx=(5, 0))
        self.prompt_textbox = ctk.CTkTextbox(self.prompt_frame, height=150, font=ctk.CTkFont(size=12))
        self.prompt_textbox.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.prompt_textbox.insert("1.0", self.prompt_template_var.get())
        self.prompt_textbox.bind("<KeyRelease>", self._sync_prompt)
        self.prompt_frame.grid_remove()
        cf = ctk.CTkFrame(t_trans)
        cf.grid(row=4, column=0, padx=10, pady=4, sticky="ew")
        self.start_btn = ctk.CTkButton(cf, text="Start Translation", command=self._start, fg_color="#2E7D32", hover_color="#388E3C")
        self.start_btn.pack(side="left", padx=5)
        self.stop_btn = ctk.CTkButton(cf, text="Stop", command=self._stop, fg_color="#D32F2F", hover_color="#E53935", state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        self.reset_btn = ctk.CTkButton(cf, text="Reset", command=self._reset_ui, fg_color="#757575", hover_color="#9E9E9E", state="disabled")
        self.reset_btn.pack(side="left", padx=5)
        self.live_btn = ctk.CTkButton(cf, text="Live", command=self._toggle_live, fg_color="#1565C0", hover_color="#1976D2", width=60)
        self.live_btn.pack(side="left", padx=5)
        self.status_label = ctk.CTkLabel(cf, text="Ready", text_color="gray")
        self.status_label.pack(side="right", padx=10)
        t_gl = self.tabview.add("Glossary")
        t_gl.grid_columnconfigure(0, weight=1)
        t_gl.grid_rowconfigure(0, weight=1)
        gl_sub = ctk.CTkTabview(t_gl)
        gl_sub.grid(row=0, column=0, sticky="nsew")
        game_tab = gl_sub.add("GAME")
        mod_tab = gl_sub.add("MOD")
        game_tab.grid_columnconfigure(0, weight=1)
        game_tab.grid_rowconfigure(7, weight=1)
        LANG_KEYS = ["English","Korean","Simplified Chinese","French","German","Spanish","Japanese","Russian","Polish","Brazilian Portuguese"]
        r = 0
        ctk.CTkLabel(game_tab, text="Game:", font=ctk.CTkFont(size=12)).grid(row=r, column=0, padx=10, pady=(10,2), sticky="w")
        r += 1
        self._g_game_var = ctk.StringVar(value=self.available_games[0])
        ctk.CTkComboBox(game_tab, variable=self._g_game_var, values=self.available_games, state="readonly").grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        r += 1
        self._g_folder_var = ctk.StringVar()
        fgf = ctk.CTkFrame(game_tab, fg_color="transparent")
        fgf.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        fgf.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(fgf, text="Lang Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkEntry(fgf, textvariable=self._g_folder_var).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(fgf, text="Browse", width=70, command=self._g_browse).grid(row=0, column=2, padx=5, pady=5)
        r += 1
        lf1 = ctk.CTkFrame(game_tab, fg_color="transparent")
        lf1.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        self._g_src_var = ctk.StringVar(value="English")
        ctk.CTkLabel(lf1, text="Source:").pack(side="left", padx=5)
        ctk.CTkComboBox(lf1, variable=self._g_src_var, values=LANG_KEYS, state="readonly", width=150).pack(side="left", padx=5)
        ctk.CTkLabel(lf1, text="  Target:").pack(side="left", padx=5)
        self._g_tgt_var = ctk.StringVar(value="Korean")
        ctk.CTkComboBox(lf1, variable=self._g_tgt_var, values=LANG_KEYS, state="readonly", width=150).pack(side="left", padx=5)
        r += 1
        lf2 = ctk.CTkFrame(game_tab, fg_color="transparent")
        lf2.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        self._g_min_var = ctk.StringVar(value="3")
        ctk.CTkLabel(lf2, text="Min co-occurrence:").pack(side="left", padx=5)
        ctk.CTkEntry(lf2, textvariable=self._g_min_var, width=50).pack(side="left", padx=5)
        ctk.CTkButton(lf2, text="Extract", fg_color="#1565C0", command=self._game_extract).pack(side="left", padx=15)
        r += 1
        self._g_search_var = ctk.StringVar()
        sf = ctk.CTkFrame(game_tab, fg_color="transparent")
        sf.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        sf.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(sf, text="Search:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        se = ctk.CTkEntry(sf, textvariable=self._g_search_var, width=400)
        se.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self._g_info_label = ctk.CTkLabel(sf, text="")
        self._g_info_label.grid(row=0, column=2, padx=10, pady=5, sticky="e")
        se.bind("<KeyRelease>", lambda e: self._game_update_display(self._g_search_var.get()))
        r += 1
        self._g_progress = ctk.CTkProgressBar(game_tab)
        self._g_progress.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        self._g_progress.set(0)
        r += 1
        self._g_result_frame = ctk.CTkScrollableFrame(game_tab)
        self._g_result_frame.grid(row=r, column=0, padx=10, pady=5, sticky="nsew")
        self._g_result_frame.grid_columnconfigure(2, weight=1)
        r += 1
        pgf = ctk.CTkFrame(game_tab, fg_color="transparent")
        pgf.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        self._g_page_label = ctk.CTkLabel(pgf, text="")
        self._g_page_label.pack(side="left", padx=5)
        ctk.CTkButton(pgf, text="< Prev", width=60, command=self._g_prev_page).pack(side="left", padx=5)
        ctk.CTkButton(pgf, text="Next >", width=60, command=self._g_next_page).pack(side="left", padx=5)
        r += 1
        gbtn = ctk.CTkFrame(game_tab, fg_color="transparent")
        gbtn.grid(row=r, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(gbtn, text="Save Glossary", fg_color="#2E7D32", command=self._game_save).pack(side="left", padx=5)
        ctk.CTkButton(gbtn, text="Load Glossary", command=self._game_load).pack(side="left", padx=5)
        mod_tab.grid_columnconfigure(0, weight=1)
        mod_tab.grid_rowconfigure(6, weight=1)
        r = 0
        ctk.CTkLabel(mod_tab, text="Game:", font=ctk.CTkFont(size=12)).grid(row=r, column=0, padx=10, pady=(10,2), sticky="w")
        r += 1
        self._m_game_var = ctk.StringVar(value=self.available_games[0])
        ctk.CTkComboBox(mod_tab, variable=self._m_game_var, values=self.available_games, state="readonly").grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        r += 1
        self._m_folder_var = ctk.StringVar()
        self._m_src_var = ctk.StringVar(value="English")
        self._m_tgt_var = ctk.StringVar(value="Korean")
        mff = ctk.CTkFrame(mod_tab, fg_color="transparent")
        mff.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        mff.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(mff, text="localisation Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkEntry(mff, textvariable=self._m_folder_var).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(mff, text="Browse", width=70, command=self._m_browse).grid(row=0, column=2, padx=5, pady=5)
        r += 1
        mlf = ctk.CTkFrame(mod_tab, fg_color="transparent")
        mlf.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        ctk.CTkLabel(mlf, text="Source:").pack(side="left", padx=5)
        self._m_src_combo = ctk.CTkComboBox(mlf, variable=self._m_src_var, values=LANG_KEYS, state="readonly", width=150)
        self._m_src_combo.pack(side="left", padx=5)
        ctk.CTkLabel(mlf, text="  Target:").pack(side="left", padx=5)
        self._m_tgt_combo = ctk.CTkComboBox(mlf, variable=self._m_tgt_var, values=LANG_KEYS, state="readonly", width=150)
        self._m_tgt_combo.pack(side="left", padx=5)
        r += 1
        self._m_min_var = ctk.StringVar(value="3")
        mf2 = ctk.CTkFrame(mod_tab, fg_color="transparent")
        mf2.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        ctk.CTkLabel(mf2, text="Min frequency:").pack(side="left", padx=5)
        ctk.CTkEntry(mf2, textvariable=self._m_min_var, width=50).pack(side="left", padx=5)
        ctk.CTkButton(mf2, text="Extract Terms", fg_color="#1565C0", command=self._mod_extract).pack(side="left", padx=15)
        ctk.CTkButton(mf2, text="Translate with LLM", fg_color="#E65100", command=self._mod_translate).pack(side="left", padx=5)
        r += 1
        self._m_search_var = ctk.StringVar()
        msf = ctk.CTkFrame(mod_tab, fg_color="transparent")
        msf.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        msf.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(msf, text="Search:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        mse = ctk.CTkEntry(msf, textvariable=self._m_search_var, width=400)
        mse.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self._m_info_label = ctk.CTkLabel(msf, text="")
        self._m_info_label.grid(row=0, column=2, padx=10, pady=5, sticky="e")
        mse.bind("<KeyRelease>", lambda e: self._mod_update_display(self._m_search_var.get()))
        r += 1
        self._m_progress = ctk.CTkProgressBar(mod_tab)
        self._m_progress.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        self._m_progress.set(0)
        r += 1
        self._m_result_frame = ctk.CTkScrollableFrame(mod_tab)
        self._m_result_frame.grid(row=r, column=0, padx=10, pady=5, sticky="nsew")
        self._m_result_frame.grid_columnconfigure(2, weight=1)
        r += 1
        mpgf = ctk.CTkFrame(mod_tab, fg_color="transparent")
        mpgf.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        self._m_page_label = ctk.CTkLabel(mpgf, text="")
        self._m_page_label.pack(side="left", padx=5)
        ctk.CTkButton(mpgf, text="< Prev", width=60, command=self._m_prev_page).pack(side="left", padx=5)
        ctk.CTkButton(mpgf, text="Next >", width=60, command=self._m_next_page).pack(side="left", padx=5)
        r += 1
        mbtn = ctk.CTkFrame(mod_tab, fg_color="transparent")
        mbtn.grid(row=r, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(mbtn, text="Retry Selected", fg_color="#E65100", command=self._mod_retry_selected).pack(side="left", padx=5)
        ctk.CTkButton(mbtn, text="Save Glossary", fg_color="#2E7D32", command=self._mod_save).pack(side="left", padx=5)
        ctk.CTkButton(mbtn, text="Load Glossary", command=self._mod_load).pack(side="left", padx=5)

    # ── Validate Tab ──

    def _validate_load_files(self):
        inp = self.input_dir.get()
        out = self.output_dir.get()
        src = self.source_lang.get()
        tgt = self.target_lang.get()
        if not inp or not out or not src or not tgt:
            self._validate_msg.configure(text="Set Input/Output folders and languages in Translate tab first", text_color="red")
            return
        src_code = {"English":"english","Korean":"korean","Simplified Chinese":"simp_chinese","French":"french","German":"german","Spanish":"spanish","Japanese":"japanese","Russian":"russian","Polish":"polish","Brazilian Portuguese":"braz_por"}.get(src, src.lower())
        tgt_code = {"English":"korean","Korean":"korean","Simplified Chinese":"simp_chinese","French":"french","German":"german","Spanish":"spanish","Japanese":"japanese","Russian":"russian","Polish":"polish","Brazilian Portuguese":"braz_por"}.get(tgt, tgt.lower())
        self._validate_file_pairs = []
        for root, dirs, fnames in os.walk(inp):
            if root[len(inp):].count(os.sep) >= 1:
                dirs[:] = []
            for fn in fnames:
                if f"l_{src_code}" not in fn or not fn.endswith((".yml", ".yaml")):
                    continue
                base = re.sub(r'_?l_[a-z]+_?', '', fn).replace(".yml", "").replace(".yaml", "")
                in_path = os.path.join(root, fn)
                rel = os.path.relpath(root, inp)
                out_fn = re.sub(r'_?l_[a-z]+_?', f'_l_{tgt_code}', fn, flags=re.IGNORECASE)
                out_path = os.path.join(out, out_fn) if rel == "." else os.path.join(out, rel, out_fn)
                if not os.path.exists(out_path):
                    continue
                self._validate_file_pairs.append({"base": base, "in": in_path, "out": out_path, "issues": 0, "severity": None})
        if not self._validate_file_pairs:
            self._validate_msg.configure(text="No matching file pairs found", text_color="red")
            return
        self._validate_file_pairs.sort(key=lambda x: (0 if x["severity"] else 1, x["base"]))
        vals = []
        for fp in self._validate_file_pairs:
            prefix = "⚠ " if fp["severity"] == "warn" else "✗ " if fp["severity"] == "error" else ""
            vals.append(f"{prefix}{fp['base']}")
        self._validate_combo.configure(values=vals)
        self._validate_combo.set(vals[0])
        self._validate_msg.configure(text=f"{len(self._validate_file_pairs)} files", text_color="gray")
        self._validate_load_file(vals[0].replace("⚠ ", "").replace("✗ ", ""))

    def _validate_load_file(self, base_name):
        base_name = base_name.replace("⚠ ", "").replace("✗ ", "")
        fp = next((f for f in self._validate_file_pairs if f["base"] == base_name), None)
        if not fp:
            return
        src = self.source_lang.get()
        tgt = self.target_lang.get()
        issues = self.engine.check_quality(fp["in"], fp["out"], src, tgt)
        with open(fp["in"], "r", encoding="utf-8-sig") as f:
            src_lines = [l.rstrip("\n") for l in f.readlines()]
        with open(fp["out"], "r", encoding="utf-8-sig") as f:
            tgt_lines = [l.rstrip("\n") for l in f.readlines()]
        self._validate_all_data = []
        has_issues = False
        sev = None
        for i in range(max(len(src_lines), len(tgt_lines))):
            s = src_lines[i] if i < len(src_lines) else ""
            t = tgt_lines[i] if i < len(tgt_lines) else ""
            key = re.match(r"^\s*([\w.]+):\s*", s)
            key_text = key.group(1) if key else ""
            src_val = re.sub(r"^\s*[\w.]+:\s*", "", s).strip('" ')
            tgt_val = re.sub(r"^\s*[\w.]+:\s*", "", t).strip('" ')
            status = "H" if key_text.startswith("l_") else "-" if (not s.strip() or s.strip().startswith("#")) else "✓"
            for iss in issues:
                if iss[0] == i:
                    status = {"UNTRANSLATED": "✗!", "FOREIGN": "✗?", "DUPLICATE": "✗D", "MISMATCH": "✗X"}.get(iss[3], "✗")
                    if iss[3] in ("UNTRANSLATED", "FOREIGN"): sev = "warn"
                    elif iss[3] == "MISMATCH": sev = "error"
                    has_issues = True
                    break
            self._validate_all_data.append({"key": key_text, "src": src_val, "tgt": tgt_val, "status": status, "ln": i, "checked": False, "_idx": len(self._validate_all_data)})
        fp["issues"] = sum(1 for d in self._validate_all_data if d["status"].startswith("✗"))
        fp["severity"] = sev
        self._validate_page = 0
        self._validate_render()
        self._validate_update_file_list()

    def _validate_update_file_list(self):
        vals = []
        for fp in self._validate_file_pairs:
            prefix = "⚠ " if fp["severity"] == "warn" else "✗ " if fp["severity"] == "error" else ""
            vals.append(f"{prefix}{fp['base']}")
        self._validate_combo.configure(values=vals)
        current = self._validate_combo.get()
        clean = current.replace("⚠ ", "").replace("✗ ", "")
        match = next((v for v in vals if v.endswith(clean)), vals[0] if vals else "")
        self._validate_combo.set(match)

    def _validate_render(self):
        for w in self._validate_frame.winfo_children():
            w.destroy()
        data = self._validate_all_data
        if not self._validate_hide_filter.get():
            data = [d for d in data if d["status"] not in ("H", "-")]
        total = len(data)
        per_page = self._validate_per_page.get()
        if per_page < 1:
            per_page = 1
        pages = max(1, (total - 1) // per_page + 1)
        if self._validate_page >= pages:
            self._validate_page = pages - 1
        start = self._validate_page * per_page
        page = data[start:start + per_page]
        WT = [3, 4, 4]
        FIXED = [40, 32, 30]
        ROW_H = 28
        LABELS = ["KEY", "Source", "Target", "Sts", "Ln", "☐"]
        for ci in range(6):
            if ci < 3:
                self._validate_frame.grid_columnconfigure(ci, weight=WT[ci], minsize=80)
            else:
                self._validate_frame.grid_columnconfigure(ci, weight=0, minsize=FIXED[ci-3])
        self._validate_frame.grid_columnconfigure(6, weight=0)
        page_chks = []
        for ci, txt in enumerate(LABELS):
            if ci < 5:
                kwargs = {"anchor": "w", "font": ctk.CTkFont(size=11, weight="bold")}
                if ci >= 3:
                    kwargs["width"] = FIXED[ci-3]
                ctk.CTkLabel(self._validate_frame, text=txt, **kwargs).grid(row=0, column=ci, sticky="w", pady=(2,0))
            else:
                hdr_chk = ctk.CTkCheckBox(self._validate_frame, text="", width=FIXED[2],
                    command=lambda: self._validate_toggle_all(page_chks, hdr_chk))
                hdr_chk.grid(row=0, column=5)
        ctk.CTkFrame(self._validate_frame, height=1, fg_color="#444444").grid(row=1, column=0, columnspan=6, sticky="ew")
        for ri, row in enumerate(page):
            r = ri + 2
            self._validate_frame.grid_rowconfigure(r, minsize=ROW_H)
            kl = ctk.CTkLabel(self._validate_frame, text=row["key"], anchor="w", font=ctk.CTkFont(size=11))
            kl.grid(row=r, column=0, sticky="ew", padx=(4,1))
            kl.bind("<Double-Button-1>", lambda e, idx=row["_idx"]: self._validate_open_editor(idx))
            sl = ctk.CTkEntry(self._validate_frame, font=ctk.CTkFont(size=11), state="disabled")
            sl.grid(row=r, column=1, sticky="ew", padx=(4,1))
            sl.configure(state="normal")
            sl.insert(0, row["src"])
            sl.configure(state="disabled")
            sl.bind("<Double-Button-1>", lambda e, idx=row["_idx"]: self._validate_open_editor(idx))
            tgt_entry = ctk.CTkEntry(self._validate_frame, font=ctk.CTkFont(size=11))
            tgt_entry.grid(row=r, column=2, sticky="ew", padx=(4,1))
            tgt_entry.insert(0, row["tgt"])
            def _on_edit(e, idx=row["_idx"]):
                new_val = e.widget.get()
                ln = self._validate_all_data[idx]["ln"]
                self._validate_modified[ln] = new_val
                self._validate_save_btn.configure(text="Save Changes *")
            tgt_entry.bind("<KeyRelease>", _on_edit)
            tgt_entry.bind("<Double-Button-1>", lambda e, idx=row["_idx"]: self._validate_open_editor(idx))
            sts = row["status"]
            sts_color = "green" if sts == "✓" else "red" if sts.startswith("✗") else "gray"
            ctk.CTkLabel(self._validate_frame, text=sts, width=FIXED[0], anchor="w", font=ctk.CTkFont(size=11), text_color=sts_color).grid(row=r, column=3, padx=(4,1))
            ctk.CTkLabel(self._validate_frame, text=str(row["ln"]), width=FIXED[1], anchor="w", font=ctk.CTkFont(size=11)).grid(row=r, column=4)
            chk = ctk.CTkCheckBox(self._validate_frame, text="", width=FIXED[2], command=lambda idx=row["_idx"]: self._validate_toggle(idx))
            chk.grid(row=r, column=5)
            if sts.startswith("✗"):
                chk.select()
            page_chks.append(chk)
        self._validate_page_label.configure(text=f"{self._validate_page+1}/{pages}")
        self._validate_info.configure(text=f"{total}/{len(self._validate_all_data)} lines  |  ✓ {sum(1 for d in self._validate_all_data if d['status']=='✓')}  ✗ {sum(1 for d in self._validate_all_data if d['status'].startswith('✗'))}")

    def _validate_prev_page(self):
        if self._validate_page > 0:
            self._validate_page -= 1
            self._validate_render()

    def _validate_next_page(self):
        data = self._validate_all_data
        if self._validate_hide_filter.get():
            data = data
        else:
            data = [d for d in data if d["status"] not in ("H", "-")]
        total = len(data)
        per_page = self._validate_per_page.get()
        if per_page < 1:
            per_page = 1
        pages = max(1, (total - 1) // per_page + 1)
        if self._validate_page < pages - 1:
            self._validate_page += 1
            self._validate_render()

    def _validate_toggle_all(self, chks, hdr):
        sel = hdr.get()
        for c in chks:
            c.select() if sel else c.deselect()

    def _validate_open_editor(self, data_idx):
        filtered = [d for d in self._validate_all_data if d["status"] not in ("H", "-")] if not self._validate_hide_filter.get() else self._validate_all_data
        if not filtered:
            return
        start_idx = next((i for i, d in enumerate(filtered) if d is self._validate_all_data[data_idx]), 0)
        popup = ctk.CTkToplevel(self)
        popup.title(f"Line Editor")
        popup.geometry("800x500")
        popup.grid_columnconfigure(0, weight=1)
        popup.grid_rowconfigure(1, weight=1)
        popup.transient(self)

        current = [start_idx]

        def get_row():
            return filtered[current[0]]

        def refresh_header():
            row = get_row()
            popup.title(f"Line {row['ln']}  |  {row['key']}")
            status_label.configure(text=f"Status: {row['status']}")
            page_label.configure(text=f"{current[0]+1}/{len(filtered)}")

        def save_current():
            row = get_row()
            val = target_textbox.get("1.0", "end-1c").strip()
            row["tgt"] = val
            self._validate_modified[row["ln"]] = val
            self._validate_save_btn.configure(text="Save Changes *")

        def load_current():
            row = get_row()
            src_textbox.configure(state="normal")
            src_textbox.delete("1.0", "end")
            src_textbox.insert("1.0", row["src"])
            src_textbox.configure(state="disabled")
            target_textbox.delete("1.0", "end")
            target_textbox.insert("1.0", row["tgt"])
            refresh_header()

        def nav(delta):
            new_idx = current[0] + delta
            if 0 <= new_idx < len(filtered):
                save_current()
                current[0] = new_idx
                load_current()

        def retry_line():
            row = get_row()
            raw_val = row["src"]
            phs = []
            phc = [0]
            def _ph(t):
                ph = f"{{PH{phc[0]}}}"; phc[0] += 1; phs.append((ph, t)); return ph
            cleaned = re.sub(r'\$[^$]+\$', lambda m: _ph(m.group(0)), raw_val)
            cleaned = re.sub(r'\[[^\]]*\]', lambda m: _ph(m.group(0)), cleaned)
            cleaned = re.sub(r'\u00a7.', lambda m: _ph(m.group(0)), cleaned)
            result = self.engine._call_ollama(self.ollama_model.get(),
                f"Translate from {self.source_lang.get()} to {self.target_lang.get()}. Preserve {{PH0}}, {{PH1}} exactly.\n{cleaned}",
                temperature=0.1, max_tokens=500)
            if result and not result.startswith("[OLLAMA_"):
                for ph, orig in phs:
                    result = result.replace(ph, orig)
                result = result.strip().strip('"')
                target_textbox.delete("1.0", "end")
                target_textbox.insert("1.0", result)
                row["tgt"] = result
                row["status"] = "✓"
                save_current()

        def close():
            save_current()
            self._validate_render()
            popup.destroy()

        # Top bar: prev/next + page info + status + retry
        topf = ctk.CTkFrame(popup)
        topf.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        ctk.CTkButton(topf, text="◀ Prev", width=60, command=lambda: nav(-1)).pack(side="left", padx=2)
        page_label = ctk.CTkLabel(topf, text="", font=ctk.CTkFont(size=12))
        page_label.pack(side="left", padx=5)
        ctk.CTkButton(topf, text="Next ▶", width=60, command=lambda: nav(1)).pack(side="left", padx=2)
        status_label = ctk.CTkLabel(topf, text="", font=ctk.CTkFont(size=12))
        status_label.pack(side="right", padx=10)

        # Source / Target panels (50:50)
        midf = ctk.CTkFrame(popup)
        midf.grid(row=1, column=0, sticky="nsew", padx=10, pady=2)
        midf.grid_columnconfigure(0, weight=1)
        midf.grid_columnconfigure(1, weight=1)
        midf.grid_rowconfigure(0, weight=1)

        src_frame = ctk.CTkFrame(midf)
        src_frame.grid(row=0, column=0, sticky="nsew", padx=2)
        src_frame.grid_rowconfigure(1, weight=1)
        src_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(src_frame, text="SOURCE", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, sticky="w", padx=3, pady=2)
        src_textbox = ctk.CTkTextbox(src_frame, wrap="word", font=ctk.CTkFont(size=12))
        src_textbox.grid(row=1, column=0, sticky="nsew", padx=3, pady=2)

        tgt_frame = ctk.CTkFrame(midf)
        tgt_frame.grid(row=0, column=1, sticky="nsew", padx=2)
        tgt_frame.grid_rowconfigure(1, weight=1)
        tgt_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(tgt_frame, text="TARGET (editable)", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, sticky="w", padx=3, pady=2)
        target_textbox = ctk.CTkTextbox(tgt_frame, wrap="word", font=ctk.CTkFont(size=12))
        target_textbox.grid(row=1, column=0, sticky="nsew", padx=3, pady=2)
        target_textbox.bind("<KeyRelease>", lambda e: save_current())

        # Bottom buttons
        bf = ctk.CTkFrame(popup)
        bf.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        retry_btn = ctk.CTkButton(bf, text="Retry with LLM", fg_color="#E65100", command=retry_line,
                                   state="normal" if self._connected else "disabled")
        retry_btn.pack(side="left", padx=5)
        ctk.CTkButton(bf, text="Close", command=close).pack(side="right", padx=5)

        popup.bind("<Escape>", lambda e: close())
        target_textbox.bind("<Escape>", lambda e: close())
        popup.protocol("WM_DELETE_WINDOW", close)

        load_current()

    def _validate_toggle(self, idx):
        pass

    def _auto_load_validate(self):
        self._validate_load_files()

    def _validate_retry_selected(self):
        checked = [d for d in self._validate_all_data if getattr(d, "_checked", False) or d["status"].startswith("✗")]
        if not checked:
            return
        src = self.source_lang.get()
        tgt = self.target_lang.get()
        model = self.ollama_model.get()
        for row in checked:
            if row["status"] in ("✓", "H", "-"):
                continue
            line = row["key"] + ": " + row["src"]
            raw_val = row["src"]
            phs = []
            phc = [0]
            def _ph(t):
                ph = f"{{PH{phc[0]}}}"; phc[0] += 1; phs.append((ph, t)); return ph
            cleaned = re.sub(r'\$[^$]+\$', lambda m: _ph(m.group(0)), raw_val)
            cleaned = re.sub(r'\[[^\]]*\]', lambda m: _ph(m.group(0)), cleaned)
            cleaned = re.sub(r'\u00a7.', lambda m: _ph(m.group(0)), cleaned)
            result = self.engine._call_ollama(model, f"Translate from {src} to {tgt}. Preserve {{PH0}}, {{PH1}} exactly.\n{cleaned}", temperature=0.1, max_tokens=500)
            if result and not result.startswith("[OLLAMA_"):
                for ph, orig in phs:
                    result = result.replace(ph, orig)
                result = result.strip().strip('"')
                row["tgt"] = result
                row["status"] = "✓"
                idx = self._validate_all_data.index(row)
                ln = row["ln"]
                self._validate_modified[ln] = result
                self._validate_save_btn.configure(text="Save Changes *")
        self._validate_render()

    def _validate_save(self):
        if not self._validate_file_pairs or not self._validate_all_data:
            return
        fp = self._validate_file_pairs[0] if self._validate_file_pairs else None
        if not fp:
            return
        try:
            with open(fp["out"], "r", encoding="utf-8-sig") as f:
                lines = f.readlines()
            for ln, new_val in self._validate_modified.items():
                if ln < len(lines):
                    key = self._validate_all_data[ln]["key"]
                    ws = re.match(r"^(\s*)", lines[ln]).group(1) if ln < len(lines) else ""
                    new_line = f'{ws}{key}: "{new_val}"\n'
                    if re.match(r'^\s*[\w.]+:\s*"[^"]*"', lines[ln]):
                        lines[ln] = new_line
            with open(fp["out"], "w", encoding="utf-8-sig") as f:
                f.writelines(lines)
            self._validate_modified.clear()
            self._validate_save_btn.configure(text="Save Changes")
        except Exception as e:
            self.log(f"[ERROR] Save failed: {e}")

    # ── Glossary helpers ──

    @staticmethod
    def _parse_yml(path):
        data = {}
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    m = re.match(r'^\s*([\w.]+):\d*\s*"(.+)"', line)
                    if m:
                        data[m.group(1)] = m.group(2)
        except Exception:
            pass
        return data

    @staticmethod
    def _tokenize_en(val):
        return re.findall(r'[a-zA-Z]+', val.lower())

    @staticmethod
    def _tokenize_tgt(val):
        stopwords = {"그", "이", "저", "것", "수", "등", "에서", "에게", "를", "에", "의", "을", "는", "과", "와", "도", "다", "다."}
        tokens = []
        for m in re.finditer(r'[\uAC00-\uD7AFa-zA-Z0-9]+', val):
            t = m.group()
            if t.endswith('다'):
                continue
            if t.lower() in stopwords:
                continue
            tokens.append(t)
        return tokens

    LANG_PREFIX = {
        "English":"l_english","Korean":"l_korean","Simplified Chinese":"l_simp_chinese",
        "French":"l_french","German":"l_german","Spanish":"l_spanish",
        "Japanese":"l_japanese","Russian":"l_russian","Polish":"l_polish",
        "Brazilian Portuguese":"l_braz_por",
    }
    PREFIX_TO_LANG = {v: k for k, v in LANG_PREFIX.items()}

    PLATFORM_PATTERNS = [
        r'steamapps[\\/]common[\\/]([^\\/]+)',
        r'GOG Games[\\/]([^\\/]+)',
        r'GOG Galaxy[\\/]Games[\\/]([^\\/]+)',
    ]

    def _detect_game_from_path(self, path):
        for pat in self.PLATFORM_PATTERNS:
            m = re.search(pat, path, re.IGNORECASE)
            if m:
                g = m.group(1)
                if g in self.available_games:
                    return g
        return None

    def _g_browse(self):
        d = filedialog.askdirectory()
        if d:
            self._g_folder_var.set(d)
            game = self._detect_game_from_path(d)
            if game:
                self._g_game_var.set(game)
            langs = self._detect_languages(d)
            if langs:
                self._g_src_var.set(langs[0] if langs[0] else "English")
                if len(langs) > 1:
                    self._g_tgt_var.set(langs[1])
                elif langs and langs[0] == "English":
                    self._g_tgt_var.set("Korean")

    def _detect_languages(self, folder):
        langs = set()
        for entry in os.listdir(folder):
            epath = os.path.join(folder, entry)
            if os.path.isdir(epath):
                for code, lang in self.PREFIX_TO_LANG.items():
                    if entry.lower() == code[len("l_"):]:
                        langs.add(lang)
            elif entry.endswith((".yml", ".yaml")):
                for code, lang in self.PREFIX_TO_LANG.items():
                    if entry.lower().startswith(code):
                        langs.add(lang)
        return sorted(langs, key=lambda x: self.available_langs.index(x) if x in self.available_langs else 99)

    def _m_browse(self):
        path = filedialog.askdirectory()
        if not path:
            return
        self._m_folder_var.set(path)
        parent = os.path.basename(os.path.dirname(path.rstrip("\\/")))
        if parent.lower() in ("localisation", "localization"):
            self._m_modname = os.path.basename(os.path.dirname(os.path.dirname(path.rstrip("\\/"))))
        else:
            self._m_modname = parent
        game = self._detect_game_from_path(path)
        if game:
            self._m_game_var.set(game)
        langs = self._detect_languages(path)
        if langs:
            self._m_src_var.set(langs[0] if langs[0] else "English")
            if len(langs) > 1:
                self._m_tgt_var.set(langs[1])
            elif langs and langs[0] == "English":
                self._m_tgt_var.set("Korean")
            if hasattr(self, '_m_lang_combo'):
                self._m_lang_combo.configure(values=self.available_langs)
                self._m_src_combo.configure(values=self.available_langs)

    # ── Glossary GAME tab ──

    def _game_extract(self):
        if getattr(self, '_g_running', False):
            return
        folder = self._g_folder_var.get()
        src_lang = self._g_src_var.get()
        tgt_lang = self._g_tgt_var.get()
        min_co = int(self._g_min_var.get())
        if not folder or not os.path.isdir(folder):
            self.log("[ERROR] Select a folder first")
            return
        src_pre = self.LANG_PREFIX.get(src_lang, f"l_{src_lang.lower()}")
        tgt_pre = self.LANG_PREFIX.get(tgt_lang, f"l_{tgt_lang.lower()}")

        def _run():
            self._g_running = True
            pairs = []
            for root, _, fnames in os.walk(folder):
                src_files = {}
                tgt_files = {}
                for fn in fnames:
                    if fn.startswith(src_pre) and fn.endswith((".yml", ".yaml")):
                        src_files[fn] = os.path.join(root, fn)
                    if fn.startswith(tgt_pre) and fn.endswith((".yml", ".yaml")):
                        tgt_files[fn] = os.path.join(root, fn)
                for sfn, sp in src_files.items():
                    base = sfn[len(src_pre):]
                    tfn = tgt_pre + base
                    if tfn in tgt_files:
                        pairs.append((sp, tgt_files[tfn]))
            if not pairs:
                self.after(0, lambda: self.log("[ERROR] No paired language files found"))
                self._g_running = False
                return
            total_pairs = len(pairs)
            self.after(0, lambda: self.log(f"Found {total_pairs} file pairs. Extracting..."))
            all_entries = []
            for pi, (sp, tp) in enumerate(pairs):
                if getattr(self, '_g_running', False) is False:
                    break
                src_data = self._parse_yml(sp)
                tgt_data = self._parse_yml(tp)
                common = set(src_data.keys()) & set(tgt_data.keys())
                for key in common:
                    en = self._tokenize_en(src_data[key])
                    ko = self._tokenize_tgt(tgt_data[key])
                    if not en or not ko:
                        continue
                    en_set, ko_set = set(en), set(ko)
                    cooccur = len(en_set & ko_set)
                    score = 2 * cooccur / (len(en_set) + len(ko_set)) if (len(en_set) + len(ko_set)) > 0 else 0
                    if cooccur >= min_co:
                        all_entries.append((score, key, src_data[key], tgt_data[key]))
                self.after(0, lambda p=pi+1, t=total_pairs: self._g_progress.configure(
                    progress=p/t if t > 0 else 0) or self._g_info_label.configure(
                    text=f"Scanning {p}/{t}..." if hasattr(self, '_g_info_label') else ""))
            all_entries.sort(key=lambda x: -x[0])
            self._g_all_entries = [(k, s, t) for _, k, s, t in all_entries]
            self._g_page = 0
            self.after(0, lambda: self._game_update_display())
            self.after(0, lambda: self.log(f"Extracted {len(all_entries)} glossary entries"))
            self._g_running = False
        threading.Thread(target=_run, daemon=True).start()

    def _game_update_display(self, filter_text=""):
        for w in self._g_result_frame.winfo_children():
            w.destroy()
        entries = self._g_all_entries if hasattr(self, '_g_all_entries') else []
        if filter_text:
            entries = [e for e in entries if filter_text.lower() in e[1].lower() or filter_text.lower() in e[2].lower()]
        total = len(entries)
        start = self._g_page * self._g_per_page
        end = start + self._g_per_page
        page_entries = entries[start:end]
        for idx, (key, src, tgt) in enumerate(page_entries):
            row = idx
            ctk.CTkLabel(self._g_result_frame, text=f"{key}", anchor="w").grid(row=row, column=0, sticky="w", padx=5)
            ctk.CTkLabel(self._g_result_frame, text=f"{src}", anchor="w").grid(row=row, column=1, sticky="w", padx=5)
            ctk.CTkLabel(self._g_result_frame, text=f"{tgt}", anchor="w").grid(row=row, column=2, sticky="w", padx=5)
        self._g_page_label.configure(text=f"Page {self._g_page+1}/{(total-1)//self._g_per_page+1 if total > 0 else 1} ({total} entries)")
        self._g_info_label.configure(text=f"{total} entries" if total else "No entries")

    def _g_prev_page(self):
        if self._g_page > 0:
            self._g_page -= 1
            self._game_update_display(self._g_search_var.get())

    def _g_next_page(self):
        entries = self._g_all_entries if hasattr(self, '_g_all_entries') else []
        if (self._g_page + 1) * self._g_per_page < len(entries):
            self._g_page += 1
            self._game_update_display(self._g_search_var.get())

    def _game_save(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not path:
            return
        entries = self._g_all_entries if hasattr(self, '_g_all_entries') else []
        try:
            with open(path, "w", encoding="utf-8") as f:
                for _, key, src, tgt in entries:
                    f.write(f"{key}:{src}|{tgt}\n")
            self.log(f"[GLOSSARY] Saved: {path}")
        except Exception as e:
            self.log(f"[ERROR] Save failed: {e}")

    def _game_load(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not path:
            return
        try:
            entries = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    m = re.match(r'^([\w.]+):(.+)\|(.+)$', line.strip())
                    if m:
                        entries.append((m.group(1), m.group(2), m.group(3)))
            self._g_all_entries = [(0, k, s, t) for _, k, s, t in entries]
            self._g_page = 0
            self._game_update_display()
            self.log(f"[GLOSSARY] Loaded {len(entries)} entries")
        except Exception as e:
            self.log(f"[ERROR] Load failed: {e}")

    # ── Glossary MOD tab ──

    def _mod_extract(self):
        if getattr(self, '_m_running', False):
            return
        folder = self._m_folder_var.get()
        min_freq = int(self._m_min_var.get())
        if not folder or not os.path.isdir(folder):
            self.log("[ERROR] Select a folder first")
            return

        def _run():
            self._m_running = True
            en_tokens = {}
            for root, _, fnames in os.walk(folder):
                for fn in fnames:
                    if fn.endswith("l_english.yml"):
                        fp = os.path.join(root, fn)
                        try:
                            with open(fp, "r", encoding="utf-8-sig") as f:
                                for line in f:
                                    m = re.match(r'^\s*([\w.]+):\d*\s*"(.+)"', line)
                                    if m:
                                        for t in self._tokenize_en(m.group(2)):
                                            en_tokens[t] = en_tokens.get(t, 0) + 1
                        except Exception:
                            pass
            self._m_entries = sorted([(k, v) for k, v in en_tokens.items() if v >= min_freq], key=lambda x: -x[1])
            self._m_page = 0
            self.after(0, lambda: self._mod_update_display())
            self.after(0, lambda: self.log(f"Extracted {len(self._m_entries)} English terms"))
            self._m_running = False
        threading.Thread(target=_run, daemon=True).start()

    def _mod_translate(self):
        if getattr(self, '_m_llm_running', False):
            return
        if not hasattr(self, '_m_entries') or not self._m_entries:
            self.log("[ERROR] Extract terms first")
            return
        src_lang = self._m_src_var.get()
        tgt_lang = self._m_tgt_var.get()
        batch_size = 30

        def _run():
            self._m_llm_running = True
            self._m_llm_checked = {}
            pending = [(t, 0) for t, _ in self._m_entries if t not in self._m_llm_checked]
            self.after(0, lambda: self.log(f"Translating {len(pending)} terms ({src_lang} -> {tgt_lang})..."))
            for idx in range(0, len(pending), batch_size):
                batch = pending[idx:idx + batch_size]
                terms = "\n".join(f"{t}" for t, _ in batch)
                prompt = f"Translate these game terms from {src_lang} to {tgt_lang}. Return ONLY 'term:translation' lines.\n\n{terms}"
                r = self.engine._call_ollama(self.ollama_model.get(), prompt, temperature=0.1, max_tokens=2000)
                for line in r.strip().split("\n"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        t = parts[0].strip()
                        tr = parts[1].strip()
                        if t in dict(pending):
                            self._m_llm_checked[t] = tr
                self.after(0, lambda i=idx+len(batch), t=len(pending): self._m_progress.configure(
                    progress=i/t if t > 0 else 0) or self._m_info_label.configure(text=f"Translating {i}/{t}..."))
            self.after(0, lambda: self._mod_update_display())
            self.after(0, lambda: self.log(f"Translated {len(self._m_llm_checked)} terms"))
            self._m_llm_running = False
        threading.Thread(target=_run, daemon=True).start()

    def _mod_update_display(self, filter_text=""):
        for w in self._m_result_frame.winfo_children():
            w.destroy()
        entries = self._m_entries if hasattr(self, '_m_entries') else []
        if filter_text:
            entries = [e for e in entries if filter_text.lower() in e[0].lower()]
        total = len(entries)
        start = self._m_page * self._m_per_page
        end = start + self._m_per_page
        page_entries = entries[start:end]
        checked = getattr(self, '_m_llm_checked', {})
        for idx, (term, freq) in enumerate(page_entries):
            row = idx
            ctk.CTkCheckBox(self._m_result_frame, text="").grid(row=row, column=0, padx=2)
            ctk.CTkLabel(self._m_result_frame, text=f"{term} ({freq})", anchor="w").grid(row=row, column=1, sticky="w", padx=5)
            trans = checked.get(term, "")
            e = ctk.CTkEntry(self._m_result_frame, width=300)
            e.grid(row=row, column=2, sticky="ew", padx=5)
            if trans:
                e.insert(0, trans)
        self._m_page_label.configure(text=f"Page {self._m_page+1}/{(total-1)//self._m_per_page+1 if total > 0 else 1} ({total} terms)")

    def _mod_retry_selected(self):
        pass

    def _m_prev_page(self):
        if self._m_page > 0:
            self._m_page -= 1
            self._mod_update_display(self._m_search_var.get())

    def _m_next_page(self):
        entries = self._m_entries if hasattr(self, '_m_entries') else []
        if (self._m_page + 1) * self._m_per_page < len(entries):
            self._m_page += 1
            self._mod_update_display(self._m_search_var.get())

    def _mod_save(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not path:
            return
        checked = getattr(self, '_m_llm_checked', {})
        try:
            with open(path, "w", encoding="utf-8") as f:
                for term, trans in checked.items():
                    if trans:
                        f.write(f"{term}:{trans}\n")
            self.log(f"[GLOSSARY] Saved: {path}")
        except Exception as e:
            self.log(f"[ERROR] Save failed: {e}")

    def _mod_load(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not path:
            return
        try:
            checked = getattr(self, '_m_llm_checked', {})
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(":", 1)
                    if len(parts) == 2:
                        checked[parts[0].strip()] = parts[1].strip()
            self._m_llm_checked = checked
            self._mod_update_display()
            self.log(f"[GLOSSARY] Loaded from: {path}")
        except Exception as e:
            self.log(f"[ERROR] Load failed: {e}")

    # ── Folders and Logging ──

    def _browse_input(self):
        d = filedialog.askdirectory()
        if d:
            _codes = {"English":"english","Korean":"korean","Simplified Chinese":"simp_chinese",
                      "French":"french","German":"german","Spanish":"spanish",
                      "Japanese":"japanese","Russian":"russian","Polish":"polish",
                      "Brazilian Portuguese":"braz_por"}
            src_code = _codes.get(self.source_lang.get(), "").lower()
            tgt_code = _codes.get(self.target_lang.get(), "").lower()
            for sub in ("localisation", "localization"):
                p = os.path.join(d, sub)
                if os.path.isdir(p):
                    d = p
                    break
            sf = os.path.join(d, src_code)
            if os.path.isdir(sf):
                d = sf
            self.input_dir.set(d)
            parent = os.path.dirname(d)
            folder = os.path.basename(d)
            if folder.lower() == src_code and src_code:
                tgt_path = os.path.join(parent, tgt_code)
                os.makedirs(tgt_path, exist_ok=True)
                self.output_dir.set(tgt_path)
            else:
                self.output_dir.set(d)
            if hasattr(self, 'output_entry'):
                self.output_entry.configure(text_color="#888888")
            self._save_config()

    def _browse_output(self):
        d = filedialog.askdirectory()
        if d:
            self.output_dir.set(d)
            if hasattr(self, 'output_entry'):
                self.output_entry.configure(text_color="white")

    def _log_dir(self):
        d = os.path.join(os.path.dirname(self._config_path()), "rog")
        os.makedirs(d, exist_ok=True)
        return d

    def _init_log_file(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._current_log_path = os.path.join(self._log_dir(), f"log_{ts}.txt")
        try:
            with open(self._current_log_path, "w", encoding="utf-8") as f:
                f.write(f"=== OllamaTranslator Log ({ts}) ===\n\n")
            logs = sorted([os.path.join(self._log_dir(), f) for f in os.listdir(self._log_dir())
                          if f.startswith("log_") and f.endswith(".txt")], reverse=True)
            for old in logs[3:]:
                os.remove(old)
        except Exception:
            pass

    def log(self, msg):
        ts = time.strftime("[%H:%M:%S]")
        full = f"{ts} {msg}"
        try:
            self.log_text.insert("end", full + "\n")
            self.log_text.see("end")
        except Exception:
            pass
        try:
            with open(self._current_log_path, "a", encoding="utf-8") as f:
                f.write(full + "\n")
        except Exception:
            pass

    def update_progress(self, current, total):
        def _u():
            self.progress_bar.set(min(current / total, 1.0) if total > 0 else 0)
            self.progress_text.configure(text=f"{current} / {total} lines")
        self.after(0, _u)

    def set_status(self, status):
        def _u():
            self.status_label.configure(text=status)
        self.after(0, _u)

    # ── Ollama Connection ──

    def _start_ollama(self):
        def _run():
            self.log("[OLLAMA] Starting server...")
            models = self.engine.start_server()
            if models:
                self.log(f"[OLLAMA] Models: {', '.join(models)}")
                self.ollama_combo.configure(values=models)
                if self.ollama_model.get() not in models:
                    self.ollama_model.set(models[0] if models else "")
            else:
                self.log("[OLLAMA] Failed to start server")
        threading.Thread(target=_run, daemon=True).start()

    def _connect_ollama(self):
        url = self.ollama_url.get().strip()
        if not url:
            self.log("[CONNECT] Enter Ollama URL first")
            return

        def _run():
            self.log("[CONNECT] Connecting...")
            self.engine.set_base_url(url)
            models = self.engine.fetch_models()
            if models:
                self.log(f"[CONNECT] Models: {', '.join(models)}")
                self.ollama_combo.configure(values=models)
                if self.ollama_model.get() not in models:
                    self.ollama_model.set(models[0] if models else "")
            else:
                self.log("[CONNECT] Cannot connect to Ollama")
                self._connected = False
                self._validate_fields()
                return
            model = self.ollama_model.get()
            if not model:
                self.log("[CONNECT] Select a model first")
                return
            self.log(f"[CONNECT] Loading model {model}...")
            tgt = self.target_lang.get() or "Korean"
            test = self.engine.test_model(model, tgt, self.selected_game.get())
            if test and not test.startswith("[OLLAMA_"):
                self._connected = True
                self.log(f"[CONNECT] Translator Ready")
            else:
                self._connected = False
                self.log(f"[CONNECT] Model test failed: {test}")
            self._validate_fields()

        threading.Thread(target=_run, daemon=True).start()

    # ── Prompt Management ──

    def _toggle_prompt(self):
        self.prompt_frame.grid() if self.show_prompt.get() else self.prompt_frame.grid_remove()

    def _sync_prompt(self, event=None):
        self.prompt_template_var.set(self.prompt_textbox.get("1.0", "end-1c"))

    def _restore_default_prompt(self):
        d = self._default_prompt()
        self.prompt_template_var.set(d)
        self.prompt_textbox.delete("1.0", "end")
        self.prompt_textbox.insert("1.0", d)

    def _load_prompt_from_file(self):
        fp = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if fp:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    c = f.read()
                self.prompt_template_var.set(c)
                self.prompt_textbox.delete("1.0", "end")
                self.prompt_textbox.insert("1.0", c)
            except Exception as e:
                self.log(f"[ERROR] Failed to load prompt: {e}")

    # ── Translation Controls ──

    def _start(self):
        self._sync_prompt()
        self.engine.prompt_template = self.prompt_template_var.get()
        self.engine.checkpoint_enabled = self.checkpoint_enabled.get()
        self.engine.debug_mode = self.debug_mode.get()
        if self.checkpoint_enabled.get():
            cp_dir = os.path.join(os.path.dirname(CONFIG_FILE), "checkpoint")
            if os.path.isdir(cp_dir):
                for fn in os.listdir(cp_dir):
                    if not fn.endswith((".yml", ".yaml")):
                        continue
                    cp = os.path.join(cp_dir, fn)
                    if not os.path.isfile(cp):
                        continue
                    for root, _, fnames in os.walk(self.output_dir.get()):
                        for fn2 in fnames:
                            if fn2.lower() == fn.lower():
                                with open(cp, "r", encoding="utf-8-sig") as f:
                                    data = f.read()
                                with open(os.path.join(root, fn2), "w", encoding="utf-8-sig") as f:
                                    f.write(data)
                                self.log(f"[RESUME] Restored checkpoint: {fn}")
                                break
        self.engine.set_base_url(self.ollama_url.get())
        self._sync_prompt()
        self.engine.prompt_template = self.prompt_template_var.get()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.reset_btn.configure(state="disabled")
        self.log_text.delete("1.0", "end")
        self.log("[START] Translation started")
        self.engine.start(self.input_dir.get(), self.output_dir.get(),
                          self.source_lang.get(), self.target_lang.get(),
                          self.ollama_model.get(), self.temperature.get(),
                          self.max_tokens.get(), self.batch_size.get(),
                          self.selected_game.get(), self.max_retries.get())
        self.after(500, self._poll_done)

    _poll_count = 0

    def _poll_done(self):
        if self.engine.busy:
            self._poll_count += 1
            if self._poll_count > 1200:
                self.log("[STOP] Translation timed out")
                self.engine.stop()
                self._reset_ui()
                return
            self.after(500, self._poll_done)
        else:
            self._done()

    def _done(self):
        self.engine.busy = False
        self._poll_count = 0
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.reset_btn.configure(state="normal")
        self.log("[DONE] Translation complete")
        self._auto_load_validate()

    def _stop(self):
        self.log("[STOP] Stopping translation...")
        self.engine.stop()

    def _reset_ui(self):
        self.log_text.delete("1.0", "end")
        self.progress_bar.set(0)
        self.progress_text.configure(text="0 / 0 lines")
        self.start_btn.configure(state="normal" if self._connected else "disabled")
        self.stop_btn.configure(state="disabled")
        self.reset_btn.configure(state="disabled")

    def _on_live_result(self, originals, translated):
        def _u():
            self.live_orig.delete("1.0", "end")
            self.live_trans.delete("1.0", "end")
            for o, t in zip(originals, translated):
                self.live_orig.insert("end", o.rstrip("\n") + "\n")
                self.live_trans.insert("end", t.rstrip("\n") + "\n")
            self.live_orig.see("end")
            self.live_trans.see("end")
        self.after(0, _u)

    def _toggle_live(self):
        if self.live_visible.get():
            self.live_frame.grid_remove()
            self.live_visible.set(False)
        else:
            self.live_frame.grid()
            self.live_visible.set(True)

def main():
    app = OllamaTranslatorGUI()
    app.mainloop()

if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ollama.pdx.translator")
        except AttributeError:
            pass
    main()
