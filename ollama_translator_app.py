# ============================================================
# Ollama Paradox Mod Translator
# ============================================================

import os, sys, json, time, re, codecs, threading, concurrent.futures
import requests, customtkinter as ctk
from tkinter import filedialog, messagebox

# ============================================================
# 경로 설정
# ============================================================
def _app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(_app_dir(), "ollama_translator_config.json")
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

# ============================================================
# 게임별 프롬프트
# ============================================================
GAME_PROMPTS = {
    "Crusader Kings 3": """You are translating text from the medieval grand strategy game 'Crusader Kings 3'.
Use a majestic, epic tone appropriate for medieval nobility and court intrigue.
CRITICAL: Elements enclosed in square brackets like [Concept|E], [Character.GetFirstName] are game code - DO NOT translate or modify them.
Maintain the medieval atmosphere while ensuring natural translation.""",
    "Hearts of Iron 4": """You are translating text from the WWII grand strategy game 'Hearts of Iron 4'.
Use a concise, military report style with professional terminology.
CRITICAL: Symbols starting with \u00a3 (like \u00a3GFX_army_experience, \u00a3pol_power) are icon codes - NEVER translate them.
Keep military terms precise and formal.""",
    "Stellaris": """You are translating text from the sci-fi grand strategy game 'Stellaris'.
Use futuristic, scientific terminology and a tone suitable for space exploration and diplomacy.
CRITICAL: Preserve all bracketed codes like [species.GetName] and variables like $PLANET_NAME$ exactly as they appear.""",
    "Europa Universalis IV": """You are translating text from the historical grand strategy game 'Europa Universalis IV' (1444-1821 period).
Use formal diplomatic language appropriate for the Early Modern period.
CRITICAL: Preserve all game codes in brackets [] and variables with $ symbols.""",
    "Victoria 3": """You are translating text from the industrial era grand strategy game 'Victoria 3' (19th century).
Use terminology appropriate for the Industrial Revolution era, including political movements, economic systems, and social reforms.""",
    "Imperator: Rome": """You are translating text from the ancient grand strategy game 'Imperator: Rome'.
Use classical, dignified language appropriate for the Roman Republic period.
CRITICAL: Do not translate any game codes within brackets or special markers."""
}

def get_enhanced_prompt(game_name, base_prompt):
    if game_name in GAME_PROMPTS:
        return f"""[GAME CONTEXT]\n{GAME_PROMPTS[game_name]}\n\n[GENERAL INSTRUCTIONS]\n{base_prompt}"""
    return base_prompt

# ============================================================
# OllamaTranslator : 번역 엔진
# ============================================================
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
        self.retry_untranslated = False
        self.max_retries = 3

    def set_base_url(self, url):
        self.base_url = url.rstrip("/")

    def _call_ollama(self, model, prompt, temperature=0.5, max_tokens=4096):
        try:
            resp = requests.post(f"{self.base_url}/api/chat", json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "stream": False
            }, timeout=120)
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            return "[OLLAMA_CONNECTION_ERROR]"
        except Exception as e:
            return f"[OLLAMA_ERROR: {e}]"

    def _save_checkpoint(self, result_lines, output_path):
        cp_path = os.path.join(os.path.dirname(CONFIG_FILE), os.path.basename(output_path))
        try:
            with codecs.open(cp_path, "w", encoding="utf-8-sig") as f:
                f.writelines(result_lines)
            self.log_callback(f"[CHECKPOINT] Saved: {cp_path}")
        except Exception as e:
            self.log_callback(f"[CHECKPOINT] Failed: {e}")

    def _extract_yml_value(self, line):
        m = re.match(r'^[^#]*?:\s*"(.+)"', line)
        return m.group(1) if m else None

    def _has_source_language(self, line, source_lang):
        val = self._extract_yml_value(line)
        if not val:
            return False
        patterns = {
            "English": r"[a-zA-Z]{4,}", "Korean": r"[\uAC00-\uD7AF]+",
            "Japanese": r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+",
            "Simplified Chinese": r"[\u4E00-\u9FFF]+",
            "French": r"[a-zA-Z\u00C0-\u017F]{4,}", "German": r"[a-zA-Z\u00C0-\u017F]{4,}",
            "Spanish": r"[a-zA-Z\u00C0-\u017F]{4,}", "Russian": r"[\u0400-\u04FF]+",
        }
        pat = patterns.get(source_lang, r"[a-zA-Z]{4,}")
        return bool(re.search(pat, val))

    def _translate_batch(self, lines, source_lang, target_lang, model, temperature, max_tokens, game="None", retry_count=0, src_filename=None):
        comment_indices = {i for i, l in enumerate(lines) if not l.strip() or l.strip().startswith('#')}
        if comment_indices:
            if len(comment_indices) == len(lines):
                return lines
            actual_lines = [l for i, l in enumerate(lines) if i not in comment_indices]
            translated_actual = self._translate_batch(actual_lines, source_lang, target_lang, model, temperature, max_tokens, game, 0, src_filename)
            if self.stop_event.is_set():
                return lines
            merged = []
            ai = 0
            for i in range(len(lines)):
                merged.append(lines[i] if i in comment_indices else translated_actual[ai])
                ai += 0 if i in comment_indices else 1
            return merged

        batch_text = "\n".join(line.rstrip("\n") for line in lines)
        if self.prompt_template:
            base_prompt = self.prompt_template.replace("{source_lang}", source_lang)
            base_prompt = base_prompt.replace("{target_lang}", target_lang)
            base_prompt = base_prompt.replace("{batch_text}", batch_text)
        else:
            base_prompt = (
                f"Translate the following YAML text from '{source_lang}' to '{target_lang}'.\n"
                f"Rules:\n1. Only translate text after ': ' in double quotes.\n"
                f"2. Do NOT translate $variables$, [brackets], \u00a7X color codes, or file paths.\n"
                f"3. Preserve all \\\\n and leading whitespace.\n"
                f"4. Keep lines like 'l_english:' or comments (#) unchanged.\n"
                f"5. Output EXACTLY one line per input line.\n"
                f"6. Do NOT wrap in code blocks or add explanations.\n\n{batch_text}")
        prompt = get_enhanced_prompt(game, base_prompt)
        result = self._call_ollama(model, prompt, temperature, max_tokens)

        if result.startswith("[OLLAMA_"):
            if len(lines) > 1:
                self.log_callback(f"[SPLIT] {result} - splitting batch of {len(lines)} lines")
                mid = len(lines) // 2
                t = min(temperature + 0.05, 1.0)
                first = self._translate_batch(lines[:mid], source_lang, target_lang, model, t, max_tokens, game, src_filename=src_filename)
                if self.stop_event.is_set():
                    return lines
                second = self._translate_batch(lines[mid:], source_lang, target_lang, model, t, max_tokens, game, src_filename=src_filename)
                return first + second
            else:
                if retry_count < self.max_retries:
                    self.log_callback(f"[RETRY {retry_count+1}/{self.max_retries}] {result} - retrying single line")
                    t = min(temperature + 0.1, 1.0)
                    return self._translate_batch(lines, source_lang, target_lang, model, t, max_tokens, game, retry_count + 1, src_filename)
                self.log_callback(f"[FAIL] {result} - returning original after {self.max_retries} retries")
                return lines

        result = result.strip()
        result = re.sub(r"```(?:yaml|yml)?\s*\n?", "", result, flags=re.IGNORECASE)
        result = re.sub(r"\n?```", "", result)
        translated = result.split("\n")

        if len(translated) != len(lines):
            if retry_count < self.max_retries:
                self.log_callback(f"[RETRY {retry_count+1}/{self.max_retries}] Line count mismatch ({len(translated)} vs {len(lines)})")
                t = min(temperature + 0.1, 1.0)
                return self._translate_batch(lines, source_lang, target_lang, model, t, max_tokens, game, retry_count + 1, src_filename)
            self.log_callback(f"[WARN] Line count mismatch ({len(translated)} vs {len(lines)}), keeping original")
            return lines

        if self.live_callback:
            self.live_callback(lines, translated)

        if self.retry_untranslated and source_lang != target_lang:
            untranslated_indices = []
            for i, (orig, trans) in enumerate(zip(lines, translated)):
                if self._has_source_language(orig, source_lang) and not self._has_source_language(trans, target_lang):
                    if self._has_source_language(trans, source_lang):
                        untranslated_indices.append(i)
            if untranslated_indices:
                self.log_callback(f"[RETRY] Re-translating {len(untranslated_indices)} untranslated line(s)")
                retry_lines = [lines[i] for i in untranslated_indices]
                retry_text = "\n".join(line.rstrip("\n") for line in retry_lines)
                retry_prompt = base_prompt.replace(batch_text, retry_text) if len(retry_lines) != len(lines) else base_prompt
                retry_prompt = get_enhanced_prompt(game, retry_prompt)
                retry_result = self._call_ollama(model, retry_prompt, min(temperature + 0.1, 1.0), max_tokens)
                if not retry_result.startswith("[OLLAMA_"):
                    retry_result = re.sub(r"```(?:yaml|yml)?\n?", "", retry_result, flags=re.IGNORECASE)
                    retry_result = re.sub(r"\n?```", "", retry_result)
                    retry_lines_result = retry_result.split("\n")
                    if len(retry_lines_result) == len(retry_lines):
                        for j, idx in enumerate(untranslated_indices):
                            translated[idx] = retry_lines_result[j]
        return translated

    def _process_file(self, input_path, output_path, source_lang, target_lang, model, temperature, max_tokens, batch_size, game="None"):
        with codecs.open(input_path, "r", encoding="utf-8-sig") as f:
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
            result.append(f"l_{target_code}:{first[first.index(':'):]}\n")
            content = lines[1:]
        else:
            result.append(first)
            content = lines[1:]

        resume_from = 0
        if os.path.exists(output_path):
            try:
                with codecs.open(output_path, "r", encoding="utf-8-sig") as f:
                    existing = f.readlines()
                if len(existing) > 1:
                    result = existing
                    resume_from = len(existing) - 1
                    self.log_callback(f"  Resuming from line {resume_from + 1}")
            except Exception:
                pass

        for i in range(resume_from, len(content), batch_size):
            if self.stop_event.is_set():
                self.log_callback("[STOPPED] Translation interrupted")
                if len(result) > 1:
                    self._save_checkpoint(result, output_path)
                return
            batch = content[i:i + batch_size]
            self.log_callback(f"  Translating lines {len(result)+1}-{len(result)+len(batch)}/{total}")
            translated = self._translate_batch(batch, source_lang, target_lang, model, temperature, max_tokens, game, src_filename=os.path.basename(input_path))
            result.extend(translated)
            self.progress_callback(min(len(result), total), total)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with codecs.open(output_path, "w", encoding="utf-8-sig") as f:
            f.writelines(result)
        self.log_callback(f"  Saved: {output_path}")
        cp_path = os.path.join(os.path.dirname(CONFIG_FILE), os.path.basename(output_path))
        if os.path.exists(cp_path):
            try:
                os.remove(cp_path)
            except Exception:
                pass

    def _worker(self, input_dir, output_dir, source_lang, target_lang, model, temperature, max_tokens, batch_size, game="None", max_workers=3):
        _codes = {"English": "english", "Korean": "korean", "Simplified Chinese": "simp_chinese",
                  "French": "french", "German": "german", "Spanish": "spanish", "Japanese": "japanese",
                  "Russian": "russian", "Polish": "polish", "Brazilian Portuguese": "braz_por"}
        source_code = _codes.get(source_lang, source_lang.lower())
        target_code = _codes.get(target_lang, target_lang.lower())

        files = []
        for root, _, fnames in os.walk(input_dir):
            for fn in fnames:
                if f"l_{source_code}" in fn.lower() and fn.lower().endswith((".yml", ".yaml")):
                    files.append(os.path.join(root, fn))
        if not files:
            self.log_callback(f"No files found with language identifier 'l_{source_code}'")
            self.status_callback("idle")
            return

        self.log_callback(f"Found {len(files)} files. Starting translation...")
        self.status_callback("translating")

        test = self._call_ollama(model, "test", temperature=0.1, max_tokens=1)
        if test.startswith("[OLLAMA_"):
            self.log_callback(f"[ERROR] Cannot connect to Ollama at {self.base_url}")
            self.status_callback("idle")
            return
        self.log_callback(f"Ollama connected. Using model: {model}")

        def _translate_one(fp):
            if self.stop_event.is_set():
                return
            rel = os.path.relpath(os.path.dirname(fp), input_dir)
            base = os.path.basename(fp)
            new_base = re.sub(f"l_{source_code}", f"l_{target_code}", base, flags=re.IGNORECASE)
            out_path = os.path.join(output_dir, rel, new_base)
            self.log_callback(f"  Processing: {os.path.basename(fp)}")
            self._process_file(fp, out_path, source_lang, target_lang, model, temperature, max_tokens, batch_size, game)

        n = min(max_workers, len(files))
        stopped = False
        if n <= 1:
            for fp in files:
                if self.stop_event.is_set():
                    stopped = True
                    break
                _translate_one(fp)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as exc:
                concurrent.futures.wait([exc.submit(_translate_one, fp) for fp in files])
                if self.stop_event.is_set():
                    stopped = True

        if stopped:
            self.log_callback("[STOPPED] Translation stopped. Checkpoints saved")
        else:
            self.log_callback("All done!")
            # 품질 검사 요약
            for fp in files:
                base = os.path.basename(fp)
                new_base = re.sub(f"l_{source_code}", f"l_{target_code}", base, flags=re.IGNORECASE)
                for root, _, fnames in os.walk(output_dir):
                    for fn2 in fnames:
                        if fn2.lower() == new_base.lower():
                            out_path = os.path.join(root, fn2)
                            issues = self.check_quality(fp, out_path, source_lang, target_lang)
                            if issues:
                                counts = {"UNTRANSLATED": 0, "FOREIGN": 0, "DUPLICATE": 0}
                                for _, _, _, typ, _ in issues:
                                    if typ in counts:
                                        counts[typ] += 1
                                parts = [f"{k.lower()} {v}" for k, v in counts.items() if v > 0]
                                self.log_callback(f"  [VALIDATE] {new_base}: {', '.join(parts)}")
                            break
        self.status_callback("idle")

    # ============================================================
    # Validate : 번역 품질 검사
    # ============================================================
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
            with codecs.open(input_path, "r", encoding="utf-8-sig") as f:
                src_lines = [l.rstrip("\n") for l in f.readlines()]
            with codecs.open(output_path, "r", encoding="utf-8-sig") as f:
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
            if not re.search(r'[가-힣a-zA-Z\u00C0-\u024F\u4E00-\u9FFF\uAC00-\uD7AF\u3040-\u30FF\u0400-\u04FF]', ct):
                continue
            dup_info = f"key '{key}' dup at {dups[key]}" if key in dups else ""
            if cs == ct:
                issues.append((i, s, t, "UNTRANSLATED", dup_info))
            elif self._has_foreign_chars(t_val, target_lang):
                issues.append((i, s, t, "FOREIGN", dup_info))
            elif dup_info:
                issues.append((i, s, t, "DUPLICATE", dup_info))
        return issues

    def start(self, input_dir, output_dir, source_lang, target_lang, model, temperature, max_tokens, batch_size, game="None", max_workers=3, retry_untranslated=False, max_retries=3):
        self.stop_event.clear()
        self.retry_untranslated = retry_untranslated
        self.max_retries = max_retries
        self.thread = threading.Thread(target=self._worker, args=(
            input_dir, output_dir, source_lang, target_lang, model, temperature, max_tokens, batch_size, game, max_workers), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

# ============================================================
# OllamaTranslatorGUI
# ============================================================
class OllamaTranslatorGUI(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Ollama PDX Translator")
        self.geometry("800x700")
        self.stop_event = threading.Event()

        self.ollama_url = ctk.StringVar()
        self.input_dir = ctk.StringVar()
        self.output_dir = ctk.StringVar()
        self.source_lang = ctk.StringVar()
        self.target_lang = ctk.StringVar()
        self.ollama_model = ctk.StringVar()
        self.temperature = ctk.DoubleVar(value=0.2)
        self.max_tokens = ctk.IntVar(value=8192)
        self.batch_size = ctk.IntVar(value=1)
        self.max_retries = ctk.IntVar(value=3)

        self.selected_game = ctk.StringVar()
        self.available_games = list(GAME_PROMPTS.keys())
        self.show_prompt = ctk.BooleanVar(value=False)
        self.prompt_template_var = ctk.StringVar(value=self._default_prompt())
        self.live_visible = ctk.BooleanVar(value=False)
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

    # ============================================================
    # Detect
    # ============================================================
    def _auto_detect_ollama(self):
        url = self.ollama_url.get().rstrip("/") or "http://localhost:11434"
        try:
            resp = requests.get(f"{url}/api/tags", timeout=3)
            if resp.status_code == 200:
                names = [m["name"] for m in resp.json().get("models", [])]
                self.ollama_url.set(url)
                self.log(f"[DETECT] Ollama connected at {url}")
                if names:
                    self.log(f"[DETECT] Models: {', '.join(names[:8])}{'...' if len(names)>8 else ''}")
                    self.ollama_model.set(names[0])
                else:
                    self.log("[DETECT] No models found")
            else:
                self.log("[DETECT] Unexpected status")
        except requests.exceptions.ConnectionError:
            self.log(f"[DETECT] Cannot connect to {url}")
        except Exception as e:
            self.log(f"[DETECT] Failed: {e}")

    def _config_path(self):
        return CONFIG_FILE

    # ============================================================
    # 설정 로드/저장
    # ============================================================
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
            self.max_retries.set(cfg.get("max_retries", cfg.get("max_workers", 3)))
            p = cfg.get("prompt_template", "")
            if p:
                self.prompt_template_var.set(p)
                self.prompt_textbox.delete("1.0", "end")
                self.prompt_textbox.insert("1.0", p)
            if cfg.get("show_prompt", False):
                self.show_prompt.set(True)
                self.prompt_frame.grid()
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
            "show_prompt": self.show_prompt.get()
        }
        try:
            with open(self._config_path(), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _on_close(self):
        self._save_config()
        self.destroy()

    def _validate_fields(self, *args):
        ok = all([self.ollama_url.get(), self.ollama_model.get(),
                  self.input_dir.get(), self.output_dir.get(),
                  self.source_lang.get(), self.target_lang.get()])
        self.start_btn.configure(state="normal" if ok else "disabled")

    def _default_prompt(self):
        return ("Translate the following YAML text from '{source_lang}' to '{target_lang}'.\n"
                "Rules:\n1. Only translate text after ': ' in double quotes.\n"
                "2. Do NOT translate $variables$, [brackets], \u00a7X color codes, or file paths.\n"
                "3. Preserve all \\n and leading whitespace.\n"
                "4. Keep lines like 'l_english:' or comments (#) unchanged.\n"
                "5. Output EXACTLY one line per input line.\n"
                "6. Do NOT wrap in code blocks or add explanations.\n\n{batch_text}")

    # ============================================================
    # UI 구성
    # ============================================================
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, sticky="nsew")

        t_trans = self.tabview.add("Translate")
        t_val = self.tabview.add("Validate")
        self.tabview.set("Translate")

        # ========== Translate Tab ==========
        t_trans.grid_columnconfigure(0, weight=1)
        t_trans.grid_rowconfigure(7, weight=1)

        pf = ctk.CTkFrame(t_trans)
        pf.grid(row=5, column=0, padx=10, pady=0, sticky="ew")
        pf.grid_columnconfigure(0, weight=1)
        self.progress_bar = ctk.CTkProgressBar(pf)
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.progress_bar.set(0)
        self.progress_text = ctk.CTkLabel(pf, text="0 / 0 lines")
        self.progress_text.grid(row=0, column=1, padx=5)

        # --- Live Output ---
        self.live_frame = ctk.CTkFrame(t_trans, height=200)
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

        # --- Log ---
        self.log_frame = ctk.CTkFrame(t_trans)
        self.log_frame.grid(row=7, column=0, padx=10, pady=0, sticky="nsew")
        self.log_frame.grid_columnconfigure(0, weight=1)
        self.log_frame.grid_rowconfigure(0, weight=1)
        self.log_text = ctk.CTkTextbox(self.log_frame, wrap="word", font=ctk.CTkFont(size=11))
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)

        # --- Title ---
        ctk.CTkLabel(t_trans, text="Ollama Paradox Mod Translator",
                      font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, pady=(6, 2), sticky="n")

        # --- Settings ---
        sf = ctk.CTkFrame(t_trans)
        sf.grid(row=1, column=0, padx=10, pady=4, sticky="ew")
        for c in range(4):
            sf.grid_columnconfigure(c, weight=[0, 1, 0, 3][c])

        def _put(row, col, var, combo_vals=None, browse=None, extra_btns=None):
            f = ctk.CTkFrame(sf, fg_color="transparent")
            f.grid(row=row, column=col, sticky="ew", padx=(0, 10) if col == 1 else (0, 5), pady=3)
            f.grid_columnconfigure(0, weight=1)
            if combo_vals:
                ctk.CTkComboBox(f, variable=var, values=combo_vals, state="readonly").grid(row=0, column=0, sticky="ew")
            else:
                ctk.CTkEntry(f, textvariable=var).grid(row=0, column=0, sticky="ew")
            bc = 2
            if browse:
                ctk.CTkButton(f, text="Browse", width=70, command=browse).grid(row=0, column=bc, padx=(5, 0))
                bc += 1
            if extra_btns:
                for lbl, cmd in extra_btns:
                    ctk.CTkButton(f, text=lbl, width=70, command=cmd).grid(row=0, column=bc, padx=(5, 0))
                    bc += 1

        def _lb(row, col, text):
            ctk.CTkLabel(sf, text=text, anchor="w").grid(row=row, column=col, sticky="w", padx=5, pady=3)

        _lb(0, 0, "Model:")
        _put(0, 1, self.ollama_model)
        _lb(0, 2, "Ollama URL:")
        _put(0, 3, self.ollama_url, extra_btns=[("Detect", self._auto_detect_ollama), ("Test", self._test_connection)])
        _lb(1, 0, "Source:")
        _put(1, 1, self.source_lang, combo_vals=self.available_langs)
        _lb(1, 2, "Input Folder:")
        _put(1, 3, self.input_dir, browse=self._browse_input)
        _lb(2, 0, "Target:")
        _put(2, 1, self.target_lang, combo_vals=self.available_langs)
        _lb(2, 2, "Output Folder:")
        _put(2, 3, self.output_dir, browse=self._browse_output)

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

        # --- Prompt ---
        ctk.CTkCheckBox(t_trans, text="Edit Prompt (sent to AI)", variable=self.show_prompt,
                        font=ctk.CTkFont(size=12), command=self._toggle_prompt).grid(row=2, column=0, padx=10, pady=0, sticky="w")
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

        # --- Buttons ---
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

        # ========== Validate Tab ==========
        t_val.grid_columnconfigure(0, weight=1)
        t_val.grid_rowconfigure(2, weight=1)

        ctk.CTkButton(t_val, text="Scan Output Files", command=self._run_validate,
                      fg_color="#1565C0").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.val_status = ctk.CTkLabel(t_val, text="", font=ctk.CTkFont(size=11))
        self.val_status.grid(row=1, column=0, padx=10, pady=2, sticky="w")
        self.val_text = ctk.CTkTextbox(t_val, wrap="word", font=ctk.CTkFont(size=11))
        self.val_text.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")

    # ============================================================
    # Validate 실행
    # ============================================================
    def _run_validate(self):
        inp = self.input_dir.get()
        out = self.output_dir.get()
        if not inp or not out:
            self.val_status.configure(text="Set Input and Output folders first")
            return
        src = self.source_lang.get()
        tgt = self.target_lang.get()
        if not src or not tgt:
            self.val_status.configure(text="Set Source and Target languages first")
            return
        self.val_text.delete("1.0", "end")
        total_issues = 0
        matched = 0
        for root, _, fnames in os.walk(out):
            for fn in fnames:
                if not fn.endswith((".yml", ".yaml")):
                    continue
                out_path = os.path.join(root, fn)
                rel = os.path.relpath(root, out)
                src_code = {"English":"english","Korean":"korean","Simplified Chinese":"simp_chinese","French":"french","German":"german","Spanish":"spanish","Japanese":"japanese","Russian":"russian","Polish":"polish","Brazilian Portuguese":"braz_por"}.get(src, src.lower())
                in_path = os.path.join(inp, rel, re.sub(r'^l_[a-z]+_', f'l_{src_code}_', fn, flags=re.IGNORECASE))
                if not os.path.exists(in_path):
                    continue
                matched += 1
                issues = self.engine.check_quality(in_path, out_path, src, tgt)
                if not issues:
                    continue
                total_issues += len(issues)
                self.val_text.insert("end", f"\n--- {fn} ({len(issues)} issues) ---\n")
                for line_num, orig, trans, typ, dup in issues:
                    tag = {"UNTRANSLATED": "!", "FOREIGN": "?", "DUPLICATE": "D", "MISMATCH": "X"}.get(typ, "?")
                    self.val_text.insert("end", f"  [{tag}] L{line_num}\n")
                    self.val_text.insert("end", f"       ○ {orig}\n")
                    self.val_text.insert("end", f"       → {trans}\n")
                    if dup:
                        self.val_text.insert("end", f"       ⚠ {dup}\n")
                self.val_text.see("end")
        if matched == 0:
            self.val_status.configure(text="No matching input/output file pairs found")
        else:
            self.val_status.configure(text=f"Scanned {matched} file(s), found {total_issues} issue(s)")

    # ============================================================
    # UI helpers
    # ============================================================
    def _browse_input(self):
        d = filedialog.askdirectory()
        if d:
            self.input_dir.set(d)

    def _browse_output(self):
        d = filedialog.askdirectory()
        if d:
            self.output_dir.set(d)

    def _log_dir(self):
        d = os.path.join(os.path.dirname(self._config_path()), "rog")
        os.makedirs(d, exist_ok=True)
        return d

    def _init_log_file(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._current_log_path = os.path.join(self._log_dir(), f"log_{ts}.txt")
        try:
            with codecs.open(self._current_log_path, "w", encoding="utf-8") as f:
                f.write(f"=== OllamaTranslator Log ({ts}) ===\n\n")
            logs = sorted([os.path.join(self._log_dir(), f) for f in os.listdir(self._log_dir())
                          if f.startswith("log_") and f.endswith(".txt")], reverse=True)
            for old in logs[3:]:
                os.remove(old)
        except Exception as e:
            print(f"Log init failed: {e}")

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        if hasattr(self, 'log_text') and self.log_text.winfo_exists():
            self.after(0, lambda: self.log_text.insert("end", line + "\n") or self.log_text.see("end"))
        if self._current_log_path:
            try:
                with codecs.open(self._current_log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def update_progress(self, current, total):
        def _u():
            self.progress_bar.set(current / total if total > 0 else 0)
            self.progress_text.configure(text=f"{current} / {total} lines")
        self.after(0, _u)

    def set_status(self, status):
        def _u():
            if status == "translating":
                self.start_btn.configure(state="disabled")
                self.stop_btn.configure(state="normal")
                self.reset_btn.configure(state="disabled")
                self.status_label.configure(text="Translating...", text_color="#64B5F6")
            else:
                self.start_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
                self.reset_btn.configure(state="normal")
                self.status_label.configure(text="Ready", text_color="gray")
        self.after(0, _u)

    # ============================================================
    # Test
    # ============================================================
    def _test_connection(self):
        url = self.ollama_url.get().rstrip("/")
        model = self.ollama_model.get()
        target = self.target_lang.get()
        if not url or not model:
            self.log("[ERROR] Enter URL and model name first"); return
        if not target:
            self.log("[ERROR] Select Target language first"); return
        game = self.selected_game.get()
        self._sync_prompt()
        raw = self.prompt_template_var.get() or self._default_prompt()
        test_text = 'key: "Hello World"'
        bp = raw.replace("{source_lang}", "English").replace("{target_lang}", target).replace("{batch_text}", test_text)
        fp = get_enhanced_prompt(game, bp)
        df = os.path.join(self._log_dir(), "debug.txt")
        with codecs.open(df, "w", encoding="utf-8") as d:
            d.write(f"URL: {url}\nModel: {model}\nTarget: {target}\nGame: {game}\n\n=== TEST TEXT ===\n{test_text}\n\n=== SENT PROMPT ===\n{fp}\n\n")
        self.log(f"[TEST] Model: {model} | English -> {target} | Game: {game}")
        self.log(f"[TEST] Debug log: {df}")
        try:
            payload = {"model": model, "messages": [{"role": "user", "content": fp}],
                       "options": {"temperature": 0.1, "num_predict": 512}, "stream": False}
            with codecs.open(df, "a", encoding="utf-8") as d:
                d.write(f"=== REQUEST ===\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n")
            resp = requests.post(f"{url}/api/chat", json=payload, timeout=60)
            with codecs.open(df, "a", encoding="utf-8") as d:
                d.write(f"=== RESPONSE ===\n{resp.status_code}\n{resp.text}\n\n")
            data = resp.json().get("message", {})
            result = data.get("content", "").strip()
            if result:
                self.log(f"[TEST] Result: {result}")
            else:
                self.log(f"[TEST] Empty response")
        except Exception as e:
            self.log(f"[TEST] Failed: {e}")
            with codecs.open(df, "a", encoding="utf-8") as d:
                d.write(f"Exception: {e}\n")

    # ============================================================
    # Prompt
    # ============================================================
    def _toggle_prompt(self):
        self.prompt_frame.grid() if self.show_prompt.get() else self.prompt_frame.grid_remove()

    def _sync_prompt(self, event=None):
        self.prompt_template_var.set(self.prompt_textbox.get("1.0", "end-1c"))

    def _restore_default_prompt(self):
        d = self._default_prompt()
        self.prompt_template_var.set(d)
        self.prompt_textbox.delete("1.0", "end")
        self.prompt_textbox.insert("1.0", d)
        self.log("[INFO] Prompt restored to default")

    def _load_prompt_from_file(self):
        fp = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not fp:
            return
        try:
            with codecs.open(fp, "r", encoding="utf-8-sig") as f:
                c = f.read()
            self.prompt_template_var.set(c)
            self.prompt_textbox.delete("1.0", "end")
            self.prompt_textbox.insert("1.0", c)
            self.log(f"[INFO] Prompt loaded from {os.path.basename(fp)}")
        except Exception as e:
            self.log(f"[ERROR] Failed to load prompt: {e}")

    # ============================================================
    # 번역 시작/중단/초기화
    # ============================================================
    def _start(self):
        if not self.input_dir.get() or not self.output_dir.get():
            messagebox.showerror("Error", "Select input and output folders"); return
        if not self.ollama_url.get():
            messagebox.showerror("Error", "Enter Ollama URL"); return
        if not self.ollama_model.get():
            messagebox.showerror("Error", "Enter model name"); return

        if hasattr(self, 'log_text') and self.log_text.winfo_exists():
            self.log_text.delete("1.0", "end")
        self.live_orig.delete("1.0", "end")
        self.live_trans.delete("1.0", "end")
        self.progress_bar.set(0)
        self.progress_text.configure(text="0 / 0 lines")

        # 체크포인트 자동 복구
        if self.output_dir.get():
            for fn in os.listdir(os.path.dirname(CONFIG_FILE)):
                if not fn.endswith((".yml", ".yaml")):
                    continue
                cp = os.path.join(os.path.dirname(CONFIG_FILE), fn)
                if not os.path.isfile(cp):
                    continue
                for root, _, fnames in os.walk(self.output_dir.get()):
                    for fn2 in fnames:
                        if fn2.lower() == fn.lower():
                            with codecs.open(cp, "r", encoding="utf-8-sig") as f:
                                data = f.read()
                            with codecs.open(os.path.join(root, fn2), "w", encoding="utf-8-sig") as f:
                                f.write(data)
                            self.log(f"[RESUME] Restored checkpoint: {fn}")
                            break

        self.engine.set_base_url(self.ollama_url.get())
        self._sync_prompt()
        self.engine.prompt_template = self.prompt_template_var.get()
        self._save_config()
        self.engine.start(self.input_dir.get(), self.output_dir.get(), self.source_lang.get(), self.target_lang.get(),
                          self.ollama_model.get(), self.temperature.get(), self.max_tokens.get(), self.batch_size.get(),
                          self.selected_game.get(), 1, False, self.max_retries.get())

    def _stop(self):
        self.engine.stop()
        self.after(300, self._reset_ui)

    def _reset_ui(self):
        if hasattr(self, 'log_text') and self.log_text.winfo_exists():
            self.log_text.delete("1.0", "end")
        self.live_orig.delete("1.0", "end")
        self.live_trans.delete("1.0", "end")
        self.progress_bar.set(0)
        self.progress_text.configure(text="0 / 0 lines")
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.reset_btn.configure(state="disabled")
        self.status_label.configure(text="Ready", text_color="gray")

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
