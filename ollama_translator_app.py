# ============================================================
# Ollama Paradox Mod Translator
# ============================================================

import os, sys, json, time, re, codecs, threading, subprocess, concurrent.futures
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
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
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
                data = resp.json().get("models", [])
                return data
        except Exception:
            pass
        return None

    def test_model(self, model, target_lang, game="None"):
        raw = self.prompt_template or (
            "Translate from English to {target_lang}.\nRules:\n"
            "1. Only translate after ': ' in quotes.\n"
            "2. Keep $vars$, [brackets], §X intact.\n"
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
            with codecs.open(cp_path, "w", encoding="utf-8-sig") as f:
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

    def _translate_batch(self, lines, source_lang, target_lang, model, temperature, max_tokens, game="None", retry_count=0):
        header_pat = re.compile(r'^l_[a-z_]+:\s*$')
        comment_indices = {i for i, l in enumerate(lines) if not l.strip() or l.strip().startswith('#') or header_pat.match(l)}
        if comment_indices:
            if len(comment_indices) == len(lines):
                return lines
            actual_lines = [l for i, l in enumerate(lines) if i not in comment_indices]
            if self.debug_mode: self.log_callback(f"[DEBUG] batch:{len(lines)}lines filter→{len(actual_lines)}content")
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

        # ── Extract values, replace game codes with placeholders ──
        ph_counter = 0
        send_data = []      # (cleaned_value, ph_map, key, ws)
        all_info = []       # per line: {"type": "passthrough"/"keep"/"send", ...}

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
                send_data.append((f"⟨{idx}⟩ {cleaned}", line_phs, key, ws))

        if not send_data:
            return lines

        # ── Send value+marker lines to LLM ──
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
                f"2. Preserve line markers like ⟨0⟩ ⟨1⟩ exactly as-is.\n"
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

        # Split by value markers ⟨0⟩ ⟨1⟩ ... (survives even if LLM merges lines)
        translated_values = re.split(r'⟨\d+⟩\s*', result)
        if translated_values and translated_values[0].strip() == '':
            translated_values = translated_values[1:]

        if self.debug_mode: self.log_callback(f"[DEBUG] marker split → {len(translated_values)} values")

        if len(translated_values) != len(send_data):
            self.log_callback(f"[WARN] LLM returned {len(translated_values)} markers (expected {len(send_data)}), keeping original")
            for i, tv in enumerate(translated_values):
                self.log_callback(f"[WARN]   out[{i}]: {repr(tv[:100])}")
            return lines

        # ── Restore placeholders and reconstruct YAML lines ──
        reconstructed = list(lines)
        for i, info in enumerate(all_info):
            t = info["type"]
            if t == "passthrough":
                continue
            elif t == "keep":
                key, ws, val, phs = info["key"], info["ws"], info["val"], info["phs"]
                restored = val
                if restored.startswith('"') and restored.endswith('"'):
                    restored = restored[1:-1]
                for ph, orig in phs:
                    restored = restored.replace(ph, orig)
                if val.startswith('"'):
                    restored = restored.replace('\n', '\\n')
                    reconstructed[i] = f'{ws}{key}: "{restored}"\n'
                else:
                    restored = restored.replace('\n', '\\n')
                    reconstructed[i] = f'{ws}{key}: {restored}\n'
            else:  # send
                _, phs, send_key, send_ws = send_data[info["midx"]]
                raw = translated_values[info["midx"]].strip()
                if raw and raw != '""':
                    val = raw.strip('"')
                    for ph, orig in phs:
                        val = val.replace(ph, orig)
                    val = val.replace('\n', '\\n')
                    reconstructed[i] = f'{send_ws}{send_key}: "{val}"\n'

        # Fix headers
        reconstructed = [re.sub(r'^l_([a-z]+)::\s*$', r'l_\1:', t) for t in reconstructed]
        tgt_code = self._LANG_CODE.get(target_lang, target_lang.lower())
        header_pat = re.compile(r'^l_[a-z_]+:\s*$')
        reconstructed = [f"l_{tgt_code}:\n" if header_pat.match(t) else t for t in reconstructed]

        if self.live_callback:
            self.live_callback(lines, reconstructed)

        return reconstructed

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

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with codecs.open(output_path, "w", encoding="utf-8-sig") as f:
            f.writelines(result)
        self.log_callback(f"  Saved: {output_path}")

    def _worker(self, input_dir, output_dir, source_lang, target_lang, model, temperature, max_tokens, batch_size, game="None"):
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
            rel = os.path.relpath(os.path.dirname(fp), input_dir)
            base = os.path.basename(fp)
            new_base = re.sub(f"l_{source_code}", f"l_{target_code}", base, flags=re.IGNORECASE)
            out_path = os.path.join(output_dir, rel, new_base)
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
                with codecs.open(os.path.join(game_dir, fn), "r", encoding="utf-8") as f:
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
        text = "\n".join(f"  {s} → {t}" for s, t in combined.items())
        return (
            f"\n[REFERENCE GLOSSARY - 참고 용어집]\n{text}\n\n"
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
        self.source_lang = ctk.StringVar(value="English")
        self.target_lang = ctk.StringVar(value="Korean")
        self.ollama_model = ctk.StringVar()
        self.temperature = ctk.DoubleVar(value=0.2)
        self.max_tokens = ctk.IntVar(value=8192)
        self.batch_size = ctk.IntVar(value=1)
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
            self.max_retries.set(cfg.get("max_retries", 3))
            p = cfg.get("prompt_template", "")
            if p:
                self.prompt_template_var.set(p)
                self.prompt_textbox.delete("1.0", "end")
                self.prompt_textbox.insert("1.0", p)
            if cfg.get("show_prompt", False):
                self.show_prompt.set(True)
                self.prompt_frame.grid()
            # Glossary tab settings
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
        self.engine.kill_server()
        self._save_config()
        self.destroy()

    def _validate_fields(self, *args):
        ok = all([self.ollama_url.get(), self.ollama_model.get(),
                  self.input_dir.get(), self.output_dir.get(),
                  self.source_lang.get(), self.target_lang.get()])
        self.start_btn.configure(state="normal" if (ok and self._connected) else "disabled")

    def _default_prompt(self):
        return ("Translate the following text from '{source_lang}' to '{target_lang}'.\n"
                "Rules:\n1. Preserve all {{PH0}}, {{PH1}}, etc. placeholders exactly as-is.\n"
                "2. Preserve line markers like ⟨0⟩ ⟨1⟩ exactly as-is.\n"
                "3. Do NOT wrap in code blocks or add explanations.\n\n{batch_text}")

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

        # --- Checkboxes row ---
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

        # ========== Glossary Tab (GAME / MOD) ==========
        t_gl = self.tabview.add("Glossary")
        t_gl.grid_columnconfigure(0, weight=1)
        t_gl.grid_rowconfigure(0, weight=1)
        gl_sub = ctk.CTkTabview(t_gl)
        gl_sub.grid(row=0, column=0, sticky="nsew")
        game_tab = gl_sub.add("GAME")
        mod_tab = gl_sub.add("MOD")

        # ───── GAME Tab ─────
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

        # ───── MOD Tab ─────
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
        self._m_src_combo = ctk.CTkComboBox(mlf, variable=self._m_src_var, values=LANG_KEYS,
                                             state="readonly", width=150)
        self._m_src_combo.pack(side="left", padx=5)
        ctk.CTkLabel(mlf, text="  Target:").pack(side="left", padx=5)
        self._m_lang_combo = ctk.CTkComboBox(mlf, variable=self._m_tgt_var, values=LANG_KEYS,
                                              state="readonly", width=150)
        self._m_lang_combo.pack(side="left", padx=5)
        self._m_min_var = ctk.StringVar(value="3")
        ctk.CTkLabel(mlf, text="Min frequency:").pack(side="left", padx=5)
        ctk.CTkEntry(mlf, textvariable=self._m_min_var, width=50).pack(side="left", padx=5)
        ctk.CTkButton(mlf, text="Extract from Mod", fg_color="#1565C0", command=self._mod_extract).pack(side="left", padx=15)
        r += 1
        self._m_search_var = ctk.StringVar()
        msf = ctk.CTkFrame(mod_tab, fg_color="transparent")
        msf.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        ctk.CTkLabel(msf, text="Search:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        mse = ctk.CTkEntry(msf, textvariable=self._m_search_var, width=400)
        mse.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self._m_info_label = ctk.CTkLabel(msf, text="")
        self._m_info_label.grid(row=0, column=2, padx=10, pady=5, sticky="e")
        mse.bind("<KeyRelease>", lambda e: (setattr(self, '_m_page', 0), self._mod_update_display(self._m_search_var.get())))
        r += 1
        self._m_progress = ctk.CTkProgressBar(mod_tab)
        self._m_progress.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        self._m_progress.set(0)
        r += 1
        self._m_result_frame = ctk.CTkScrollableFrame(mod_tab)
        self._m_result_frame.grid(row=r, column=0, padx=10, pady=5, sticky="nsew")
        self._m_result_frame.grid_columnconfigure(2, weight=1)
        r += 1
        mpf = ctk.CTkFrame(mod_tab, fg_color="transparent")
        mpf.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        self._m_page_label = ctk.CTkLabel(mpf, text="")
        self._m_page_label.pack(side="left", padx=5)
        ctk.CTkButton(mpf, text="< Prev", width=60, command=self._m_prev_page).pack(side="left", padx=5)
        ctk.CTkButton(mpf, text="Next >", width=60, command=self._m_next_page).pack(side="left", padx=5)
        r += 1
        mbtn = ctk.CTkFrame(mod_tab, fg_color="transparent")
        mbtn.grid(row=r, column=0, padx=10, pady=5, sticky="ew")
        self._mod_tl_btn = ctk.CTkButton(mbtn, text="Translate with LLM", fg_color="#E65100", command=self._mod_translate,
                                          state="disabled")
        self._mod_tl_btn.pack(side="left", padx=5)
        self._mod_retry_btn = ctk.CTkButton(mbtn, text="Retry Selected", fg_color="#FF8F00",
                                             state="disabled", command=self._mod_retry_selected)
        self._mod_retry_btn.pack(side="left", padx=5)
        ctk.CTkButton(mbtn, text="Save Glossary", fg_color="#2E7D32", command=self._mod_save).pack(side="left", padx=5)
        ctk.CTkButton(mbtn, text="Load Glossary", command=self._mod_load).pack(side="left", padx=5)

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
                in_path = os.path.join(inp, rel, re.sub(r'_?l_[a-z]+_', f'_l_{src_code}_', fn, flags=re.IGNORECASE))
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
    # Glossary - GAME / MOD shared helpers
    # ============================================================
    @staticmethod
    def _parse_yml(path):
        data = {}
        try:
            with codecs.open(path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    m = re.match(r'^\s*([\w.]+):\d*\s*"(.+)"', line)
                    if m:
                        data[m.group(1)] = m.group(2)
        except Exception:
            pass
        return data

    @staticmethod
    def _tokenize_en(val):
        clean = re.sub(r'\[.*?\]|\$.*?\$|§.', '', val)
        words = re.findall(r'[a-zA-Z]{3,}', clean)
        STOP = {"the","and","for","are","but","not","you","all","can","had","her","was","one","our","out","has","have","been","its","his","that","with","from","this","will","your","which","than","what","when","were","been","also","each","any","how","who","may","their","them","than","about","after","before","just","like","more","most","much","only","other","over","some","such","than","then","very","well","than","even"}
        return [w.lower() for w in words if w.lower() not in STOP and not w.isdigit()]

    @staticmethod
    def _tokenize_tgt(val):
        clean = re.sub(r'\[.*?\]|\$.*?\$|§.', '', val)
        tokens = re.findall(r'[\uAC00-\uD7AFa-zA-Z\u4E00-\u9FFF\u0400-\u04FF]{2,}', clean)
        KO_STOP = {"\uc6b0\ub9ac", "\uc800\ud76c", "\uadf8\uac83", "\uc774\uac83", "\uc800\uac83",
                   "\ub204\uad70\uac00", "\ubb34\uc5b8\uac00", "\ubaa8\ub4e0", "\uc5ec\ub7ec", "\ub2e4\ub978",
                   "\uac19\uc740", "\uadf8\ub7f0", "\uc774\ub7f0", "\uc800\ub7f0",
                   "\uc544\uc8fc", "\ub9e4\uc6b0", "\uc815\ub9d0", "\ub108\ubb34", "\ub610\ud55c",
                   "\uadf8\ub9ac\uace0", "\ud558\uc9c0\ub9cc", "\ub54c\ubb38", "\uc704\ud574", "\ud1b5\ud574",
                   "\ud1b5\ud55c", "\ub300\ud55c", "\ub300\ud574", "\uc720\ub9ac", "\ubb34\ub8cc",
                   "\uac00\ub2a5", "\ubaa8\ub4e0\uac83", "\uc544\ubb34", "\uc5b8\uc81c", "\uc5b4\ub514",
                   "\ubb34\uc2a8", "\ub204\uad6c", "\uc65c", "\uc5b4\ub5bb\uac8c", "\ub9ce\uc740"}
        result = []
        for t in tokens:
            if t in KO_STOP:
                continue
            if len(t) >= 3 and t[-1] == '\ub2e4':
                continue
            result.append(t)
        return result

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
        r'Epic Games[\\/]([^\\/]+)',
        r'XboxGames[\\/]([^\\/]+)[\\/]Content',
    ]

    GAME_NAME_MAP = {
        "Crusader Kings III": "Crusader Kings 3", "ck3": "Crusader Kings 3",
        "Hearts of Iron IV": "Hearts of Iron 4", "hoi4": "Hearts of Iron 4",
        "Europa Universalis IV": "Europa Universalis IV", "eu4": "Europa Universalis IV",
        "Victoria 3": "Victoria 3", "vic3": "Victoria 3",
        "ImperatorRome": "Imperator: Rome", "Imperator Rome": "Imperator: Rome",
        "imperator": "Imperator: Rome",
    }

    def _detect_game_from_path(self, path):
        for pat in self.PLATFORM_PATTERNS:
            m = re.search(pat, path)
            if m:
                folder = m.group(1)
                if folder in self.GAME_NAME_MAP:
                    return self.GAME_NAME_MAP[folder]
                fl = folder.lower()
                for key, val in self.GAME_NAME_MAP.items():
                    if key.lower() == fl:
                        return val
                for g in self.available_games:
                    if g.lower().replace(" ", "").replace(":", "") == fl.replace(" ", "").replace(":", ""):
                        return g
                break
        return None

    def _g_browse(self):
        path = filedialog.askdirectory()
        if not path:
            return
        self._g_folder_var.set(path)
        game = self._detect_game_from_path(path)
        if game:
            self._g_game_var.set(game)

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

    # ───── GAME tab methods ─────
    def _game_extract(self):
        if getattr(self, '_g_running', False):
            return
        folder = self._g_folder_var.get()
        src_lang = self._g_src_var.get()
        tgt_lang = self._g_tgt_var.get()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Select a valid Languages Folder")
            return
        try:
            min_occ = int(self._g_min_var.get())
        except:
            min_occ = 3
        self._g_running = True
        self._g_progress.set(0)
        self._g_info_label.configure(text="Scanning files...")

        def _run():
            src_pre = self.LANG_PREFIX.get(src_lang, f"l_{src_lang.lower()}")
            tgt_pre = self.LANG_PREFIX.get(tgt_lang, f"l_{tgt_lang.lower()}")
            src_short = src_lang.lower()
            tgt_short = tgt_lang.lower()
            pairs = {}
            for root, _, fnames in os.walk(folder):
                for fn in fnames:
                    if not fn.endswith((".yml", ".yaml")):
                        continue
                    lower = fn.lower()
                    fp = os.path.join(root, fn)
                    if lower.startswith(src_pre.lower()):
                        pairs.setdefault(fn[len(src_pre):], {})["src"] = fp
                    elif lower.startswith(tgt_pre.lower()):
                        pairs.setdefault(fn[len(tgt_pre):], {})["tgt"] = fp
                    dirname = os.path.basename(os.path.dirname(fp)).lower()
                    if dirname in (src_short, tgt_short):
                        base = re.sub(r'_l_[a-z]+', '', fn, flags=re.IGNORECASE)
                        base = re.sub(r'^l_[a-z]+_?', '', base, flags=re.IGNORECASE)
                        if dirname == src_short:
                            pairs.setdefault(base, {})["src"] = fp
                        elif dirname == tgt_short:
                            pairs.setdefault(base, {})["tgt"] = fp
            complete = [(b, f) for b, f in pairs.items() if "src" in f and "tgt" in f]
            total = len(complete)
            if total == 0:
                self.after(0, lambda: self._g_info_label.configure(text="No matching file pairs found"))
                self._g_running = False
                return

            def _process_pair(base, fdict):
                co = {}
                en_c = {}
                ko_c = {}
                sd = self._parse_yml(fdict["src"])
                td = self._parse_yml(fdict["tgt"])
                common = set(sd) & set(td)
                for key in common:
                    sv = OllamaTranslator._strip_codes(sd[key])
                    tv = OllamaTranslator._strip_codes(td[key])
                    en_set = set(self._tokenize_en(sv))
                    ko_set = set(self._tokenize_tgt(tv))
                    for st in en_set:
                        en_c[st] = en_c.get(st, 0) + 1
                    for tt in ko_set:
                        ko_c[tt] = ko_c.get(tt, 0) + 1
                    for st in en_set:
                        for tt in ko_set:
                            co.setdefault(st, {}).setdefault(tt, 0)
                            co[st][tt] += 1
                return co, en_c, ko_c

            cooccur = {}
            en_total = {}
            ko_total = {}
            _g_last_update = [0.0]
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futs = [pool.submit(_process_pair, b, f) for b, f in complete]
                done = 0
                for fut in concurrent.futures.as_completed(futs):
                    if not self._g_running:
                        break
                    co, en_c, ko_c = fut.result()
                    for st, td in co.items():
                        if st not in cooccur:
                            cooccur[st] = {}
                            en_total[st] = 0
                        en_total[st] += en_c.get(st, 0)
                        for tt, c in td.items():
                            ko_total[tt] = ko_total.get(tt, 0) + ko_c.get(tt, 0)
                            cooccur[st][tt] = cooccur[st].get(tt, 0) + c
                    done += 1
                    now = time.time()
                    if now - _g_last_update[0] >= 0.5 or done == total:
                        _g_last_update[0] = now
                        self.after(0, lambda p=done/total: self._g_progress.set(p))
                        self.after(0, lambda d=done, t=total: self._g_info_label.configure(
                            text=f"Processing {d}/{t} file pairs..."))

            # Dice coefficient scoring
            scored = []
            for st, td in cooccur.items():
                en_freq = en_total.get(st, 0)
                if en_freq < min_occ:
                    continue
                for tt, c in td.items():
                    ko_freq = ko_total.get(tt, 0)
                    dice = 2 * c / (en_freq + ko_freq)
                    scored.append((st, tt, dice, c, en_freq))

            # For each English word, pick Korean with highest Dice
            best_map = {}
            for st, tt, dice, c, ef in scored:
                if st not in best_map or dice > best_map[st][1]:
                    best_map[st] = (tt, dice, ef)

            entries = [{"src": st, "tgt": tt, "count": ef, "dice": round(dice, 3)}
                       for st, (tt, dice, ef) in best_map.items()]
            # 중복 한국어 제거: 같은 번역이면 Dice 점수 높은 영어만 유지
            tgt_seen = {}
            for e in entries:
                tgt = e["tgt"]
                if tgt not in tgt_seen or e["dice"] > tgt_seen[tgt]["dice"]:
                    tgt_seen[tgt] = e
            result = sorted(tgt_seen.values(), key=lambda x: -x["dice"])
            self.after(0, lambda: self._g_progress.set(1))
            self.after(0, lambda r=result, c=total: self._g_entries_update(r, c))
            self._g_running = False

        threading.Thread(target=_run, daemon=True).start()

    def _g_entries_update(self, entries, pair_count):
        self._g_entries = entries
        self._g_page = 0
        self._game_update_display(self._g_search_var.get())
        total = len(entries)
        self._g_info_label.configure(text=f"{total} terms from {pair_count} file pairs")

    def _game_update_display(self, filter_text=""):
        for w in self._g_result_frame.winfo_children():
            w.destroy()
        items = self._g_entries if hasattr(self, '_g_entries') else []
        if filter_text:
            fl = filter_text.lower()
            items = [e for e in items if fl in e["src"].lower() or fl in e["tgt"].lower()]
        if not items:
            ctk.CTkLabel(self._g_result_frame, text="No terms. Click 'Extract' first.").pack(pady=10)
            self._g_page_label.configure(text="")
            return
        total_pages = max(1, (len(items) + self._g_per_page - 1) // self._g_per_page)
        if self._g_page >= total_pages:
            self._g_page = total_pages - 1
        start = self._g_page * self._g_per_page
        end = min(start + self._g_per_page, len(items))
        page_items = items[start:end]
        for entry in page_items:
            row = ctk.CTkFrame(self._g_result_frame, fg_color="transparent")
            row.pack(fill="x", padx=2, pady=1)
            row.grid_columnconfigure(2, weight=1)
            ctk.CTkLabel(row, text=entry["src"], width=140, anchor="w").grid(row=0, column=0, padx=3)
            var = ctk.StringVar(value=entry["tgt"])
            entry["_var"] = var
            ent = ctk.CTkEntry(row, textvariable=var, width=180)
            ent.grid(row=0, column=1, padx=3)
            var.trace_add("write", lambda *a, e=entry, v=var: e.update({"tgt": v.get()}))
            ctk.CTkLabel(row, text=str(entry["count"]), width=40, anchor="e").grid(row=0, column=2, padx=3)
        self._g_page_label.configure(text=f"Page {self._g_page+1}/{total_pages} ({len(items)} total)")

    def _g_prev_page(self):
        if self._g_page > 0:
            self._g_page -= 1
            self._game_update_display(self._g_search_var.get())

    def _g_next_page(self):
        items = self._g_entries if hasattr(self, '_g_entries') else []
        total_pages = max(1, (len(items) + self._g_per_page - 1) // self._g_per_page)
        if self._g_page < total_pages - 1:
            self._g_page += 1
            self._game_update_display(self._g_search_var.get())


    def _game_save(self):
        game = self._g_game_var.get()
        tgt = self._g_tgt_var.get()
        if not hasattr(self, '_g_entries') or not self._g_entries:
            messagebox.showinfo("Info", "No terms to save. Extract first.")
            return
        g_dir = os.path.join(OllamaTranslator._glossary_dir(), game)
        os.makedirs(g_dir, exist_ok=True)
        path = os.path.join(g_dir, f"game_{tgt.lower()}.txt")
        try:
            with codecs.open(path, "w", encoding="utf-8") as f:
                f.write(f"# Glossary for {game} - {tgt}\n")
                for e in self._g_entries:
                    if e.get("tgt", "").strip():
                        f.write(f"{e['src']}:{e['tgt'].strip()}\n")
            self.log(f"[GLOSSARY] Saved {len(self._g_entries)} terms to {path}")
            messagebox.showinfo("Glossary", f"Saved to {path}")
        except Exception as ex:
            messagebox.showerror("Error", f"Save failed: {ex}")

    def _game_load(self):
        game = self._g_game_var.get()
        tgt = self._g_tgt_var.get()
        path = os.path.join(OllamaTranslator._glossary_dir(), game, f"game_{tgt.lower()}.txt")
        if not os.path.exists(path):
            messagebox.showinfo("Glossary", f"No file at {path}")
            return
        try:
            entries = []
            with codecs.open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or ":" not in line:
                        continue
                    s, t = line.split(":", 1)
                    entries.append({"src": s.strip(), "tgt": t.strip(), "count": 0})
            self._g_entries = entries
            self._game_update_display(self._g_search_var.get())
            self._g_info_label.configure(text=f"Loaded {len(entries)} terms")
            self.log(f"[GLOSSARY] Loaded {len(entries)} terms from {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Load failed: {e}")

    # ───── MOD tab methods ─────
    def _mod_extract(self):
        if getattr(self, '_m_running', False):
            return
        folder = self._m_folder_var.get()
        tgt_lang = self._m_tgt_var.get()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Select a valid Mod English Folder")
            return
        try:
            min_freq = int(self._m_min_var.get())
        except:
            min_freq = 3
        self._m_running = True
        self._m_progress.set(0)
        self._m_info_label.configure(text="Scanning...")
        def _run():
            _m_last_update = [0.0]
            term_counts = {}
            files = []
            for root, _, fnames in os.walk(folder):
                for fn in fnames:
                    if fn.endswith((".yml", ".yaml")):
                        files.append(os.path.join(root, fn))
            if not files:
                self._m_running = False
                self.after(0, lambda: self._m_info_label.configure(text="No YML files found"))
                return
            for idx, fp in enumerate(files):
                if not self._m_running:
                    break
                with codecs.open(fp, "r", encoding="utf-8-sig") as f:
                    for line in f:
                        m = re.match(r'^\s*([\w.]+):\d*\s*"(.+)"', line)
                        if not m:
                            continue
                        for w in self._tokenize_en(m.group(2)):
                            term_counts[w] = term_counts.get(w, 0) + 1
                now = time.time()
                if now - _m_last_update[0] >= 0.5 or idx == len(files) - 1:
                    _m_last_update[0] = now
                    self.after(0, lambda p=(idx+1)/len(files): self._m_progress.set(p))
                    self.after(0, lambda i=idx, t=len(files): self._m_info_label.configure(text=f"Scanning {i+1}/{t}..."))
            filtered = sorted([(c, w) for w, c in term_counts.items() if c >= min_freq], reverse=True)
            entries = [{"src": w, "tgt": "", "count": c, "_checked": False} for c, w in filtered]
            self.after(0, lambda: self._m_progress.set(1))
            self.after(0, lambda e=entries: self._m_entries_update(e))
            self._m_running = False
        threading.Thread(target=_run, daemon=True).start()

    def _m_entries_update(self, entries):
        self._m_entries = entries
        self._mod_update_display(self._m_search_var.get())
        self._m_info_label.configure(text=f"{len(entries)} terms")

    def _mod_translate(self):
        if getattr(self, '_m_llm_running', False):
            return
        if not hasattr(self, '_m_entries') or not self._m_entries:
            messagebox.showinfo("Info", "Extract terms first")
            return
        pending = [e for e in self._m_entries if not e["tgt"].strip()]
        if not pending:
            messagebox.showinfo("Info", "All terms already have translations")
            return
        self._m_llm_running = True
        self._mod_tl_btn.configure(state="disabled")
        src_lang = self._m_src_var.get()
        tgt_lang = self._m_tgt_var.get()
        batch_size = 30
        total_batches = (len(pending) + batch_size - 1) // batch_size
        self.log(f"[MOD] Translating {len(pending)} terms via LLM...")
        def _run():
            _m_llm_last = [0.0]
            for i in range(0, len(pending), batch_size):
                batch = pending[i:i+batch_size]
                terms = "\n".join(e["src"] for e in batch)
                prompt = f"Translate these game terms from {src_lang} to {tgt_lang}. Return ONLY 'term:translation' lines.\n\n{terms}"
                try:
                    r = self.engine._call_ollama(self.ollama_model.get(), prompt, temperature=0.1, max_tokens=2000)
                    if r and not r.startswith("[OLLAMA_"):
                        for line in r.strip().split("\n"):
                            if ":" in line:
                                parts = line.split(":", 1)
                                s = parts[0].strip().lower()
                                t = parts[1].strip()
                                for e in batch:
                                    if e["src"].lower() == s:
                                        e["tgt"] = t
                                        break
                except Exception as ex:
                    self.log(f"[MOD] LLM error: {ex}")
                now = time.time()
                if now - _m_llm_last[0] >= 0.5 or i + batch_size >= len(pending):
                    _m_llm_last[0] = now
                    pct = min(1.0, (i + batch_size) / len(pending))
                    self.after(0, lambda p=pct: self._m_progress.set(p))
                    self.after(0, lambda: self._m_info_label.configure(text=f"Translating..."))
            self.after(0, lambda: self._m_progress.set(1))
            self.after(0, lambda: self._mod_update_display(self._m_search_var.get()))
            self.after(0, lambda: self._mod_tl_btn.configure(state="normal" if self._connected else "disabled"))
            self._m_llm_running = False
            self.log(f"[MOD] Translation done")
        threading.Thread(target=_run, daemon=True).start()

    def _mod_update_display(self, filter_text=""):
        for w in self._m_result_frame.winfo_children():
            w.destroy()
        items = self._m_entries if hasattr(self, '_m_entries') else []
        if filter_text:
            fl = filter_text.lower()
            items = [e for e in items if fl in e["src"].lower() or fl in e["tgt"].lower()]
        if not items:
            ctk.CTkLabel(self._m_result_frame, text="No terms. Click 'Extract from Mod' first.").pack(pady=10)
            self._m_page_label.configure(text="")
            return
        total_pages = max(1, (len(items) + self._m_per_page - 1) // self._m_per_page)
        if self._m_page >= total_pages:
            self._m_page = total_pages - 1
        start = self._m_page * self._m_per_page
        end = min(start + self._m_per_page, len(items))
        page_items = items[start:end]
        any_checked = False
        for entry in page_items:
            row = ctk.CTkFrame(self._m_result_frame, fg_color="transparent")
            row.pack(fill="x", padx=2, pady=1)
            row.grid_columnconfigure(3, weight=1)
            checked = ctk.BooleanVar(value=entry.get("_checked", False))
            entry["_checked_var"] = checked
            def _cb(*a, e=entry, v=checked):
                e["_checked"] = v.get()
                self._mod_update_retry_btn()
            checked.trace_add("write", _cb)
            ctk.CTkCheckBox(row, variable=checked, text="", width=20).grid(row=0, column=0, padx=2)
            ctk.CTkLabel(row, text=f"{entry['src']} ({entry['count']}x)", width=130, anchor="w").grid(row=0, column=1, padx=3)
            var = ctk.StringVar(value=entry["tgt"])
            entry["_var"] = var
            ent = ctk.CTkEntry(row, textvariable=var, width=170)
            ent.grid(row=0, column=2, padx=3)
            var.trace_add("write", lambda *a, e=entry, v=var: e.update({"tgt": v.get()}))
            status = "✓" if entry["tgt"].strip() else "..."
            ctk.CTkLabel(row, text=status, width=25, anchor="w").grid(row=0, column=3, padx=3)
            if entry.get("_checked", False):
                any_checked = True
        self._m_page_label.configure(text=f"Page {self._m_page+1}/{total_pages} ({len(items)} total)")
        self._mod_update_retry_btn()

    def _mod_update_retry_btn(self):
        has = any(e.get("_checked", False) for e in (self._m_entries if hasattr(self, '_m_entries') else []))
        if hasattr(self, '_mod_retry_btn'):
            self._mod_retry_btn.configure(state="normal" if (has and self._connected) else "disabled")

    def _mod_retry_selected(self):
        if getattr(self, '_m_llm_running', False):
            return
        selected = [e for e in (self._m_entries if hasattr(self, '_m_entries') else []) if e.get("_checked", False)]
        if not selected:
            return
        self._m_llm_running = True
        self._mod_retry_btn.configure(state="disabled")
        src_lang = self._m_src_var.get()
        tgt_lang = self._m_tgt_var.get()
        batch_size = 30
        self.log(f"[MOD] Retranslating {len(selected)} selected terms...")
        def _run():
            for i in range(0, len(selected), batch_size):
                batch = selected[i:i+batch_size]
                terms = "\n".join(e["src"] for e in batch)
                prompt = f"Translate these game terms from {src_lang} to {tgt_lang}. Return ONLY 'term:translation' lines.\n\n{terms}"
                try:
                    r = self.engine._call_ollama(self.ollama_model.get(), prompt, temperature=0.1, max_tokens=2000)
                    if r and not r.startswith("[OLLAMA_"):
                        for line in r.strip().split("\n"):
                            if ":" in line:
                                parts = line.split(":", 1)
                                s = parts[0].strip().lower()
                                t = parts[1].strip()
                                for e in batch:
                                    if e["src"].lower() == s:
                                        e["tgt"] = t
                                        if "_var" in e:
                                            e["_var"].set(t)
                                        break
                except Exception as ex:
                    self.log(f"[MOD] LLM error: {ex}")
                pct = min(1.0, (i + batch_size) / len(selected))
                self.after(0, lambda p=pct: self._m_progress.set(p))
            self.after(0, lambda: self._m_progress.set(1))
            self.after(0, lambda: self._mod_update_display(self._m_search_var.get()))
            self._m_llm_running = False
            self.log(f"[MOD] Retranslation done")
        threading.Thread(target=_run, daemon=True).start()

    def _m_prev_page(self):
        if self._m_page > 0:
            self._m_page -= 1
            self._mod_update_display(self._m_search_var.get())

    def _m_next_page(self):
        items = self._m_entries if hasattr(self, '_m_entries') else []
        total_pages = max(1, (len(items) + self._m_per_page - 1) // self._m_per_page)
        if self._m_page < total_pages - 1:
            self._m_page += 1
            self._mod_update_display(self._m_search_var.get())

    def _mod_save(self):
        game = self._m_game_var.get()
        mod = self._m_modname or "unknown"
        tgt = self._m_tgt_var.get()
        if not hasattr(self, '_m_entries') or not self._m_entries:
            messagebox.showinfo("Info", "No terms to save. Extract first.")
            return
        g_dir = os.path.join(OllamaTranslator._glossary_dir(), game)
        os.makedirs(g_dir, exist_ok=True)
        path = os.path.join(g_dir, f"mod_{mod}_{tgt.lower()}.txt")
        try:
            with codecs.open(path, "w", encoding="utf-8") as f:
                f.write(f"# Glossary for {game} - {mod} ({tgt})\n")
                for e in self._m_entries:
                    if e.get("tgt", "").strip():
                        f.write(f"{e['src']}:{e['tgt'].strip()}\n")
            self.log(f"[GLOSSARY] Saved {len(self._m_entries)} terms to {path}")
            messagebox.showinfo("Glossary", f"Saved to {path}")
        except Exception as ex:
            messagebox.showerror("Error", f"Save failed: {ex}")

    def _mod_load(self):
        game = self._m_game_var.get()
        tgt = self._m_tgt_var.get()
        mod = self._m_modname or "unknown"
        fd = os.path.join(OllamaTranslator._glossary_dir(), game)
        fn = f"mod_{mod}_{tgt.lower()}.txt"
        fp = os.path.join(fd, fn)
        if not os.path.exists(fp):
            messagebox.showinfo("Glossary", f"No glossary file: {fn}")
            return
        if not fp:
            return
        try:
            entries = []
            with codecs.open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or ":" not in line:
                        continue
                    s, t = line.split(":", 1)
                    entries.append({"src": s.strip(), "tgt": t.strip(), "count": 0})
            self._m_entries = entries
            self._mod_update_display(self._m_search_var.get())
            self._m_info_label.configure(text=f"Loaded {len(entries)} terms")
            self.log(f"[GLOSSARY] Loaded {len(entries)} terms from {fp}")
        except Exception as e:
            messagebox.showerror("Error", f"Load failed: {e}")

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
                not_busy = not getattr(self.engine, 'busy', False)
                can_start = not_busy and self._connected and all([
                    self.ollama_url.get(), self.ollama_model.get(),
                    self.input_dir.get(), self.output_dir.get(),
                    self.source_lang.get(), self.target_lang.get()])
                self.start_btn.configure(state="normal" if can_start else "disabled")
                self.stop_btn.configure(state="disabled")
                self.reset_btn.configure(state="normal")
                self.status_label.configure(text="Ready", text_color="gray")
                if not self._connected:
                    self.start_btn.configure(state="disabled")
        self.after(0, _u)

    # ============================================================
    # Test
    # ============================================================
    def _start_ollama(self):
        url = self.ollama_url.get().rstrip("/") or "http://localhost:11434"
        self.log("[OLLAMA] Starting server (wait up to 30s)...")
        def _run():
            models = self.engine.start_server()
            if models:
                self.after(0, lambda m=models, u=url: (
                    self.ollama_url.set(u),
                    self.ollama_combo.configure(values=m),
                    self.ollama_model.set(m[0])
                ))
                self.log(f"[OLLAMA] Connected. Models: {', '.join(models[:8])}")
            else:
                self.log("[OLLAMA] Failed to start or connect")
        threading.Thread(target=_run, daemon=True).start()

    def _connect_ollama(self):
        model = self.ollama_model.get()
        target = self.target_lang.get()
        if not model or model == "(none)":
            self.log("[ERROR] No model selected. Click Start Ollama first."); return
        if not target:
            self.log("[ERROR] Select Target language first"); return

        self.log("[CONNECT] Loading model...")
        self.start_btn.configure(state="disabled")
        self.live_btn.configure(state="disabled")
        self.reset_btn.configure(state="disabled")
        def _run():
            m = model
            self.log(f"[CONNECT] Checking model {m}...")
            running = self.engine.get_running_models()
            if running is None:
                self.log("[ERROR] Cannot reach Ollama server")
                self.after(0, lambda: (self.start_btn.configure(state="normal"),
                                       self.live_btn.configure(state="normal"),
                                       self.reset_btn.configure(state="normal")))
                return
            if m not in [x.get("name", "") for x in running]:
                self.log(f"[CONNECT] Loading model {m} (may take a while)...")
                self.engine._call_ollama(m, "test", temperature=0.1, max_tokens=1)
                for i in range(60):
                    time.sleep(1)
                    running = self.engine.get_running_models()
                    if running and m in [x.get("name", "") for x in running]:
                        self.log(f"[CONNECT] Model loaded ({i+1}s)")
                        break

            game = self.selected_game.get()
            self._sync_prompt()
            self.engine.prompt_template = self.prompt_template_var.get()
            result = self.engine.test_model(m, target, game)
            if result and not result.startswith("[OLLAMA_"):
                self.log(f"[CONNECT] {m}: {result.strip()}")
                self.log("[CONNECT] Translator Ready")
            else:
                self.log(f"[CONNECT] Test failed: {result}")
            def _done():
                self._connected = True
                self._validate_fields()
                self.live_btn.configure(state="normal")
                self.reset_btn.configure(state="normal")
                if hasattr(self, '_mod_tl_btn'):
                    self._mod_tl_btn.configure(state="normal")
                if hasattr(self, '_mod_retry_btn'):
                    self._mod_update_retry_btn()
            self.after(0, _done)
        threading.Thread(target=_run, daemon=True).start()

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

        # 체크포인트 복구 + 디버그
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
                          self.selected_game.get(), self.max_retries.get())

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
