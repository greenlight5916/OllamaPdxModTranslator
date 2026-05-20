import os, sys, json, time, re, threading, subprocess, concurrent.futures, traceback
import requests

def _app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(_app_dir(), "ollama_translator_config.json")

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

class OllamaTranslator:
    def __init__(self, log_callback=None, progress_callback=None, status_callback=None, live_callback=None, warn_callback=None):
        self.log_callback = log_callback or (lambda x: None)
        self.progress_callback = progress_callback or (lambda *a: None)
        self.status_callback = status_callback or (lambda x: None)
        self.live_callback = live_callback
        self.warn_callback = warn_callback or (lambda *a: None)
        self.base_url = "http://localhost:11434"
        self.stop_event = threading.Event()
        self.busy = False
        self.process = None
        self._consecutive_errors = 0
        self.max_retries = 3
        self.config = {}
        if os.path.isfile(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as _f:
                    self.config = json.load(_f)
            except Exception:
                self.config = {}
        self._checkpoint_dir = "checkpoints"
        self.prompt_template = None
        self.checkpoint_enabled = True
        self.debug_mode = False
        self.glossary_limit = 500
        self._checkpoint_restored = False
        self._file_warnings = {}

    def set_base_url(self, url):
        self.base_url = url.rstrip("/")

    def start_server(self):
        """Start Ollama server and return model list."""
        try:
            self.log_callback("[OLLAMA] Starting Ollama server...")
            self.process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            # Wait for server to start and become responsive
            for i in range(30):
                time.sleep(1)
                try:
                    r = requests.get(f"{self.base_url}/api/tags", timeout=2)
                    if r.status_code == 200:
                        models = [m["name"] for m in r.json().get("models", [])]
                        self.log_callback(f"[OLLAMA] Server ready ({len(models)} model(s) found)")
                        return models
                except Exception:
                    continue
            self.log_callback("[OLLAMA] Server start timed out after 30s")
            return None
        except Exception as e:
            self.log_callback(f"[OLLAMA] Failed to start server: {e}")
            return None

    def kill_server(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

    def fetch_models(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if r.status_code == 200:
                data = r.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return None

    def get_running_models(self):
        try:
            r = requests.get(f"{self.base_url}/api/ps", timeout=5)
            if r.status_code == 200:
                data = r.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def test_model(self, model, target_lang, game="None", num_ctx=4096):
        raw = self.prompt_template or (
            "Translate from English to {target_lang}.\nRules:\n"
            "1. Only translate after ': ' in quotes.\n"
            "2. Keep $vars$, [brackets], sectX intact.\n"
            "3. Output exactly one line.\n\n{batch_text}")
        test_text = 'key: "Hello World"'
        bp = raw.replace("{source_lang}", "English").replace("{target_lang}", target_lang).replace("{batch_text}", test_text)
        prompt = get_enhanced_prompt(game, bp)
        return self._call_ollama(model, prompt, temperature=0.1, max_tokens=512, num_ctx=num_ctx)

    def _check_fatal(self):
        self._consecutive_errors += 1
        if self._consecutive_errors >= self.max_retries + 2:
            self.log_callback("[FATAL] Consecutive LLM failures - stopping translation")
            self.stop_event.set()
            return True
        return False

    def _call_ollama(self, model, prompt, temperature=0.5, max_tokens=4096, num_ctx=4096, timeout=90):
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": num_ctx},
                    "stream": False
                },
                timeout=timeout
            )
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

    @staticmethod
    def _normalize_text(text):
        replacements = {
            '\u201c': '"', '\u201d': '"', '\u201e': '"', '\u201f': '"',
            '\u2018': "'", '\u2019': "'", '\u201a': "'", '\u201b': "'",
            '\u2014': '-', '\u2013': '-',
            '\u2026': '...',
            '\u00b7': '.',
            '\uff01': '!', '\uff0c': ',', '\uff1a': ':',
            '\u3001': ',', '\u3002': '.',
            '\u30fb': '.',
            '\u00e9': 'e', '\u00e8': 'e', '\u00ea': 'e', '\u00eb': 'e',
            '\u00e0': 'a', '\u00e1': 'a', '\u00e2': 'a', '\u00e3': 'a', '\u00e4': 'a', '\u00e5': 'a',
            '\u00ec': 'i', '\u00ed': 'i', '\u00ee': 'i', '\u00ef': 'i',
            '\u00f2': 'o', '\u00f3': 'o', '\u00f4': 'o', '\u00f5': 'o', '\u00f6': 'o',
            '\u00f9': 'u', '\u00fa': 'u', '\u00fb': 'u', '\u00fc': 'u',
            '\u00f1': 'n',
            '\u00a0': ' ',
            '\u0430': 'a', '\u0431': 'b', '\u0432': 'v', '\u0433': 'g',
            '\u0434': 'd', '\u0435': 'e', '\u0436': 'zh', '\u0437': 'z',
            '\u0438': 'i', '\u0439': 'i', '\u043a': 'k', '\u043b': 'l',
            '\u043c': 'm', '\u043d': 'n', '\u043e': 'o', '\u043f': 'p',
            '\u0440': 'r', '\u0441': 'c', '\u0442': 't', '\u0443': 'u',
            '\u0444': 'f', '\u0445': 'kh', '\u0446': 'ts', '\u0447': 'ch',
            '\u0448': 'sh', '\u0449': 'shch', '\u044b': 'y', '\u044d': 'e',
            '\u044e': 'yu', '\u044f': 'ya',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _translate_batch(self, lines, source_lang, target_lang, model, temperature, max_tokens, num_ctx=4096, game="None", retry_count=0, timeout=90, mod_folder="", output_path="", _file_offset=0):
        header_pat = re.compile(r'^l_[a-z_]+:\s*$')
        comment_indices = {i for i, l in enumerate(lines) if not l.strip() or l.strip().startswith('#') or header_pat.match(l)}
        if comment_indices:
            if len(comment_indices) == len(lines):
                return lines
            actual_lines = [l for i, l in enumerate(lines) if i not in comment_indices]
            if self.debug_mode:
                self.log_callback(f"[DEBUG] batch:{len(lines)}lines filter->{len(actual_lines)}content")
            translated_actual = self._translate_batch(actual_lines, source_lang, target_lang, model, temperature, max_tokens, num_ctx, game, 0, timeout, mod_folder, output_path, _file_offset)
            if self.stop_event.is_set():
                return lines
            merged = []
            ai = 0
            tgt_code = self._LANG_CODE.get(target_lang, target_lang.lower())
            for i in range(len(lines)):
                if i in comment_indices:
                    l = lines[i]
                    if header_pat.match(l):
                        l = f"l_{tgt_code}:\n"
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

            loc_type = None
            m_loc = re.match(r'^(\d+)\s+', value_part)
            if m_loc:
                loc_type = m_loc.group(1)

            text_for_ph = value_part
            if loc_type:
                text_for_ph = value_part[m_loc.end():]
            if text_for_ph.startswith('"') and text_for_ph.endswith('"'):
                text_for_ph = text_for_ph[1:-1]

            line_phs = []
            def _ph_replacer(text):
                nonlocal ph_counter
                ph = f"{{PH{ph_counter}}}"
                ph_counter += 1
                line_phs.append((ph, text))
                return ph
            cleaned = re.sub(r'\$[^$]+\$', lambda m: _ph_replacer(m.group(0)), text_for_ph)
            cleaned = re.sub(r'\[[^\]]*\]', lambda m: _ph_replacer(m.group(0)), cleaned)
            cleaned = re.sub(r'\u00a7.', lambda m: _ph_replacer(m.group(0)), cleaned)
            cleaned = re.sub(r'\u00a3[^\u00a3]+\u00a3', lambda m: _ph_replacer(m.group(0)), cleaned)
            only_phs = not cleaned.strip() or re.match(r'^(\{PH\d+\}\s*)+\s*$', cleaned.strip())
            if only_phs:
                all_info.append({"type": "keep", "key": key, "ws": ws, "val": value_part, "phs": line_phs, "loc_type": loc_type})
            else:
                send_val = value_part
                for ph_text, orig in line_phs:
                    send_val = send_val.replace(orig, ph_text, 1)
                idx = len(send_data)
                all_info.append({"type": "send", "midx": idx, "key": key, "ws": ws, "loc_type": loc_type})
                send_data.append((f"\u27e8{idx}\u27e9 {send_val}", line_phs, key, ws, loc_type))

        if not send_data:
            return lines

        batch_text = "\n".join(s[0] for s in send_data)
        if self.debug_mode:
            self.log_callback(f"[DEBUG] sending {len(send_data)} lines to LLM: {repr(batch_text[:200])}")
        glossary = self._get_glossary_text(source_lang, target_lang, game, mod_folder)
        if glossary and not getattr(self, '_glossary_logged', False):
            cnt = glossary.count("\n") + 1
            self.log_callback(f"[GLOSSARY] Applied {game} glossary ({cnt} terms)")
            self._glossary_logged = True
        if self.prompt_template:
            base_prompt = self.prompt_template.replace("{source_lang}", source_lang)
            base_prompt = base_prompt.replace("{target_lang}", target_lang)
            if glossary:
                base_prompt = base_prompt.replace("{batch_text}",
                    f"[GLOSSARY]\n{glossary}\n\n[TEXT TO TRANSLATE]\n{batch_text}")
            else:
                base_prompt = base_prompt.replace("{batch_text}", batch_text)
            prompt = get_enhanced_prompt(game, base_prompt)
        else:
            instructions = (
                f"Translate the following text from '{source_lang}' to '{target_lang}'.\n"
                f"Rules:\n1. Preserve all {{PH0}}, {{PH1}}, etc. placeholders exactly as-is.\n"
                f"2. Preserve line markers like \u27e80\u27e9 \u27e81\u27e9 exactly as-is.\n"
                f"3. Do NOT wrap in code blocks or add explanations.\n"
                f"4. Translate proper nouns (person names, place names, character names) based on pronunciation, not meaning.")
            prompt = get_enhanced_prompt(game, instructions)
            if glossary:
                prompt += f"\n\n[GLOSSARY]\n{glossary}"
            prompt += f"\n\n[TEXT TO TRANSLATE]\n{batch_text}"
        result = self._call_ollama(model, prompt, temperature, max_tokens, num_ctx=num_ctx, timeout=timeout)

        if result.startswith("[OLLAMA_"):
            if len(lines) > 1:
                self.log_callback(f"[SPLIT] {result} - splitting batch of {len(lines)} lines")
                mid = len(lines) // 2
                t = min(temperature + 0.05, 1.0)
                first = self._translate_batch(lines[:mid], source_lang, target_lang, model, t, max_tokens, num_ctx, game, timeout=timeout, output_path=output_path, _file_offset=_file_offset)
                if self.stop_event.is_set():
                    return lines
                second = self._translate_batch(lines[mid:], source_lang, target_lang, model, t, max_tokens, num_ctx, game, timeout=timeout, output_path=output_path, _file_offset=_file_offset + mid)
                return first + second
            else:
                if retry_count < self.max_retries:
                    self.log_callback(f"[RETRY {retry_count+1}/{self.max_retries}] {result} - retrying single line")
                    t = min(temperature + 0.1, 1.0)
                    return self._translate_batch(lines, source_lang, target_lang, model, t, max_tokens, num_ctx, game, retry_count + 1, timeout, mod_folder, output_path, _file_offset)
                self.log_callback(f"[FAIL] {result} - returning original after {self.max_retries} retries")
                if self.live_callback:
                    self.live_callback(lines, lines)
                return lines

        result = result.strip()
        result = re.sub(r"```(?:yaml|yml)?\s*\n?", "", result, flags=re.IGNORECASE)
        result = re.sub(r"\n?```", "", result)
        if self.debug_mode:
            self.log_callback(f"[DEBUG] raw response ({len(result)} chars): {repr(result[:300])}")

        translated_values = re.split(r'\u27e8\d+\u27e9\s*', result)
        if translated_values and translated_values[0].strip() == '':
            translated_values = translated_values[1:]
        if self.debug_mode:
            self.log_callback(f"[DEBUG] marker split -> {len(translated_values)} values")

        if len(translated_values) != len(send_data):
            self.log_callback(f"[WARN] LLM returned {len(translated_values)} markers (expected {len(send_data)}), keeping original")
            for i, tv in enumerate(translated_values):
                self.log_callback(f"[WARN]   out[{i}]: {repr(tv[:100])}")
            if output_path:
                for k in range(len(lines)):
                    self._file_warnings.setdefault(output_path, set()).add(_file_offset + k)
            return lines

        reconstructed = list(lines)
        for i, info in enumerate(all_info):
            if info["type"] == "passthrough":
                continue
            elif info["type"] == "keep":
                key, ws, val, phs, loc_type = info["key"], info["ws"], info["val"], info["phs"], info.get("loc_type")
                restored = val
                if restored.startswith('"') and restored.endswith('"'):
                    restored = restored[1:-1]
                for ph, orig in phs:
                    restored = restored.replace(ph, orig)
                restored = restored.replace('\n', '\\n')
                lt = f'{loc_type} ' if loc_type else ' '
                reconstructed[i] = f'{ws}{key}:{lt}"{restored}"\n'
            else:
                _, phs, send_key, send_ws, _ = send_data[info["midx"]]
                raw = translated_values[info["midx"]].strip()
                if raw and raw != '""':
                    lt_inline = None
                    ml = re.match(r'^(\d+)\s+"(.*)"\s*$', raw)
                    if ml:
                        lt_inline, val = ml.group(1), ml.group(2)
                    else:
                        mq = re.match(r'^"(.*)"\s*$', raw)
                        if mq:
                            val = mq.group(1)
                        else:
                            val = raw
                    for ph, orig in phs:
                        val = val.replace(ph, orig)
                    val = self._normalize_text(val)
                    val = val.replace('"', "'")
                    val = val.replace('\n', '\\n')
                    lt = f'{lt_inline} ' if lt_inline else ' '
                    reconstructed[i] = f'{send_ws}{send_key}:{lt}"{val}"\n'

        reconstructed = [re.sub(r'^l_([a-z]+)::\s*$', r'l_\1:', t) for t in reconstructed]
        tgt_code = self._LANG_CODE.get(target_lang, target_lang.lower())
        header_pat = re.compile(r'^l_[a-z_]+:\s*$')
        reconstructed = [f"l_{tgt_code}:\n" if header_pat.match(t) else t for t in reconstructed]
        if self.live_callback:
            self.live_callback(lines, reconstructed)
        return reconstructed

    def _process_file(self, input_path, output_path, source_lang, target_lang, model, temperature, max_tokens, batch_size, num_ctx=4096, game="None", timeout=90, mod_folder="", file_index=1, total_files=1):
        with open(input_path, "rb") as _f:
            raw = _f.read(3)
        has_bom = raw.startswith(b'\xef\xbb\xbf')
        enc = "utf-8-sig" if has_bom else "utf-8"
        with open(input_path, "r", encoding=enc) as f:
            lines = f.readlines()
        if not lines:
            return
        total = len(lines)
        result = []
        first = lines[0]
        id_match = re.match(r"^(l_[a-z]+:)", first)
        if id_match:
            target_code = self._LANG_CODE.get(target_lang, target_lang.lower())
            result.append(f"l_{target_code}:{first[first.index(':') + 1:]}")
            content = lines[1:]
        else:
            result.append(first)
            content = lines[1:]
        skip = 0
        if self._checkpoint_restored and os.path.isfile(output_path):
            try:
                with open(output_path, "r", encoding="utf-8-sig") as f:
                    existing = f.readlines()
                if len(existing) == len(lines):
                    self.log_callback(f"  [RESUME] Output already complete, skipping")
                    return
                if len(existing) > 1:
                    result = existing
                    skip = len(existing) - 1
                    self.log_callback(f"  [RESUME] Resuming from line {skip+2}/{total}")
            except Exception:
                pass
        for i in range(skip, len(content), batch_size):
            if self.stop_event.is_set():
                self.log_callback("[STOPPED] Translation interrupted")
                if len(result) > 1:
                    self._save_checkpoint(result, output_path)
                return
            batch = content[i:i + batch_size]
            self.log_callback(f"  Translating lines {len(result)+1}-{len(result)+len(batch)}/{total}")
            translated = self._translate_batch(batch, source_lang, target_lang, model, temperature, max_tokens, num_ctx, game, timeout=timeout, mod_folder=mod_folder, output_path=output_path, _file_offset=1 + i)
            result.extend(translated)
            self.progress_callback(min(len(result), total), total, file_index, total_files)
            if i % (batch_size * 5) == 0:
                self._save_checkpoint(result, output_path)
        with open(output_path, "w", encoding="utf-8-sig") as f:
            f.writelines(result)
        self.log_callback(f"  Saved: {output_path}")
        # ── Write sidecar .warn file for marker-mismatch failures ──
        warned = self._file_warnings.get(output_path)
        if warned:
            warn_path = output_path + ".warn"
            try:
                with open(warn_path, "w", encoding="utf-8") as f:
                    for ln in sorted(warned):
                        f.write(f"{ln}\n")
            except Exception:
                pass

    def _worker(self, input_dir, output_dir, source_lang, target_lang, model, temperature, max_tokens, batch_size, num_ctx=4096, game="None", timeout=90, mod_folder=""):
        source_code = self._LANG_CODE.get(source_lang, source_lang.lower())
        target_code = self._LANG_CODE.get(target_lang, target_lang.lower())
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
        test = self._call_ollama(model, "test", temperature=0.1, max_tokens=1, num_ctx=num_ctx, timeout=timeout)
        if test.startswith("[OLLAMA_"):
            self.log_callback(f"[ERROR] Cannot connect to Ollama at {self.base_url}")
            self.busy = False
            self.status_callback("idle")
            return

        def _translate_one(filepath, fi, nf):
            if self.stop_event.is_set():
                return
            base = os.path.basename(filepath)
            out_fn = re.sub(re.escape(f"l_{source_code}"), f"l_{target_code}", base, count=1, flags=re.IGNORECASE)
            out_path = os.path.join(output_dir, out_fn)
            self._process_file(filepath, out_path, source_lang, target_lang, model, temperature, max_tokens, batch_size, num_ctx, game, timeout, mod_folder, file_index=fi, total_files=nf)

        nfiles = len(files)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = {executor.submit(_translate_one, f, i+1, nfiles): f for i, f in enumerate(files)}
            concurrent.futures.wait(futures.keys())
        if self.stop_event.is_set():
            self.busy = False
            self.status_callback("idle")
            return
        # Quality check after all files done
        source_code2 = source_code
        target_code2 = target_code
        # Fix: use original source/target codes since we renamed files
        self.log_callback("Running quality check...")
        for fname in os.listdir(output_dir):
            if f"l_{target_code2}" in fname and fname.endswith((".yml", ".yaml")):
                spath = os.path.join(output_dir, fname.replace(f"l_{target_code2}", f"l_{source_code2}", 1))
                tpath = os.path.join(output_dir, fname)
                if os.path.isfile(spath):
                    issues = self.check_quality(spath, tpath, source_lang, target_lang)
                    if issues:
                        self.log_callback(f"  ⚠ {fname}: {issues}")
        self.log_callback("✓ All files processed")
        if not self.stop_event.is_set():
            cp_dir = os.path.join(os.path.dirname(CONFIG_FILE), "checkpoint")
            if os.path.isdir(cp_dir) and self.checkpoint_enabled:
                for fn in os.listdir(cp_dir):
                    if fn.endswith((".yml", ".yaml")):
                        try:
                            os.remove(os.path.join(cp_dir, fn))
                            self.log_callback(f"[CHECKPOINT] Cleaned: {fn}")
                        except Exception:
                            pass
        self.busy = False
        self.status_callback("idle")

    @staticmethod
    def _glossary_dir():
        if getattr(sys, 'frozen', False):
            base = os.path.dirname(sys.executable)
        else:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        gd = os.path.join(base, "glossary")
        os.makedirs(gd, exist_ok=True)
        return gd

    def _get_glossary_text(self, source_lang, target_lang, game="None", mod_folder=""):
        if game == "None" or not game:
            return ""
        base_dir = OllamaTranslator._glossary_dir()
        game_dir = os.path.join(base_dir, game)
        if not os.path.isdir(game_dir):
            return ""
        src_code = OllamaTranslator._LANG_CODE.get(source_lang, source_lang.lower())
        tgt_code = OllamaTranslator._LANG_CODE.get(target_lang, target_lang.lower())
        suffix = f"_{src_code}_{tgt_code}.txt".lower()
        exact_name = f"{src_code}_{tgt_code}.txt"
        # ── Derive mod name from folder to filter matching glossary files ──
        mod_prefix = ""
        if mod_folder:
            from ollama_translator.utils import _derive_modname
            mod_prefix = _derive_modname(mod_folder).lower() + "_"
        terms = []
        seen = set()
        files = sorted(os.listdir(game_dir))
        for fn in files:
            if not fn.endswith(".txt"):
                continue
            fn_lower = fn.lower()
            if fn_lower == exact_name.lower():
                is_mod = False
            elif fn_lower.endswith(suffix) and fn_lower.startswith(mod_prefix):
                is_mod = True
            else:
                continue
            fpath = os.path.join(game_dir, fn)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception:
                continue
            limit = 200 if not is_mod else 100
            for line in lines[:limit]:
                parts = line.strip().split(":", 1)
                if len(parts) == 2:
                    k, v = parts[0].strip(), parts[1].strip()
                    if k not in seen:
                        seen.add(k)
                        terms.append(f"{k} -> {v}")
            if len(terms) >= self.glossary_limit:
                break
        if not terms:
            return ""
        return "\n".join(terms[:self.glossary_limit])

    @staticmethod
    def _strip_codes(text):
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\$[^\$]+\$', '', text)
        text = text.replace('§', '')
        return text.strip()

    @staticmethod
    def _find_duplicate_keys(lines):
        keys = {}
        for i, line in enumerate(lines, 1):
            m = re.match(r'^\s*([\w.]+):\s*\d+\s*["\[]', line)
            if m:
                keys.setdefault(m.group(1), []).append(i)
        return {k: v for k, v in keys.items() if len(v) > 1}

    TEXT_GROUPS = {
        "latin": re.compile(r'[\u0000-\u024F]'),
        "hangul": re.compile(r'[\uAC00-\uD7AF]'),
        "cjk": re.compile(r'[\u4E00-\u9FFF]'),
        "cyrillic": re.compile(r'[\u0400-\u04FF]'),
    }
    ALLOWED_GROUPS = {
        "english": {"latin"},
        "korean": {"latin", "hangul"},
        "simp_chinese": {"latin", "cjk"},
        "french": {"latin"},
        "german": {"latin"},
        "spanish": {"latin"},
        "japanese": {"latin", "cjk"},
        "russian": {"latin", "cyrillic"},
        "polish": {"latin"},
        "braz_por": {"latin"},
    }

    def _has_foreign_chars(self, text, target_lang):
        tgt_code = self._LANG_CODE.get(target_lang, target_lang.lower())
        allowed = self.ALLOWED_GROUPS.get(tgt_code, {"latin"})
        for ch in text:
            for gname, grex in self.TEXT_GROUPS.items():
                if grex.match(ch):
                    if gname not in allowed:
                        return True
                    break
        return False

    def check_quality(self, src_path, tgt_path, source_lang, target_lang):
        issues = []
        try:
            with open(src_path, "r", encoding="utf-8-sig") as f:
                src_lines = [l.rstrip("\n") for l in f.readlines()]
            with open(tgt_path, "r", encoding="utf-8-sig") as f:
                tgt_lines = [l.rstrip("\n") for l in f.readlines()]
        except FileNotFoundError:
            return issues
        dups = self._find_duplicate_keys(tgt_lines)
        min_len = min(len(src_lines), len(tgt_lines))
        if len(src_lines) != len(tgt_lines):
            issues.append((0, f"Line count: src={len(src_lines)} tgt={len(tgt_lines)}", "", "MISMATCH", ""))
        for i in range(1, min_len):
            s = src_lines[i]
            t = tgt_lines[i]
            if not s.strip() or s.strip().startswith('#'):
                continue
            m = re.match(r'^\s*([\w.]+):\s*', s)
            if not m:
                continue
            key = m.group(1)
            sv = re.match(r'^\s*[\w.]+:\d+\s*"(.+)"', s)
            tv = re.match(r'^\s*[\w.]+:\d+\s*"(.+)"', t)
            if not sv or not tv:
                continue
            s_val, t_val = sv.group(1), tv.group(1)
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

    def start(self, input_dir, output_dir, source_lang, target_lang, model, temperature, max_tokens, batch_size, num_ctx=4096, game="None", max_retries=3, timeout=90, mod_folder=""):
        self.busy = True
        self.stop_event.clear()
        self.max_retries = max_retries
        self._consecutive_errors = 0
        self._glossary_logged = False
        def _run():
            try:
                self._worker(input_dir, output_dir, source_lang, target_lang, model, temperature, max_tokens, batch_size, num_ctx, game, timeout, mod_folder)
            except Exception as e:
                for line in traceback.format_exc().split("\n"):
                    self.log_callback(line)
                self.busy = False
                self.status_callback("idle")
        threading.Thread(target=_run, daemon=True).start()

    def stop(self):
        self.stop_event.set()
        self.busy = False
