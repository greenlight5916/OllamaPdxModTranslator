import os, sys, json, time, re, threading, subprocess, concurrent.futures, traceback
import requests

def _app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(_app_dir(), "ollama_translator_config.json")

class OllamaTranslator:
    def __init__(self, log_callback=None, progress_callback=None, status_callback=None, live_callback=None):
        self.log_callback = log_callback or (lambda x: None)
        self.progress_callback = progress_callback or (lambda x: None)
        self.status_callback = status_callback or (lambda x: None)
        self.live_callback = live_callback
        self.base_url = "http://localhost:11434"
        self.stop_event = threading.Event()
        self.busy = False
        self.process = None
        self._fatal_errors = 0
        self.config = {}
        if os.path.isfile(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as _f:
                    self.config = json.load(_f)
            except Exception:
                self.config = {}
        self._checkpoint_dir = "checkpoints"

    def set_base_url(self, url):
        self.base_url = url

    def start_server(self):
        if self.process and self.process.poll() is None:
            return True
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return True
        except Exception:
            pass
        try:
            self.process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            for _ in range(30):
                time.sleep(1)
                try:
                    requests.get(f"{self.base_url}/api/tags", timeout=2)
                    return True
                except Exception:
                    continue
            return False
        except Exception:
            return False

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
        return []

    def get_running_models(self):
        try:
            r = requests.get(f"{self.base_url}/api/ps", timeout=5)
            if r.status_code == 200:
                data = r.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def test_model(self, model):
        mc = self._call_ollama(model, "test", temperature=0.1, max_tokens=1)
        return not mc.startswith("[OLLAMA_")

    def _check_fatal(self):
        self._fatal_errors += 1
        if self._fatal_errors >= 5:
            self.stop_event.set()

    def _call_ollama(self, model, prompt, temperature=0.1, max_tokens=8192):
        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens}
                },
                timeout=600
            )
            if r.status_code == 200:
                return r.json().get("message", {}).get("content", "")
            return f"[OLLAMA_ERROR:{r.status_code}]"
        except requests.exceptions.ConnectionError:
            return "[OLLAMA_CONNECTION_ERROR]"
        except Exception as e:
            return f"[OLLAMA_ERROR:{e}]"

    def _save_checkpoint(self, path, lines):
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        cp_path = os.path.join(self._checkpoint_dir, os.path.basename(path) + ".ckp")
        with open(cp_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    _LANG_CODE = {
        "English":"english","Korean":"korean","Simplified Chinese":"simp_chinese",
        "French":"french","German":"german","Spanish":"spanish",
        "Japanese":"japanese","Russian":"russian","Polish":"polish",
        "Brazilian Portuguese":"braz_por",
    }

    def _translate_batch(self, batch, source_lang, target_lang, model, temperature, max_tokens, game="None"):
        target_code = self._LANG_CODE.get(target_lang, target_lang.lower())
        base_prompt = (
            "Translate each line from English to the target language.\n"
            "Output format: each original line followed by a tab, then the translation.\n"
            "Preserve all placeholders, brackets, and special formatting exactly.\n"
            "Keep the original line unchanged, only add the translation after a tab.\n"
            "Do not skip any lines.\n"
        )
        glossary = self._get_glossary_text(source_lang, target_lang, game)
        if glossary:
            base_prompt += "\n[REFERENCE GLOSSARY]\n" + glossary + "\n[/REFERENCE GLOSSARY]"
        sys_msg = {"role": "system", "content": base_prompt}
        batch_lines = "\n".join([f"{i}: {line}" for i, line in enumerate(batch)])
        user_msg = {"role": "user", "content": f"[BATCH START]\n{batch_lines}\n[BATCH END]"}
        for attempt in range(5):
            resp = self._call_ollama(model, f"{base_prompt}\n\n{batch_lines}", temperature=temperature, max_tokens=max_tokens)
            if resp.startswith("[OLLAMA_"):
                self.log_callback(f"[RETRY {attempt+1}] {resp}")
                time.sleep(2)
                continue
            translated = {}
            for line in resp.split("\n"):
                if "\t" in line:
                    parts = line.split("\t", 1)
                    try:
                        idx = int(parts[0].strip())
                        translated[idx] = parts[1].strip()
                    except ValueError:
                        pass
            if translated:
                break
        result = []
        for i, line in enumerate(batch):
            if i in translated and translated[i].strip():
                stripped = self._strip_codes(translated[i].strip())
                if stripped and stripped[-1] in ('.', '!', '?', '~', '"', ')', ']', '}', '>'):
                    result.append(f'{line}\t{translated[i].strip()}')
                else:
                    result.append(f'{line}\t{translated[i].strip()}')
            else:
                result.append(line)
        if self.live_callback:
            reconstructed = []
            for i, line in enumerate(result):
                if "\t" in line:
                    orig, trans = line.split("\t", 1)
                    reconstructed.append(f'{orig}\t{trans}')
                else:
                    reconstructed.append(line)
            self.live_callback(batch, reconstructed)
        return result

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
            target_code = self._LANG_CODE.get(target_lang, target_lang.lower())
            result.append(f"l_{target_code}:{first[first.index(':') + 1:]}")
            content = lines[1:]
        else:
            result.append(first)
            content = lines[1:]
        for i in range(0, len(content), batch_size):
            if self.stop_event.is_set():
                self.log_callback("[STOPPED] Translation interrupted")
                if len(result) > 1:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.writelines(result)
                return
            batch = content[i:i + batch_size]
            batch_result = self._translate_batch(
                [x.rstrip("\n") for x in batch],
                source_lang, target_lang, model, temperature, max_tokens, game
            )
            result.extend([x + "\n" for x in batch_result])
            progress = min((i + len(batch)) / total, 1.0)
            self.progress_callback(progress)
            if (i + len(batch)) % (batch_size * 5) < batch_size:
                self._save_checkpoint(output_path, result)
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(result)
        self.log_callback(f"✓ Saved {output_path}")

    def _worker(self, input_dir, output_dir, source_lang, target_lang, model, temperature, max_tokens, batch_size, game="None"):
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
        test = self._call_ollama(model, "test", temperature=0.1, max_tokens=1)
        if test.startswith("[OLLAMA_"):
            self.log_callback(f"[ERROR] Cannot connect to Ollama at {self.base_url}")
            self.busy = False
            self.status_callback("idle")
            return

        def _translate_one(filepath):
            if self.stop_event.is_set():
                return
            base = os.path.basename(filepath)
            out_fn = base.replace(f"l_{source_code}", f"l_{target_code}", 1) if f"l_{source_code}" in base.lower() else base
            out_path = os.path.join(output_dir, out_fn)
            self._process_file(filepath, out_path, source_lang, target_lang, model, temperature, max_tokens, batch_size, game)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = {executor.submit(_translate_one, f): f for f in files}
            concurrent.futures.wait(futures.keys())
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
                    issues = self.check_quality(spath, tpath)
                    if issues:
                        self.log_callback(f"  ⚠ {fname}: {issues}")
        self.log_callback("✓ All files processed")
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

    @staticmethod
    def _get_glossary_text(source_lang, target_lang, game="None"):
        if game == "None" or not game:
            return ""
        base_dir = OllamaTranslator._glossary_dir()
        game_dir = os.path.join(base_dir, game)
        if not os.path.isdir(game_dir):
            return ""
        src_code = OllamaTranslator._LANG_CODE.get(source_lang, source_lang.lower())
        tgt_code = OllamaTranslator._LANG_CODE.get(target_lang, target_lang.lower())
        fname = f"{src_code}_{tgt_code}.txt"
        fpath = os.path.join(game_dir, fname)
        if not os.path.isfile(fpath):
            fpath_lower = os.path.join(game_dir, fname.lower())
            if os.path.isfile(fpath_lower):
                fpath = fpath_lower
            else:
                for fn in os.listdir(game_dir):
                    if fn.lower() == f"{src_code}_{tgt_code}.txt".lower() and fn.endswith(".txt"):
                        fpath = os.path.join(game_dir, fn)
                        break
                else:
                    return ""
        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        terms = []
        for line in lines[:100]:
            parts = line.strip().split(":", 1)
            if len(parts) == 2:
                terms.append(f"{parts[0].strip()} -> {parts[1].strip()}")
        return "\n".join(terms)

    @staticmethod
    def _strip_codes(text):
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\$[^\$]+\$', '', text)
        text = text.replace('§', '')
        return text.strip()

    @staticmethod
    def _find_duplicate_keys(path):
        seen = {}
        dupes = {}
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                m = re.match(r"^([^:]+):\s*", line)
                if m:
                    key = m.group(1).strip().lower()
                    if key in seen:
                        dupes[key] = (seen[key], line.strip())
                    else:
                        seen[key] = line.strip()
        return dupes

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

    def check_quality(self, src_path, tgt_path):
        issues = []
        if not os.path.isfile(src_path) or not os.path.isfile(tgt_path):
            return ["Missing file"]
        with open(src_path, "r", encoding="utf-8-sig") as f:
            src_lines = f.readlines()
        with open(tgt_path, "r", encoding="utf-8-sig") as f:
            tgt_lines = f.readlines()
        if len(src_lines) != len(tgt_lines):
            issues.append(f"Line count mismatch: src={len(src_lines)} tgt={len(tgt_lines)}")
        for i, (sl, tl) in enumerate(zip(src_lines, tgt_lines)):
            tl_content = tl.split("\t", 1)[-1].strip() if "\t" in tl else tl.strip()
            if tl_content and tl_content == sl.strip():
                issues.append(f"Line {i+1}: untranslated")
            if tl_content and self._has_foreign_chars(tl_content, "english"):
                issues.append(f"Line {i+1}: foreign chars")
        dupes = self._find_duplicate_keys(tgt_path)
        if dupes:
            for k, (l1, l2) in dupes.items():
                issues.append(f"Duplicate key '{k}': {l1} and {l2}")
        return issues

    def start(self, input_dir, output_dir, source_lang, target_lang, model, temperature, max_tokens, batch_size, game="None"):
        self.busy = True
        self.stop_event.clear()
        self._fatal_errors = 0
        def _run():
            try:
                self._worker(input_dir, output_dir, source_lang, target_lang, model, temperature, max_tokens, batch_size, game)
            except Exception as e:
                for line in traceback.format_exc().split("\n"):
                    self.log_callback(line)
                self.busy = False
                self.status_callback("idle")
        threading.Thread(target=_run, daemon=True).start()

    def stop(self):
        self.stop_event.set()
        self.busy = False
