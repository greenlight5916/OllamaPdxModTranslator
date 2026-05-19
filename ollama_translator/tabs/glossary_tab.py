import os, json, re, threading, traceback
import customtkinter as ctk
from tkinter import filedialog

from ollama_translator.engine import OllamaTranslator
from ollama_translator.utils import (
    _PREFIX_TO_LANG, PLATFORM_PATTERNS,
    tokenize_game_en, tokenize_tgt_pure, parse_yml,
    tokenize_mod_en, load_stopwords, _glossary_dir, _derive_modname
)

class GlossaryTabMixin:

    # ── Game detection ──

    def _detect_game_from_path(self, path):
        for pat in PLATFORM_PATTERNS:
            m = re.search(pat, path, re.IGNORECASE)
            if m:
                g = m.group(1)
                if g in self.available_games:
                    return g
        return None

    # ── 인 게임 용어로 통일하기 위해 개발했으나 현재 개발중단 ──
    # def _g_browse(self):
    #     d = filedialog.askdirectory()
    #     if not d:
    #         return
    #     self._g_folder_var.set(d)
    #     game = self._detect_game_from_path(d)
    #     if game:
    #         self.selected_game.set(game)
    #     langs = self._detect_languages(d)
    #     if langs:
    #         self.source_lang.set(langs[0] if langs[0] else "English")
    #         if len(langs) > 1:
    #             self.target_lang.set(langs[1])
    #         elif langs[0] == "English":
    #             self.target_lang.set("Korean")

    def _detect_languages(self, folder):
        langs = set()
        for entry in os.listdir(folder):
            epath = os.path.join(folder, entry)
            if os.path.isdir(epath):
                for code, lang in _PREFIX_TO_LANG.items():
                    if entry.lower() == code[len("l_"):]:
                        langs.add(lang)
            elif entry.endswith((".yml", ".yaml")):
                for code, lang in _PREFIX_TO_LANG.items():
                    if code in entry.lower():
                        langs.add(lang)
        return sorted(langs, key=lambda x: self.available_langs.index(x) if x in self.available_langs else 99)

    def _m_browse(self):
        path = filedialog.askdirectory()
        if not path:
            return
        self._m_folder_var.set(path)
        self.input_dir.set(path)
        self._m_modname = _derive_modname(path)
        game = self._detect_game_from_path(path)
        if game:
            self.selected_game.set(game)
        langs = self._detect_languages(path)
        if langs:
            self.source_lang.set(langs[0] if langs[0] else "English")
            if len(langs) > 1:
                self.target_lang.set(langs[1])
            elif langs and langs[0] == "English":
                self.target_lang.set("Korean")

    # ── 인 게임 용어로 통일하기 위해 개발했으나 현재 개발중단 ──
    # # ── Glossary GAME tab ──
    # 
    # def _game_extract(self):
    #     if getattr(self, '_g_running', False):
    #         return
    #     folder = self._g_folder_var.get()
    #     src_lang = self.source_lang.get()
    #     tgt_lang = self.target_lang.get()
    #     min_freq = int(self._g_min_var.get())
    #     if not folder or not os.path.isdir(folder):
    #         self.log("[ERROR] Select a game folder first")
    #         return
    #     loc_dir = folder
    #     for sub in ("localisation", "localization"):
    #         p = os.path.join(folder, sub)
    #         if os.path.isdir(p):
    #             loc_dir = p
    #             break
    #     def _run():
    #         try:
    #             self._g_running = True
    #             self.after(0, lambda: self._g_progress.set(0))
    #             self.after(0, lambda: self._g_info_label.configure(text="Scanning files..."))
    #             all_files = {}
    #             for root, _, fnames in os.walk(loc_dir):
    #                 for fn in fnames:
    #                     if not fn.endswith((".yml", ".yaml")):
    #                         continue
    #                     for code, lang in _PREFIX_TO_LANG.items():
    #                         lower_fn = fn.lower()
    #                         if code in lower_fn:
    #                             idx = lower_fn.index(code)
    #                             base = fn[:idx] + fn[idx + len(code):]
    #                             all_files.setdefault(base, {})[lang] = os.path.join(root, fn)
    #                             break
    #             pairs = []
    #             for base, lang_map in all_files.items():
    #                 if src_lang in lang_map and tgt_lang in lang_map:
    #                     pairs.append((lang_map[src_lang], lang_map[tgt_lang]))
    #             if not pairs:
    #                 self.after(0, lambda: self.log("[ERROR] No paired language files found"))
    #                 return
    #             total_pairs = len(pairs)
    #             self.after(0, lambda: self.log(f"Found {total_pairs} file pairs. Extracting..."))
    #             src_freq = {}
    #             tgt_freq = {}
    #             cooccur = {}
    #             for pi, (sp, tp) in enumerate(pairs):
    #                 if getattr(self, '_g_running', False) is False:
    #                     break
    #                 src_data = parse_yml(open(sp, "r", encoding="utf-8-sig").read())
    #                 tgt_data = parse_yml(open(tp, "r", encoding="utf-8-sig").read())
    #                 common = set(src_data.keys()) & set(tgt_data.keys())
    #                 for key in common:
    #                     src_tokens = tokenize_game_en(src_data[key], f"l_{OllamaTranslator._LANG_CODE.get(src_lang, src_lang)}")
    #                     tgt_tokens = tokenize_tgt_pure(tgt_data[key])
    #                     tgt_tokens = [t for t in tgt_tokens if t not in load_stopwords(f"l_{OllamaTranslator._LANG_CODE.get(tgt_lang, tgt_lang)}")]
    #                     if not src_tokens or not tgt_tokens:
    #                         continue
    #                     for st in src_tokens:
    #                         src_freq[st] = src_freq.get(st, 0) + 1
    #                         if st not in cooccur:
    #                             cooccur[st] = {}
    #                         for tt in tgt_tokens:
    #                             tgt_freq[tt] = tgt_freq.get(tt, 0) + 1
    #                             cooccur[st][tt] = cooccur[st].get(tt, 0) + 1
    #                 self.after(0, lambda p=pi+1, t=total_pairs: (
    #                     self._g_progress.set(p/t if t > 0 else 0),
    #                     self._g_info_label.configure(text=f"Scanning {p}/{t}...")
    #                 ))
    #             dice_best = {}
    #             for st, td in cooccur.items():
    #                 sf = src_freq.get(st, 0)
    #                 if sf < min_freq:
    #                     continue
    #                 for tt, c in td.items():
    #                     tf = tgt_freq.get(tt, 0)
    #                     dice = 2 * c / (sf + tf) if (sf + tf) > 0 else 0
    #                     if st not in dice_best or dice > dice_best[st][1]:
    #                         dice_best[st] = (tt, dice)
    #             tgt_dup = {}
    #             for st, (tt, d) in dice_best.items():
    #                 if tt not in tgt_dup or d > tgt_dup[tt][1]:
    #                     tgt_dup[tt] = (st, d)
    #             deduped = {}
    #             for tt, (st, d) in tgt_dup.items():
    #                 deduped[st] = (tt, d)
    #             sorted_entries = sorted(deduped.items(), key=lambda x: -x[1][1])
    #             self._g_entries = [(st, tt, int(src_freq.get(st, 0))) for st, (tt, _) in sorted_entries]
    #             self._g_checked = {}
    #             self._g_page = 0
    #             self.after(0, lambda: self._game_update_display())
    #             self.after(0, lambda: self._g_info_label.configure(text=f"Extracted {len(self._g_entries)} terms"))
    #             self.after(0, lambda: self.log(f"[GLOSSARY] Extracted {len(self._g_entries)} term pairs"))
    #             self._g_dirty = False
    #         except Exception as e:
    #             tb = traceback.format_exc()
    #             self.after(0, lambda: self.log(f"[ERROR] _game_extract:\n{tb}"))
    #         finally:
    #             self._g_running = False
    #     threading.Thread(target=_run, daemon=True).start()
    # 
    # def _g_per_page(self):
    #     v = self._g_per_page_var.get().strip().lower()
    #     if v == "all":
    #         return 10**9
    #     try:
    #         return max(1, int(v))
    #     except (ValueError, TypeError):
    #         return 20
    # 
    # def _game_update_display(self, filter_text=""):
    #     for w in self._g_result_frame.winfo_children():
    #         w.destroy()
    #     entries = self._g_entries if hasattr(self, '_g_entries') else []
    #     if filter_text:
    #         entries = [e for e in entries if filter_text.lower() in e[0].lower()]
    #     total = len(entries)
    #     pp = self._g_per_page()
    #     start = self._g_page * pp
    #     end = start + pp
    #     page_entries = entries[start:end]
    #     checked = getattr(self, '_g_checked', {})
    #     for idx, (src_tok, tgt_tok, freq) in enumerate(page_entries):
    #         row = idx
    #         ctk.CTkCheckBox(self._g_result_frame, text="").grid(row=row, column=0, padx=2)
    #         ctk.CTkLabel(self._g_result_frame, text=src_tok, anchor="w").grid(row=row, column=1, sticky="w", padx=5)
    #         trans = checked.get(src_tok, tgt_tok)
    #         e = ctk.CTkEntry(self._g_result_frame, width=300)
    #         e.grid(row=row, column=2, sticky="ew", padx=5)
    #         e.insert(0, trans)
    #         e.bind("<KeyRelease>", lambda _: setattr(self, '_g_dirty', True))
    #         ctk.CTkLabel(self._g_result_frame, text=str(freq), anchor="e", width=40).grid(row=row, column=3, sticky="e", padx=5)
    #     total_pages = (total - 1) // pp + 1 if total > 0 else 1
    #     self._g_page_label.configure(text=f"Page {self._g_page+1}/{total_pages} ({total} terms)")
    #     self._g_info_label.configure(text=f"{total} terms" if total else "No terms")
    # 
    # def _g_prev_page(self):
    #     if self._g_page > 0:
    #         self._g_page -= 1
    #         self._game_update_display(self._g_search_var.get())
    # 
    # def _g_next_page(self):
    #     entries = self._g_entries if hasattr(self, '_g_entries') else []
    #     if (self._g_page + 1) * self._g_per_page() < len(entries):
    #         self._g_page += 1
    #         self._game_update_display(self._g_search_var.get())
    # 
    # def _game_save(self):
    #     game = self.selected_game.get()
    #     src = self.source_lang.get()
    #     tgt = self.target_lang.get()
    #     if not game:
    #         self.log("[ERROR] No game specified")
    #         return
    #     src_code = OllamaTranslator._LANG_CODE.get(src, src.lower())
    #     tgt_code = OllamaTranslator._LANG_CODE.get(tgt, tgt.lower())
    #     fname = f"{src_code}_{tgt_code}.txt".lower()
    #     base_dir = _glossary_dir()
    #     gd = os.path.join(base_dir, game)
    #     os.makedirs(gd, exist_ok=True)
    #     path = os.path.join(gd, fname)
    #     checked = getattr(self, '_g_checked', {})
    #     with open(path, "w", encoding="utf-8") as f:
    #         for src_tok, tgt_tok, freq in self._g_entries:
    #             val = checked.get(src_tok, tgt_tok)
    #             f.write(f"{src_tok}: {val}\n")
    #     self._g_dirty = False
    #     self.log(f"[GLOSSARY] Saved {len(self._g_entries)} terms to {path}")
    # 
    # def _game_load(self):
    #     game = self.selected_game.get()
    #     src = self.source_lang.get()
    #     tgt = self.target_lang.get()
    #     if not game:
    #         self.log("[ERROR] No game specified")
    #         return
    #     src_code = OllamaTranslator._LANG_CODE.get(src, src.lower())
    #     tgt_code = OllamaTranslator._LANG_CODE.get(tgt, tgt.lower())
    #     fname = f"{src_code}_{tgt_code}.txt".lower()
    #     base_dir = _glossary_dir()
    #     path = os.path.join(base_dir, game, fname)
    #     if not os.path.isfile(path):
    #         self.log(f"[ERROR] No glossary file found at {path}")
    #         return
    #     try:
    #         checked = {}
    #         with open(path, "r", encoding="utf-8") as f:
    #             for line in f:
    #                 parts = line.strip().split(":", 1)
    #                 if len(parts) == 2:
    #                     checked[parts[0].strip()] = parts[1].strip()
    #         self._g_checked = checked
    #         self._g_entries = [(k, v, 0) for k, v in checked.items()]
    #         self._g_page = 0
    #         self._game_update_display()
    #         self._g_dirty = False
    #         self.log(f"[GLOSSARY] Loaded {len(checked)} terms from: {path}")
    #     except Exception as e:
    #         self.log(f"[ERROR] Load failed: {e}")
    # 
    # # ── Validate popup ──
    # 
    # def _game_validate(self):
    #     if not hasattr(self, '_g_entries') or not self._g_entries:
    #         self.log("[ERROR] No glossary entries to validate. Extract or load first.")
    #         return
    #     src_display = self.source_lang.get()
    #     tgt_display = self.target_lang.get()
    #     game = self.selected_game.get()
    #     entries = list(self._g_entries)
    #     checked = dict(getattr(self, '_g_checked', {}))
    #     src_code = OllamaTranslator._LANG_CODE.get(src_display, src_display.lower())
    #     tgt_code = OllamaTranslator._LANG_CODE.get(tgt_display, tgt_display.lower())
    #     cache_path = os.path.join(_glossary_dir(), game, f"_validate_{src_code}_{tgt_code}.json")
    #     self._validate_cache_path = cache_path
    #     self._validate_game = game
    #     self._validate_src_display = src_display
    #     self._validate_tgt_display = tgt_display
    #     if os.path.isfile(cache_path):
    #         try:
    #             with open(cache_path, "r", encoding="utf-8") as f:
    #                 cached = json.load(f)
    #             self.log(f"[VALIDATE] Loaded {len(cached)} cached validation results for {game}")
    #             self._g_validate_win(entries, checked, cached)
    #             self._g_validate_done(cached)
    #             return
    #         except Exception:
    #             self.log("[VALIDATE] Cache load failed")
    #     self._g_validate_win(entries, checked, {})
    # 
    # def _g_validate_run_llm(self):
    #     if not hasattr(self, '_validate_toplevel') or not self._validate_toplevel:
    #         return
    #     model = self.ollama_model.get()
    #     if not model or model == "(none)":
    #         self.log("[ERROR] Select a model in the Translate tab first")
    #         return
    #     entries = self._validate_entries
    #     checked = self._validate_checked
    #     tgt_display = self._validate_tgt_display
    #     game = self._validate_game
    #     cache_path = self._validate_cache_path
    #     batch_size = max(1, self.batch_size.get())
    #     self.engine.set_base_url(self.ollama_url.get())
    #     self.log(f"[VALIDATE] Validating {len(entries)} terms with {model} ({batch_size}/batch)...")
    #     self._validate_info.configure(text="Starting LLM validation...")
    #     self._validate_progress.set(0)
    #     def _remove_btn():
    #         for w in self._validate_toplevel.grid_slaves():
    #             try:
    #                 if int(w.grid_info()["row"]) == 3:
    #                     w.destroy()
    #             except Exception:
    #                 pass
    #     _remove_btn()
    #     def _run():
    #         try:
    #             all_results = {}
    #             total_batches = (len(entries) + batch_size - 1) // batch_size
    #             for bi in range(0, len(entries), batch_size):
    #                 batch = entries[bi:bi + batch_size]
    #                 batch_num = bi // batch_size + 1
    #                 prompt = (
    #                     f"Review these English -> {tgt_display} glossary terms for the game '{game}'.\n"
    #                     "For each term, reply in this exact format:\n"
    #                     "term: OK\n"
    #                     "term: REJECT:suggested_translation\n\n"
    #                     "- OK = translation is approximately correct for game context.\n"
    #                     "- REJECT = translation is completely wrong or unrelated.\n"
    #                     "  Provide a suggested correct translation after the second colon.\n\n"
    #                 )
    #                 for e in batch:
    #                     cur = checked.get(e[0], e[1])
    #                     prompt += f"{e[0]} -> {cur}\n"
    #                 result = self.engine._call_ollama(model, prompt, temperature=0.1, max_tokens=4096)
    #                 for line in result.split("\n"):
    #                     line = line.strip()
    #                     if ":" in line and not line.startswith("```") and not line.startswith("#"):
    #                         parts = line.split(":", 1)
    #                         src = parts[0].strip().lower()
    #                         rest = parts[1].strip()
    #                         rest_up = rest.upper()
    #                         if rest_up.startswith("OK"):
    #                             all_results[src] = ("OK", "")
    #                         elif rest_up.startswith("REJECT"):
    #                             sug = rest.split(":", 1)[1].strip() if ":" in rest else ""
    #                             all_results[src] = ("REJECT", sug)
    #                 self.after(0, lambda p=batch_num, t=total_batches: (
    #                     self._update_validate_progress(p / t if t > 0 else 0, f"Validating {p}/{t}..."),
    #                     self._g_progress.set(p / t if t > 0 else 0)
    #                 ))
    #             try:
    #                 os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    #                 with open(cache_path, "w", encoding="utf-8") as f:
    #                     json.dump(all_results, f)
    #             except Exception:
    #                 pass
    #             self.after(0, lambda: self._g_validate_done(all_results))
    #         except Exception as e:
    #             tb = traceback.format_exc()
    #             self.after(0, lambda e=e, tb=tb: self.log(f"[ERROR] _g_validate_run_llm:\n{tb}"))
    #             self.after(0, lambda: self._validate_info.configure(text="LLM validation failed"))
    #     threading.Thread(target=_run, daemon=True).start()
    # 
    # def _g_validate_win(self, entries, checked, llm_results):
    #     if hasattr(self, '_validate_toplevel') and self._validate_toplevel:
    #         try:
    #             self._validate_toplevel.destroy()
    #         except Exception:
    #             pass
    #     win = ctk.CTkToplevel(self)
    #     win.title(f"Validate Glossary - {self.selected_game.get()}")
    #     win.geometry("900x650")
    #     win.minsize(600, 400)
    #     win.transient(self)
    #     win.protocol("WM_DELETE_WINDOW", self._g_validate_cancel)
    #     self._validate_toplevel = win
    #     win.grid_columnconfigure(0, weight=1)
    #     win.grid_rowconfigure(2, weight=1)
    #     progress = ctk.CTkProgressBar(win)
    #     progress.grid(row=0, column=0, padx=10, pady=(10, 2), sticky="ew")
    #     progress.set(0)
    #     self._validate_progress = progress
    #     info = ctk.CTkLabel(win, text="Waiting for LLM response...")
    #     info.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="w")
    #     self._validate_info = info
    #     frame = ctk.CTkScrollableFrame(win)
    #     frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
    #     frame.grid_columnconfigure(2, weight=1)
    #     frame.grid_columnconfigure(3, weight=0)
    #     self._validate_frame = frame
    #     self._validate_entries = entries
    #     self._validate_checked = dict(checked)
    #     self._validate_llm = dict(llm_results)
    #     self._validate_row_widgets = []
    #     if not llm_results:
    #         self._validate_info.configure(text="No cached validation results.")
    #         btn = ctk.CTkButton(win, text="Run LLM Validation", fg_color="#7B1FA2",
    #                             command=self._g_validate_run_llm, height=40, font=ctk.CTkFont(size=14, weight="bold"))
    #         btn.grid(row=3, column=0, padx=10, pady=10)
    # 
    # def _update_validate_progress(self, val, text):
    #     if hasattr(self, '_validate_progress') and self._validate_progress:
    #         self._validate_progress.set(val)
    #     if hasattr(self, '_validate_info') and self._validate_info:
    #         self._validate_info.configure(text=text)
    # 
    # def _g_validate_g_per_page(self):
    #     v = self._validate_g_per_page_var.get().strip().lower()
    #     if v == "all":
    #         return 10**9
    #     try:
    #         return max(1, int(v))
    #     except (ValueError, TypeError):
    #         return 20
    # 
    # def _g_validate_done(self, llm_results):
    #     if not hasattr(self, '_validate_toplevel') or not self._validate_toplevel:
    #         return
    #     self._validate_llm = llm_results
    #     entries = self._validate_entries
    #     checked = self._validate_checked
    #     reject_count = sum(1 for v in llm_results.values() if isinstance(v, (list, tuple)) and v[0] == "REJECT")
    #     show_all_var = ctk.BooleanVar(value=False)
    #     self._validate_show_all = show_all_var
    #     def _render(*_):
    #         for w in self._validate_frame.winfo_children():
    #             w.destroy()
    #         self._validate_row_widgets = []
    #         ff = ctk.CTkFrame(self._validate_frame, fg_color="transparent")
    #         ff.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 2))
    #         ctk.CTkCheckBox(ff, text="Show all terms (including OK)", variable=show_all_var, command=_render).pack(side="left", padx=5)
    #         lbl_text = f"({reject_count} REJECTED)" if reject_count else "(All OK)"
    #         lbl_color = "orange" if reject_count else "green"
    #         ctk.CTkLabel(ff, text=lbl_text, text_color=lbl_color).pack(side="left", padx=5)
    #         cf = ctk.CTkFrame(self._validate_frame, fg_color="transparent")
    #         cf.grid(row=1, column=0, columnspan=4, sticky="ew")
    #         for ci, txt in enumerate(["Source", "Current Glossary", "Correction", "Select"]):
    #             kwargs = {"anchor": "w", "font": ctk.CTkFont(size=11, weight="bold")}
    #             ctk.CTkLabel(cf, text=txt, **kwargs).grid(row=0, column=ci, sticky="w", padx=5)
    #         ctk.CTkFrame(self._validate_frame, height=1, fg_color="#444444").grid(row=2, column=0, columnspan=4, sticky="ew")
    #         to_render = []
    #         for src_tok, tgt_tok, freq in entries:
    #             src_lower = src_tok.lower()
    #             result = llm_results.get(src_lower)
    #             if not isinstance(result, (list, tuple)):
    #                 continue
    #             verdict, suggestion = result
    #             if verdict == "OK" and not show_all_var.get():
    #                 continue
    #             cur = checked.get(src_tok, tgt_tok)
    #             default_val = suggestion if verdict == "REJECT" and suggestion else cur
    #             to_render.append((src_tok, cur, default_val, freq, verdict))
    #         self._validate_to_render = to_render
    #         pp = self._g_validate_g_per_page()
    #         max_page = max(0, (len(to_render) - 1) // pp) if to_render else 0
    #         if self._validate_g_page > max_page:
    #             self._validate_g_page = max_page
    #         start = self._validate_g_page * pp
    #         end = start + pp
    #         page_items = to_render[start:end]
    #         info_text = f"Done. {len(llm_results)}/{len(entries)} terms. "
    #         if reject_count:
    #             info_text += f"{reject_count} REJECTED. " + ("Showing all." if show_all_var.get() else "Showing only REJECTED.")
    #         else:
    #             info_text += "All terms OK."
    #         self._validate_info.configure(text=info_text)
    #         BATCH = 30
    #         def _render_batch(i):
    #             for j in range(i, min(i + BATCH, len(page_items))):
    #                 src_tok, cur, default_val, freq, verdict = page_items[j]
    #                 r = j + 3
    #                 ctk.CTkLabel(self._validate_frame, text=src_tok, anchor="w", font=ctk.CTkFont(size=11)).grid(row=r, column=0, sticky="w", padx=5, pady=2)
    #                 ctk.CTkLabel(self._validate_frame, text=cur, anchor="w", font=ctk.CTkFont(size=11), text_color="red" if verdict == "REJECT" else "gray").grid(row=r, column=1, sticky="w", padx=5, pady=2)
    #                 e = ctk.CTkEntry(self._validate_frame, font=ctk.CTkFont(size=11))
    #                 e.grid(row=r, column=2, sticky="ew", padx=5, pady=2)
    #                 e.insert(0, default_val)
    #                 cb_var = ctk.BooleanVar(value=False)
    #                 ctk.CTkCheckBox(self._validate_frame, text="", variable=cb_var, width=20).grid(row=r, column=3, padx=5, pady=2)
    #                 self._validate_row_widgets.append((src_tok, freq, e, verdict, cb_var))
    #             if i + BATCH < len(page_items):
    #                 self.after(5, lambda: _render_batch(i + BATCH))
    #             else:
    #                 self._g_validate_show_nav()
    #         _render_batch(0)
    #     _render()
    # 
    # def _g_validate_show_nav(self):
    #     if not hasattr(self, '_validate_toplevel') or not self._validate_toplevel:
    #         return
    #     self._validate_progress.set(1)
    #     vgf = ctk.CTkFrame(self._validate_toplevel, fg_color="transparent")
    #     vgf.grid(row=3, column=0, padx=10, pady=(0, 2), sticky="ew")
    #     def _nav_refresh():
    #         self._validate_g_page = 0
    #         self._g_validate_done(self._validate_llm)
    #     self._validate_g_page_label = ctk.CTkLabel(vgf, text="")
    #     self._validate_g_page_label.pack(side="left", padx=5)
    #     ctk.CTkButton(vgf, text="< Prev", width=60, command=self._g_validate_prev_page).pack(side="left", padx=5)
    #     ctk.CTkButton(vgf, text="Next >", width=60, command=self._g_validate_next_page).pack(side="left", padx=5)
    #     ctk.CTkLabel(vgf, text="  Lines/page:").pack(side="left", padx=2)
    #     _vg_pp = ctk.CTkEntry(vgf, textvariable=self._validate_g_per_page_var, width=50)
    #     _vg_pp.pack(side="left", padx=5)
    #     _vg_pp.bind("<KeyRelease>", lambda e: _nav_refresh())
    #     self._g_validate_update_page_label()
    #     bf = ctk.CTkFrame(self._validate_toplevel, fg_color="transparent")
    #     bf.grid(row=4, column=0, padx=10, pady=(0, 10), sticky="ew")
    #     ctk.CTkButton(bf, text="Apply Changes", fg_color="#2E7D32", command=self._g_validate_apply).pack(side="left", padx=5)
    #     self._g_validate_retry_btn = ctk.CTkButton(bf, text="Re-validate Selected", fg_color="#1565C0", command=self._g_validate_retry_selected)
    #     self._g_validate_retry_btn.pack(side="left", padx=5)
    #     ctk.CTkButton(bf, text="Cancel", command=self._g_validate_cancel).pack(side="left", padx=5)
    # 
    # def _g_validate_update_page_label(self):
    #     if not hasattr(self, '_validate_to_render') or not hasattr(self, '_validate_g_page_label'):
    #         return
    #     total = len(self._validate_to_render)
    #     pp = self._g_validate_g_per_page()
    #     tp = (total - 1) // pp + 1 if total > 0 else 1
    #     self._validate_g_page_label.configure(text=f"Page {self._validate_g_page+1}/{tp} ({total} terms)")
    # 
    # def _g_validate_prev_page(self):
    #     if self._validate_g_page > 0:
    #         self._validate_g_page -= 1
    #         self._g_validate_done(self._validate_llm)
    # 
    # def _g_validate_next_page(self):
    #     total = len(getattr(self, '_validate_to_render', []))
    #     pp = self._g_validate_g_per_page()
    #     if (self._validate_g_page + 1) * pp < total:
    #         self._validate_g_page += 1
    #         self._g_validate_done(self._validate_llm)
    # 
    # def _g_validate_apply(self):
    #     if not hasattr(self, '_validate_row_widgets'):
    #         return
    #     new_checked = dict(self._validate_checked)
    #     new_entries = list(self._validate_entries)
    #     updated = 0
    #     for src_tok, freq, entry, verdict, cb_var in self._validate_row_widgets:
    #         user_val = entry.get().strip()
    #         if user_val:
    #             new_checked[src_tok] = user_val
    #             for i, (st, _, f) in enumerate(new_entries):
    #                 if st == src_tok:
    #                     new_entries[i] = (st, user_val, f)
    #                     updated += 1
    #                     break
    #     if not updated:
    #         self.log("[VALIDATE] Nothing to change")
    #         return
    #     self._g_checked = new_checked
    #     self._g_entries = new_entries
    #     self._g_page = 0
    #     self._game_update_display()
    #     self.log(f"[VALIDATE] Updated {updated} REJECTED terms")
    #     self._g_validate_cancel()
    # 
    # def _g_validate_retry_selected(self):
    #     if not hasattr(self, '_validate_row_widgets'):
    #         return
    #     selected = [(st, e.get().strip()) for st, _, e, _, cb in self._validate_row_widgets if cb.get() and e.get().strip()]
    #     if not selected:
    #         self.log("[VALIDATE] No terms selected for re-validation")
    #         return
    #     model = self.ollama_model.get()
    #     if not model or model == "(none)":
    #         self.log("[ERROR] Select a model first")
    #         return
    #     tgt_display = self.target_lang.get()
    #     game = self.selected_game.get()
    #     self._validate_info.configure(text=f"Re-validating {len(selected)} terms...")
    #     self._validate_progress.set(0)
    #     self.engine.stop_event.clear()
    #     def _run():
    #         try:
    #             if self.engine.stop_event.is_set():
    #                 self.after(0, lambda: self.log("[STOP] Validate retry interrupted"))
    #                 return
    #             prompt = (
    #                 f"Review these English -> {tgt_display} glossary terms for the game '{game}'.\n"
    #                 "For each term, reply in this exact format:\n"
    #                 "term: OK\n"
    #                 "term: REJECT:suggested_translation\n\n"
    #                 "- OK = translation is approximately correct for game context.\n"
    #                 "- REJECT = translation is completely wrong or unrelated.\n"
    #                 "  Provide a suggested correct translation after the second colon.\n\n"
    #             )
    #             for src, cur in selected:
    #                 prompt += f"{src} -> {cur}\n"
    #             result = self.engine._call_ollama(model, prompt, temperature=0.1, max_tokens=4096)
    #             for line in result.split("\n"):
    #                 line = line.strip()
    #                 if ":" in line and not line.startswith("```") and not line.startswith("#"):
    #                     parts = line.split(":", 1)
    #                     src = parts[0].strip().lower()
    #                     rest = parts[1].strip()
    #                     rest_up = rest.upper()
    #                     if rest_up.startswith("OK"):
    #                         self._validate_llm[src] = ("OK", "")
    #                     elif rest_up.startswith("REJECT"):
    #                         sug = rest.split(":", 1)[1].strip() if ":" in rest else ""
    #                         self._validate_llm[src] = ("REJECT", sug)
    #             self.after(0, lambda: self._g_validate_done(self._validate_llm))
    #         except Exception as e:
    #             tb = traceback.format_exc()
    #             self.after(0, lambda: self.log(f"[ERROR] _g_validate_retry_selected:\n{tb}"))
    #         finally:
    #             pass
    #     threading.Thread(target=_run, daemon=True).start()
    # 
    # def _g_validate_cancel(self):
    #     if hasattr(self, '_validate_toplevel') and self._validate_toplevel:
    #         self._validate_toplevel.destroy()
    #         del self._validate_toplevel
    #     for attr in ('_validate_frame', '_validate_progress', '_validate_info',
    #                  '_validate_entries', '_validate_checked', '_validate_llm',
    #                  '_validate_row_widgets', '_validate_to_render',
    #                  '_validate_g_page_label', '_validate_g_per_page_var',
    #                  '_validate_cache_path', '_validate_game',
    #                  '_validate_src_display', '_validate_tgt_display'):
    #         if hasattr(self, attr):
    #             delattr(self, attr)
    
    # ── Glossary MOD tab ──

    def _mod_extract(self):
        if getattr(self, '_m_running', False):
            return
        folder = self._m_folder_var.get() or self.input_dir.get()
        min_freq = int(self._g_min_var.get())
        if not folder or not os.path.isdir(folder):
            self.log("[ERROR] Select a folder first")
            return
        def _run():
            try:
                self._m_running = True
                self.after(0, lambda: self.stop_btn.configure(state="normal"))
                self.after(0, lambda: self._m_progress.set(0))
                src_freq = {}
                all_files = []
                for root, _, fnames in os.walk(folder):
                    for fn in fnames:
                        if fn.lower().endswith((".yml", ".yaml")):
                            all_files.append(os.path.join(root, fn))
                total_files = len(all_files)
                for fi, fp in enumerate(all_files):
                    if self.engine.stop_event.is_set():
                        self.after(0, lambda: self.log("[STOP] Extract interrupted"))
                        break
                    with open(fp, "r", encoding="utf-8-sig") as f:
                        for line in f:
                            if ":" in line:
                                val = line.split(":", 1)[1].strip()
                                if val.startswith('"') and val.endswith('"'):
                                    val = val[1:-1]
                                src_lang = getattr(self, 'source_lang', None)
                                prefix = f"l_{OllamaTranslator._LANG_CODE.get(src_lang.get().lower(), src_lang.get().lower())}" if src_lang and src_lang.get() else "l_english"
                                for t in tokenize_mod_en(val, prefix):
                                        src_freq[t] = src_freq.get(t, 0) + 1
                    self.after(0, lambda v=(fi+1)/max(total_files,1): self._m_progress.set(v))
                sorted_entries = sorted(((k, v) for k, v in src_freq.items() if v >= min_freq), key=lambda x: -x[1])
                self._m_entries = sorted_entries
                self._m_checked = {}
                self._m_page = 0
                self.after(0, lambda: self._mod_update_display())
                self.after(0, lambda: self.log(f"[MOD] Extracted {len(sorted_entries)} unique English terms"))
            except Exception as e:
                tb = traceback.format_exc()
                self.after(0, lambda: self.log(f"[ERROR] _mod_extract:\n{tb}"))
            finally:
                self._m_running = False
                self.after(0, lambda: self.stop_btn.configure(state="disabled"))
        threading.Thread(target=_run, daemon=True).start()

    def _mod_translate(self):
        if getattr(self, '_m_translating', False):
            return
        entries = getattr(self, '_m_entries', [])
        if not entries:
            self.log("[ERROR] Extract terms first")
            return
        model = self.ollama_model.get()
        tgt_display = self.target_lang.get()
        if not model or model == "(none)":
            self.log("[ERROR] Select a model")
            return
        batch_size = getattr(self, 'batch_size', None)
        batch = int(batch_size.get()) if batch_size and batch_size.get() else 50
        nctx = getattr(self, 'num_ctx', None)
        num_ctx_val = int(nctx.get()) if nctx and nctx.get() else 4096
        def _run():
            try:
                self.engine.stop_event.clear()
                self.after(0, lambda: self.stop_btn.configure(state="normal"))
                self._m_translating = True
                self.after(0, lambda: self._m_progress.set(0))
                # ── Proper noun filtering ──
                total = len(entries)
                self.after(0, lambda total=total: self.log(f"[MOD] Filtering proper nouns ({total} terms)..."))
                keep = []
                for i in range(0, total, batch):
                    if self.engine.stop_event.is_set():
                        self.after(0, lambda: self.log("[STOP] Mod filtering interrupted"))
                        break
                    chunk = [e[0] for e in entries[i:i+batch]]
                    prompt = (
                        "Classify each English word. If it is a proper noun "
                        "(specific name of a person, place, character, planet, ship, item, faction, event, etc. in a game context), "
                        "write 'word: PROPER'. Otherwise write 'word: COMMON'.\n"
                        "Reply with one classification per line, preserving the exact word.\n"
                        + "\n".join(chunk)
                    )
                    result = self.engine._call_ollama(model, prompt,
                        temperature=0.1, max_tokens=4096, num_ctx=num_ctx_val)
                    word_map = {}
                    for line in result.split("\n"):
                        if ":" in line:
                            parts = line.split(":", 1)
                            w = parts[0].strip().lower()
                            tag = parts[1].strip().upper()
                            if tag == "PROPER":
                                word_map[w] = True
                    for e in entries[i:i+batch]:
                        if e[0].lower() in word_map:
                            keep.append(e)
                    self.after(0, lambda i=i, total=total: self.log(
                        f"[MOD] Filtering batch {i//batch+1}/{(total-1)//batch+1}..."))
                if self.engine.stop_event.is_set():
                    self._m_translating = False
                    return
                removed = len(entries) - len(keep)
                self._m_entries = keep
                filtered = keep
                self.after(0, lambda r=removed: self.log(f"[MOD] Removed {r} common words, {len(keep)} proper nouns remaining"))
                if not filtered:
                    self.after(0, lambda: self.log("[MOD] No proper nouns found"))
                    self._m_translating = False
                    return
                # ── Translation ──
                self.after(0, lambda: self._m_progress.set(0))
                total = len(filtered)
                checked = {}
                for i in range(0, total, batch):
                    if self.engine.stop_event.is_set():
                        self.after(0, lambda: self.log("[STOP] Mod translation interrupted"))
                        break
                    chunk = filtered[i:i+batch]
                    prompt = (
                        f"Translate each English word to {tgt_display}. "
                        f"Return exactly one 'word: translation' per line with a colon separator.\n"
                        + "\n".join([f"{e[0]}" for e in chunk])
                    )
                    result = self.engine._call_ollama(model, prompt,
                        temperature=0.1, max_tokens=4096, num_ctx=num_ctx_val)
                    translated = {}
                    for line in result.split("\n"):
                        if ":" in line:
                            parts = line.split(":", 1)
                            translated[parts[0].strip().lower()] = parts[1].strip()
                    for src, freq in chunk:
                        if src in translated:
                            checked[src] = translated[src]
                    pct = min(1.0, (i + batch) / total)
                    self.after(0, lambda v=pct: self._m_progress.set(v))
                    self.after(0, lambda c=len(checked), i=i: self.log(
                        f"[MOD] Batch {i//batch+1}/{(total-1)//batch+1}: {c} translated"))
                self._m_checked = checked
                self._m_page = 0
                self.after(0, lambda: self._mod_update_display())
                self.after(0, lambda: self.log(f"[MOD] Translated {len(checked)}/{len(filtered)} terms"))
            except Exception as e:
                tb = traceback.format_exc()
                self.after(0, lambda: self.log(f"[ERROR] _mod_translate:\n{tb}"))
            finally:
                self._m_translating = False
                self.after(0, lambda: self.stop_btn.configure(state="disabled"))
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
        page = entries[start:end]
        checked = getattr(self, '_m_checked', {})
        self._m_check_vars = {}
        
        ROW_H = 28
        for ci, txt in enumerate(["Source", "Target", "Freq", "☑"]):
            kwargs = {"anchor": "w", "font": ctk.CTkFont(size=11, weight="bold")}
            ctk.CTkLabel(self._m_result_frame, text=txt, **kwargs).grid(row=0, column=ci, sticky="w", padx=5)
        ctk.CTkFrame(self._m_result_frame, height=1, fg_color="#444444").grid(row=1, column=0, columnspan=4, sticky="ew", pady=2)
        
        for ri, entry in enumerate(page):
            row = ri + 2
            src_tok = entry[0]
            freq = entry[2] if len(entry) >= 3 else entry[1]
            self._m_result_frame.grid_rowconfigure(row, minsize=ROW_H)
            ctk.CTkLabel(self._m_result_frame, text=src_tok, anchor="w", font=ctk.CTkFont(size=11)).grid(row=row, column=0, sticky="ew", padx=5)
            e = ctk.CTkEntry(self._m_result_frame, font=ctk.CTkFont(size=11))
            e.grid(row=row, column=1, sticky="ew", padx=5)
            e.insert(0, checked.get(src_tok, ""))
            e.bind("<FocusOut>", lambda evt, src=src_tok: self._m_save_entry(src, e.get()))
            ctk.CTkLabel(self._m_result_frame, text=str(freq), anchor="e", font=ctk.CTkFont(size=11)).grid(row=row, column=2, sticky="e", padx=5)
            chk_var = ctk.BooleanVar(value=False)
            self._m_check_vars[src_tok] = chk_var
            ctk.CTkCheckBox(self._m_result_frame, text="", variable=chk_var, width=20).grid(row=row, column=3, padx=5, pady=2)
        self._m_page_label.configure(text=f"Page {self._m_page+1}/{(total-1)//self._m_per_page+1 if total > 0 else 1} ({total} terms)")

    def _m_save_entry(self, src_tok, value):
        """Save entry value to checked dict"""
        checked = getattr(self, '_m_checked', {})
        if value.strip():
            checked[src_tok] = value
        elif src_tok in checked:
            del checked[src_tok]
        self._m_checked = checked

    def _m_remove_selected(self):
        check_vars = getattr(self, '_m_check_vars', {})
        if not check_vars:
            return
        selected = [src for src, var in check_vars.items() if var.get()]
        if not selected:
            return
        entries = getattr(self, '_m_entries', [])
        entries = [(s, f) for s, f in entries if s not in selected]
        self._m_entries = entries
        checked = getattr(self, '_m_checked', {})
        for src in selected:
            checked.pop(src, None)
        self._m_checked = checked
        self._m_page = 0
        self._mod_update_display(self._m_search_var.get() if hasattr(self, '_m_search_var') else "")
        self.log(f"[MOD] Removed {len(selected)} terms ({len(entries)} remaining)")

    def _mod_retry_selected(self):
        check_vars = getattr(self, '_m_check_vars', {})
        if not check_vars:
            return
        selected = [src for src, var in check_vars.items() if var.get()]
        if not selected:
            self.log("[MOD] No terms selected for retry")
            return
        model = self.ollama_model.get()
        tgt_display = self.target_lang.get()
        if not model or model == "(none)":
            self.log("[ERROR] Select a model first")
            return
        batch = 50
        def _run():
            try:
                self.engine.stop_event.clear()
                self.after(0, lambda: self.stop_btn.configure(state="normal"))
                for i in range(0, len(selected), batch):
                    if self.engine.stop_event.is_set():
                        self.after(0, lambda: self.log("[STOP] Mod retry interrupted"))
                        break
                    chunk = selected[i:i+batch]
                    prompt = (
                        f"Translate each English word to {tgt_display}. "
                        f"Return exactly one 'word: translation' per line with a colon separator.\n"
                        + "\n".join(chunk)
                    )
                    result = self.engine._call_ollama(model, prompt,
                        temperature=0.1, max_tokens=4096)
                    translated = {}
                    for line in result.split("\n"):
                        if ":" in line:
                            parts = line.split(":", 1)
                            translated[parts[0].strip().lower()] = parts[1].strip()
                    checked = getattr(self, '_m_checked', {})
                    prev = dict(checked)
                    for src in chunk:
                        if src in translated:
                            checked[src] = translated[src]
                    self._m_checked = checked
                    self.after(0, lambda c=len(checked), i=i, total=len(selected): self.log(
                        f"[MOD] Retry batch {i//batch+1}/{(total-1)//batch+1}: {c} translated"))
                self.after(0, lambda: self._mod_update_display())
                self.after(0, lambda: self.log(f"[MOD] Retry done: {len(selected)} terms re-translated"))
            except Exception as e:
                tb = traceback.format_exc()
                self.after(0, lambda: self.log(f"[ERROR] _mod_retry_selected:\n{tb}"))
            finally:
                self.after(0, lambda: self.stop_btn.configure(state="disabled"))
        threading.Thread(target=_run, daemon=True).start()

    def _m_prev_page(self):
        if self._m_page > 0:
            self._m_page -= 1
            self._mod_update_display()

    def _m_next_page(self):
        entries = self._m_entries if hasattr(self, '_m_entries') else []
        if (self._m_page + 1) * self._m_per_page < len(entries):
            self._m_page += 1
            self._mod_update_display()

    def _mod_save(self):
        game = self.selected_game.get()
        folder = self._m_folder_var.get() or self.input_dir.get()
        if not game:
            self.log("[ERROR] No game specified")
            return
        if not folder or not os.path.isdir(folder):
            self.log("[ERROR] Select a valid mod folder first")
            return
        modname = _derive_modname(folder)
        self._m_modname = modname
        src = self.source_lang.get()
        tgt = self.target_lang.get()
        src_code = OllamaTranslator._LANG_CODE.get(src, src.lower())
        tgt_code = OllamaTranslator._LANG_CODE.get(tgt, tgt.lower())
        base_dir = _glossary_dir()
        gd = os.path.join(base_dir, game)
        os.makedirs(gd, exist_ok=True)
        fname = f"{modname}_{src_code}_{tgt_code}.txt".lower()
        path = os.path.join(gd, fname)
        checked = getattr(self, '_m_checked', {})
        entries = self._m_entries if hasattr(self, '_m_entries') else []
        with open(path, "w", encoding="utf-8") as f:
            for src_tok, _ in entries:
                val = checked.get(src_tok, "")
                f.write(f"{src_tok}: {val}\n")
        self.log(f"[MOD] Saved {len(entries)} terms to {path}")

    def _mod_load(self):
        game = self.selected_game.get()
        folder = self._m_folder_var.get() or self.input_dir.get()
        if not game:
            self.log("[ERROR] No game specified")
            return
        if not folder or not os.path.isdir(folder):
            self.log("[ERROR] Select a valid mod folder first")
            return
        modname = _derive_modname(folder)
        src = self.source_lang.get()
        tgt = self.target_lang.get()
        src_code = OllamaTranslator._LANG_CODE.get(src, src.lower())
        tgt_code = OllamaTranslator._LANG_CODE.get(tgt, tgt.lower())
        base_dir = _glossary_dir()
        fname = f"{modname}_{src_code}_{tgt_code}.txt".lower()
        path = os.path.join(base_dir, game, fname)
        if not os.path.isfile(path):
            old = os.path.join(base_dir, game, f"{modname}.txt")
            if os.path.isfile(old):
                path = old
            else:
                self.log(f"[ERROR] No glossary file found at {path}")
                return
        entries = []
        checked = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(":", 1)
                if len(parts) == 2:
                    k, v = parts[0].strip(), parts[1].strip()
                    entries.append((k, v))
                    if v:
                        checked[k] = v
        if entries:
            self._m_entries = entries
            self._m_checked = checked
            self._m_page = 0
            self._mod_update_display()
            self.log(f"[MOD] Loaded {len(entries)} terms from {path}")
        else:
            self.log(f"[ERROR] No valid entries in {path}")
