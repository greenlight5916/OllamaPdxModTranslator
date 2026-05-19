import os, sys, json, time, re, threading, subprocess, traceback
import customtkinter as ctk
from tkinter import filedialog, messagebox

from ollama_translator.engine import OllamaTranslator
from ollama_translator.tabs.glossary_tab import GlossaryTabMixin

def _app_dir():
    """Return the application directory (where config and glossary are stored)."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    # When running as script, return the ollama_translator package directory
    return os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(_app_dir(), "ollama_translator_config.json")
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

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
# (Moved to ollama_translator/engine.py)

# ── GUI ──

class OllamaTranslatorGUI(ctk.CTk, GlossaryTabMixin):
    def __init__(self):
        super().__init__()
        self.title("Ollama PDX Translator")
        self.geometry("950x750")
        self.stop_event = threading.Event()
        self.ollama_url = ctk.StringVar(value="http://localhost:11434")
        self.input_dir = ctk.StringVar()
        self.output_dir = ctk.StringVar()
        self.source_lang = ctk.StringVar(value="English")
        self.target_lang = ctk.StringVar(value="Korean")
        self.ollama_model = ctk.StringVar()
        self.temperature = ctk.DoubleVar(value=0.2)
        self.max_tokens = ctk.IntVar(value=8192)
        self.num_ctx = ctk.IntVar(value=4096)
        self.batch_size = ctk.IntVar(value=20)
        self.max_retries = ctk.IntVar(value=3)
        self.timeout = ctk.IntVar(value=90)
        self.selected_game = ctk.StringVar()
        self.available_games = list(GAME_PROMPTS.keys())
        self.show_prompt = ctk.BooleanVar(value=False)
        self.checkpoint_enabled = ctk.BooleanVar(value=True)
        self.debug_mode = ctk.BooleanVar(value=False)
        self.prompt_template_var = ctk.StringVar(value=self._default_prompt())
        self.live_visible = ctk.BooleanVar(value=False)
        self._connected = False
        # self._g_running = False  # 인 게임 용어로 통일하기 위해 개발했으나 현재 개발중단
        self._m_modname = "mod"
        self._m_folder_var = ctk.StringVar()
        # self._g_page = 0  # 인 게임 용어로 통일하기 위해 개발했으나 현재 개발중단
        # self._g_per_page_var = ctk.StringVar(value="20")
        # self._g_dirty = False
        self._m_page = 0
        self._m_per_page = 20
        self._validate_file_pairs = []
        self._validate_all_data = []
        self._validate_modified = {}
        self._validate_page = 0
        self._validate_hide_filter = ctk.BooleanVar(value=False)
        self._validate_per_page = ctk.IntVar(value=20)
        self._validate_g_page = 0
        self._validate_g_per_page_var = ctk.StringVar(value="20")
        self.available_langs = ["English", "Korean", "Simplified Chinese", "French", "German",
                                "Spanish", "Japanese", "Brazilian Portuguese", "Russian", "Polish"]
        self.engine = OllamaTranslator(
            log_callback=self.log, progress_callback=self.update_progress,
            status_callback=self.set_status,
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
            self.num_ctx.set(cfg.get("num_ctx", 4096))
            self.batch_size.set(cfg.get("batch_size", 1))
            self.max_retries.set(cfg.get("max_retries", 3))
            self.timeout.set(cfg.get("timeout", 90))
            p = cfg.get("prompt_template", "")
            if p:
                self.prompt_template_var.set(p)
                self.prompt_textbox.delete("1.0", "end")
                self.prompt_textbox.insert("1.0", p)
            if cfg.get("show_prompt", False):
                self.show_prompt.set(True)
                self.prompt_frame.grid()
            # ── 인 게임 용어로 통일하기 위해 개발했으나 현재 개발중단 ──
            # if hasattr(self, '_g_folder_var'):
            #     gg = cfg.get("glossary_game", "")
            #     if gg in self.available_games:
            #         self.selected_game.set(gg)
            #     self.source_lang.set(cfg.get("glossary_src", cfg.get("source_lang", "English")))
            #     self.target_lang.set(cfg.get("glossary_tgt", cfg.get("target_lang", "Korean")))
            #     self._g_min_var.set(str(cfg.get("glossary_min", 3)))
            #     self._g_folder_var.set(cfg.get("glossary_folder", ""))
            mg = cfg.get("glossary_mod_game", "")
            if mg in self.available_games:
                self.selected_game.set(mg)
            self._g_min_var.set(str(cfg.get("glossary_mod_min", 3)))
            self._m_folder_var.set(cfg.get("glossary_mod_folder", ""))
            if not self._m_folder_var.get() and self.input_dir.get():
                self._m_folder_var.set(self.input_dir.get())
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
            "num_ctx": self.num_ctx.get(),
            "batch_size": self.batch_size.get(),
            "max_retries": self.max_retries.get(),
            "timeout": self.timeout.get(),
            "prompt_template": self.prompt_template_var.get(),
            "show_prompt": self.show_prompt.get(),
            # ── 인 게임 용어로 통일하기 위해 개발했으나 현재 개발중단 ──
            # "glossary_game": self.selected_game.get(),
            # "glossary_src": self.source_lang.get(),
            # "glossary_tgt": self.target_lang.get(),
            "glossary_min": int(self._g_min_var.get()) if hasattr(self, '_g_min_var') else 3,
            # "glossary_folder": self._g_folder_var.get() if hasattr(self, '_g_folder_var') else "",
            "glossary_mod_game": self.selected_game.get(),
            "glossary_mod_folder": self._m_folder_var.get() if hasattr(self, '_m_folder_var') else "",
            "glossary_mod_tgt": self.target_lang.get(),
            "glossary_mod_min": int(self._g_min_var.get()),
            "glossary_mod_name": self._m_modname if hasattr(self, '_m_modname') else "",
        }
        try:
            with open(self._config_path(), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _on_close(self):
        # self._g_running = False  # 인 게임 용어로 통일하기 위해 개발했으나 현재 개발중단
        self._m_running = False
        dirty = bool(getattr(self, '_validate_modified', None))
        if dirty:
            dlg = ctk.CTkToplevel(self)
            dlg.title("Exit")
            dlg.transient(self)
            dlg.grab_set()
            dlg.grid_columnconfigure(0, weight=1)
            result = [None]
            ctk.CTkLabel(dlg, text="Save changes before exiting?", font=ctk.CTkFont(size=13)).grid(row=0, column=0, pady=(15, 10), padx=20)
            bf = ctk.CTkFrame(dlg, fg_color="transparent")
            bf.grid(row=1, column=0, pady=5)
            for i, t in enumerate(["Save & Exit", "Exit", "Cancel"]):
                ctk.CTkButton(bf, text=t, width=100,
                              fg_color="#D32F2F" if i == 1 else "#2E7D32" if i == 0 else "#757575",
                              hover_color="#E53935" if i == 1 else "#388E3C" if i == 0 else "#9E9E9E",
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
                # if getattr(self, '_g_dirty', False):  # 인 게임 용어로 통일하기 위해 개발했으나 현재 개발중단
                #     self._game_save()
                if getattr(self, '_validate_modified', None):
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
        if hasattr(self, '_g_validate_btn'):
            gst = "normal" if self._connected and self.ollama_model.get() not in ("", "(none)") else "disabled"
            self._g_validate_btn.configure(state=gst)
        if hasattr(self, '_validate_retry_btn'):
            self._validate_retry_btn.configure(state=st)

    def _default_prompt(self):
        return ("Translate the following text from '{source_lang}' to '{target_lang}'.\n"
                "Rules:\n"
                "1. Never create new {{PH0}}, {{PH1}} etc. — they are special sentinels I inserted. "
                "Keep every one exactly as-is. Do NOT add missing ones, do NOT duplicate them.\n"
                "2. Preserve line markers like \u27e80\u27e9 \u27e81\u27e9 exactly as-is.\n"
                "3. Do NOT wrap in code blocks or add explanations.\n"
                "4. Translate proper nouns (person names, place names, character names) based on pronunciation, not meaning.\n\n{batch_text}")

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
                    btn = ctk.CTkButton(f, text=lbl, width=100, command=cmd)
                    btn.grid(row=0, column=bc, padx=(5, 0))
                    bc += 1
                    if lbl == "Start Ollama":
                        self._start_ollama_btn = btn
                    elif lbl == "Connect":
                        self._connect_btn = btn
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
        for lbl, var, w in [("Temp:", self.temperature, 50), ("Tokens:", self.max_tokens, 60),
                            ("Ctx:", self.num_ctx, 60), ("Batch:", self.batch_size, 45), ("Retry:", self.max_retries, 35), ("Timeout:", self.timeout, 45)]:
            ctk.CTkLabel(af, text=lbl).pack(side="left", padx=(6, 2))
            ctk.CTkEntry(af, textvariable=var, width=w).pack(side="left", padx=(0, 4))
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
        # self.reset_btn = ctk.CTkButton(cf, text="Reset", command=self._reset_ui, fg_color="#757575", hover_color="#9E9E9E", state="disabled")
        # self.reset_btn.pack(side="left", padx=5)
        self.live_btn = ctk.CTkButton(cf, text="Live", command=self._toggle_live, fg_color="#1565C0", hover_color="#1976D2", width=60)
        self.live_btn.pack(side="left", padx=5)
        self.status_label = ctk.CTkLabel(cf, text="Ready", text_color="gray")
        self.status_label.pack(side="right", padx=10)
        t_gl = self.tabview.add("Glossary")
        t_gl.grid_columnconfigure(0, weight=1)
        t_gl.grid_rowconfigure(0, weight=1)
        gl_sub = ctk.CTkTabview(t_gl)
        gl_sub.grid(row=0, column=0, sticky="nsew")
        # ── GAME 탭: 인 게임 용어로 통일하기 위해 개발했으나 현재 개발중단 ──
        # game_tab = gl_sub.add("GAME")
        # game_tab.grid_columnconfigure(0, weight=1)
        # game_tab.grid_rowconfigure(5, weight=1)
        # r = 0
        # self._g_folder_var = ctk.StringVar()
        # fgf = ctk.CTkFrame(game_tab, fg_color="transparent")
        # fgf.grid(row=r, column=0, padx=10, pady=(10,2), sticky="ew")
        # fgf.grid_columnconfigure(1, weight=1)
        # ctk.CTkLabel(fgf, text="Game Path:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        # ctk.CTkEntry(fgf, textvariable=self._g_folder_var).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        # ctk.CTkButton(fgf, text="Browse", width=70, command=self._g_browse).grid(row=0, column=2, padx=5, pady=5)
        # r += 1
        # lf1 = ctk.CTkFrame(game_tab, fg_color="transparent")
        # lf1.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        # ctk.CTkLabel(lf1, text="Game:").pack(side="left", padx=5)
        # ctk.CTkComboBox(lf1, variable=self.selected_game, values=self.available_games, state="readonly", width=150).pack(side="left", padx=5)
        # ctk.CTkLabel(lf1, text="  Source:").pack(side="left", padx=5)
        # ctk.CTkComboBox(lf1, variable=self.source_lang, values=self.available_langs, state="readonly", width=150).pack(side="left", padx=5)
        # ctk.CTkLabel(lf1, text="  Target:").pack(side="left", padx=5)
        # ctk.CTkComboBox(lf1, variable=self.target_lang, values=self.available_langs, state="readonly", width=150).pack(side="left", padx=5)
        # r += 1
        # lf2 = ctk.CTkFrame(game_tab, fg_color="transparent")
        # lf2.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        self._g_min_var = ctk.StringVar(value="3")
        # ctk.CTkLabel(lf2, text="Min frequency:").pack(side="left", padx=5)
        # ctk.CTkEntry(lf2, textvariable=self._g_min_var, width=50).pack(side="left", padx=5)
        # ctk.CTkButton(lf2, text="Extract", fg_color="#1565C0", command=self._game_extract).pack(side="left", padx=15)
        # r += 1
        # self._g_search_var = ctk.StringVar()
        # sf = ctk.CTkFrame(game_tab, fg_color="transparent")
        # sf.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        # sf.grid_columnconfigure(1, weight=1)
        # ctk.CTkLabel(sf, text="Search:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        # se = ctk.CTkEntry(sf, textvariable=self._g_search_var, width=400)
        # se.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        # self._g_info_label = ctk.CTkLabel(sf, text="")
        # self._g_info_label.grid(row=0, column=2, padx=10, pady=5, sticky="e")
        # se.bind("<KeyRelease>", lambda e: self._game_update_display(self._g_search_var.get()))
        # r += 1
        # self._g_progress = ctk.CTkProgressBar(game_tab)
        # self._g_progress.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        # self._g_progress.set(0)
        # r += 1
        # self._g_result_frame = ctk.CTkScrollableFrame(game_tab)
        # self._g_result_frame.grid(row=r, column=0, padx=10, pady=5, sticky="nsew")
        # self._g_result_frame.grid_columnconfigure(2, weight=1)
        # r += 1
        # pgf = ctk.CTkFrame(game_tab, fg_color="transparent")
        # pgf.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        # self._g_page_label = ctk.CTkLabel(pgf, text="")
        # self._g_page_label.pack(side="left", padx=5)
        # ctk.CTkButton(pgf, text="< Prev", width=60, command=self._g_prev_page).pack(side="left", padx=5)
        # ctk.CTkButton(pgf, text="Next >", width=60, command=self._g_next_page).pack(side="left", padx=5)
        # ctk.CTkLabel(pgf, text="  Lines/page:").pack(side="left", padx=2)
        # _g_pp_entry = ctk.CTkEntry(pgf, textvariable=self._g_per_page_var, width=50)
        # _g_pp_entry.pack(side="left", padx=5)
        # _g_pp_entry.bind("<KeyRelease>", lambda e: self._game_update_display(self._g_search_var.get()))
        # r += 1
        # gbtn = ctk.CTkFrame(game_tab, fg_color="transparent")
        # gbtn.grid(row=r, column=0, padx=10, pady=5, sticky="ew")
        # ctk.CTkButton(gbtn, text="Save Glossary", fg_color="#2E7D32", command=self._game_save).pack(side="left", padx=5)
        # ctk.CTkButton(gbtn, text="Load Glossary", command=self._game_load).pack(side="left", padx=5)
        # self._g_validate_btn = ctk.CTkButton(gbtn, text="Validate With LLM", fg_color="#7B1FA2", command=self._game_validate)
        # self._g_validate_btn.pack(side="left", padx=5)
        # ── (중복) GAME 탭 UI ──
        # game_tab.grid_columnconfigure(0, weight=1)
        # game_tab.grid_rowconfigure(5, weight=1)
        # r = 0
        # self._g_folder_var = ctk.StringVar()
        # fgf = ctk.CTkFrame(game_tab, fg_color="transparent")
        # fgf.grid(row=r, column=0, padx=10, pady=(10,2), sticky="ew")
        # fgf.grid_columnconfigure(1, weight=1)
        # ctk.CTkLabel(fgf, text="Game Path:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        # ctk.CTkEntry(fgf, textvariable=self._g_folder_var).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        # ctk.CTkButton(fgf, text="Browse", width=70, command=self._g_browse).grid(row=0, column=2, padx=5, pady=5)
        # r += 1
        # lf1 = ctk.CTkFrame(game_tab, fg_color="transparent")
        # lf1.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        # ctk.CTkLabel(lf1, text="Game:").pack(side="left", padx=5)
        # ctk.CTkComboBox(lf1, variable=self.selected_game, values=self.available_games, state="readonly", width=150).pack(side="left", padx=5)
        # ctk.CTkLabel(lf1, text="  Source:").pack(side="left", padx=5)
        # ctk.CTkComboBox(lf1, variable=self.source_lang, values=self.available_langs, state="readonly", width=150).pack(side="left", padx=5)
        # ctk.CTkLabel(lf1, text="  Target:").pack(side="left", padx=5)
        # ctk.CTkComboBox(lf1, variable=self.target_lang, values=self.available_langs, state="readonly", width=150).pack(side="left", padx=5)
        # r += 1
        # lf2 = ctk.CTkFrame(game_tab, fg_color="transparent")
        # lf2.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        # self._g_min_var = ctk.StringVar(value="3")
        # ctk.CTkLabel(lf2, text="Min frequency:").pack(side="left", padx=5)
        # ctk.CTkEntry(lf2, textvariable=self._g_min_var, width=50).pack(side="left", padx=5)
        # ctk.CTkButton(lf2, text="Extract", fg_color="#1565C0", command=self._game_extract).pack(side="left", padx=15)
        # r += 1
        # self._g_search_var = ctk.StringVar()
        # sf = ctk.CTkFrame(game_tab, fg_color="transparent")
        # sf.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        # sf.grid_columnconfigure(1, weight=1)
        # ctk.CTkLabel(sf, text="Search:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        # se = ctk.CTkEntry(sf, textvariable=self._g_search_var, width=400)
        # se.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        # self._g_info_label = ctk.CTkLabel(sf, text="")
        # self._g_info_label.grid(row=0, column=2, padx=10, pady=5, sticky="e")
        # se.bind("<KeyRelease>", lambda e: self._game_update_display(self._g_search_var.get()))
        # r += 1
        # self._g_progress = ctk.CTkProgressBar(game_tab)
        # self._g_progress.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        # self._g_progress.set(0)
        # r += 1
        # self._g_result_frame = ctk.CTkScrollableFrame(game_tab)
        # self._g_result_frame.grid(row=r, column=0, padx=10, pady=5, sticky="nsew")
        # self._g_result_frame.grid_columnconfigure(2, weight=1)
        # r += 1
        # pgf = ctk.CTkFrame(game_tab, fg_color="transparent")
        # pgf.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        # self._g_page_label = ctk.CTkLabel(pgf, text="")
        # self._g_page_label.pack(side="left", padx=5)
        # ctk.CTkButton(pgf, text="< Prev", width=60, command=self._g_prev_page).pack(side="left", padx=5)
        # ctk.CTkButton(pgf, text="Next >", width=60, command=self._g_next_page).pack(side="left", padx=5)
        # ctk.CTkLabel(pgf, text="  Lines/page:").pack(side="left", padx=2)
        # _g_pp_entry = ctk.CTkEntry(pgf, textvariable=self._g_per_page_var, width=50)
        # _g_pp_entry.pack(side="left", padx=5)
        # _g_pp_entry.bind("<KeyRelease>", lambda e: self._game_update_display(self._g_search_var.get()))
        # r += 1
        # gbtn = ctk.CTkFrame(game_tab, fg_color="transparent")
        # gbtn.grid(row=r, column=0, padx=10, pady=5, sticky="ew")
        # ctk.CTkButton(gbtn, text="Save Glossary", fg_color="#2E7D32", command=self._game_save).pack(side="left", padx=5)
        # ctk.CTkButton(gbtn, text="Load Glossary", command=self._game_load).pack(side="left", padx=5)
        # self._g_validate_btn = ctk.CTkButton(gbtn, text="Validate With LLM", fg_color="#7B1FA2", command=self._game_validate)
        # self._g_validate_btn.pack(side="left", padx=5)
        mod_tab = gl_sub.add("MOD")
        mod_tab.grid_columnconfigure(0, weight=1)
        mod_tab.grid_rowconfigure(5, weight=1)
        r = 0
        mff = ctk.CTkFrame(mod_tab, fg_color="transparent")
        mff.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        mff.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(mff, text="Mod Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkEntry(mff, textvariable=self._m_folder_var).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(mff, text="Browse", width=70, command=self._m_browse).grid(row=0, column=2, padx=5, pady=5)
        r += 1
        mlf = ctk.CTkFrame(mod_tab, fg_color="transparent")
        mlf.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        ctk.CTkLabel(mlf, text="Source:").pack(side="left", padx=5)
        ctk.CTkComboBox(mlf, variable=self.source_lang, values=self.available_langs, state="readonly", width=150).pack(side="left", padx=5)
        ctk.CTkLabel(mlf, text="  Target:").pack(side="left", padx=5)
        ctk.CTkComboBox(mlf, variable=self.target_lang, values=self.available_langs, state="readonly", width=150).pack(side="left", padx=5)
        r += 1
        mf2 = ctk.CTkFrame(mod_tab, fg_color="transparent")
        mf2.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
        ctk.CTkLabel(mf2, text="Min frequency:").pack(side="left", padx=5)
        ctk.CTkEntry(mf2, textvariable=self._g_min_var, width=50).pack(side="left", padx=5)
        ctk.CTkButton(mf2, text="Extract Terms", fg_color="#1565C0", command=self._mod_extract).pack(side="left", padx=15)
        self._m_translate_btn = ctk.CTkButton(mf2, text="Translate with LLM", fg_color="#E65100", command=self._mod_translate)
        self._m_translate_btn.pack(side="left", padx=5)
        ctk.CTkLabel(mf2, text="  Glossary limit:").pack(side="left", padx=5)
        self._m_glossary_limit_var = ctk.StringVar(value="500")
        ctk.CTkEntry(mf2, textvariable=self._m_glossary_limit_var, width=60).pack(side="left", padx=5)
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
        # 열 설정: Source(3), Target(4), Frequency(4)
        self._m_result_frame.grid_columnconfigure(0, weight=3, minsize=120)
        self._m_result_frame.grid_columnconfigure(1, weight=4, minsize=160)
        self._m_result_frame.grid_columnconfigure(2, weight=0, minsize=60)
        self._m_result_frame.grid_columnconfigure(3, weight=0, minsize=30)
        r += 1
        # Prev/Next와 Retry/Save/Load를 한 줄에 표시
        mbtn = ctk.CTkFrame(mod_tab, fg_color="transparent")
        mbtn.grid(row=r, column=0, padx=10, pady=5, sticky="ew")
        # 왼쪽: 페이지 정보와 Prev/Next
        mleft = ctk.CTkFrame(mbtn, fg_color="transparent")
        mleft.pack(side="left", fill="both", expand=True)
        self._m_page_label = ctk.CTkLabel(mleft, text="")
        self._m_page_label.pack(side="left", padx=5)
        ctk.CTkButton(mleft, text="< Prev", width=60, command=self._m_prev_page).pack(side="left", padx=5)
        ctk.CTkButton(mleft, text="Next >", width=60, command=self._m_next_page).pack(side="left", padx=5)
        # 오른쪽: 액션 버튼들
        mright = ctk.CTkFrame(mbtn, fg_color="transparent")
        mright.pack(side="right", fill="both")
        ctk.CTkButton(mright, text="Remove Selected", fg_color="#D32F2F", command=self._m_remove_selected).pack(side="left", padx=5)
        self._m_retry_btn = ctk.CTkButton(mright, text="Retry Selected", fg_color="#E65100", command=self._mod_retry_selected)
        self._m_retry_btn.pack(side="left", padx=5)
        ctk.CTkButton(mright, text="Save Glossary", fg_color="#2E7D32", command=self._mod_save).pack(side="left", padx=5)
        ctk.CTkButton(mright, text="Load Glossary", command=self._mod_load).pack(side="left", padx=5)

    # ── Validate Tab ──

    def _validate_load_files(self):
        inp = self.input_dir.get()
        out = self.output_dir.get()
        src = self.source_lang.get()
        tgt = self.target_lang.get()
        if not inp or not out or not src or not tgt:
            self._validate_msg.configure(text="Set Input/Output folders and languages in Translate tab first", text_color="red")
            return
        src_code = OllamaTranslator._LANG_CODE.get(src, src.lower())
        tgt_code = OllamaTranslator._LANG_CODE.get(tgt, tgt.lower())
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
        self._validate_file_pairs.sort(key=lambda x: (
            0 if x["severity"] == "error" else 1 if x["severity"] == "warn" else 2,
            x["base"]
        ))
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
        # ── Read sidecar .warn file ──
        warned_lines = set()
        warn_path = fp["out"] + ".warn"
        if os.path.isfile(warn_path):
            try:
                with open(warn_path, "r") as f:
                    warned_lines = {int(line.strip()) for line in f if line.strip().isdigit()}
            except Exception:
                pass
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
            if i in warned_lines:
                status = "⚠"
                sev = "warn"
                has_issues = True
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
        # 정렬: error 우선, warn, 그 다음 normal
        sorted_pairs = sorted(self._validate_file_pairs, key=lambda x: (
            0 if x["severity"] == "error" else 1 if x["severity"] == "warn" else 2,
            x["base"]
        ))
        vals = []
        for fp in sorted_pairs:
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
        # 정렬: ✗ 항목 우선 (✗!, ✗?, ✗D, ✗X), 그 다음 ✓
        status_priority = {"⚠": 0, "✗!": 1, "✗?": 2, "✗D": 3, "✗X": 4, "✗": 5, "✓": 6, "H": 7, "-": 8}
        data.sort(key=lambda d: (status_priority.get(d["status"].split()[0] if " " in d["status"] else d["status"], 99), d["ln"]))
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
            sts_color = "green" if sts == "✓" else "orange" if sts == "⚠" else "red" if sts.startswith("✗") else "gray"
            ctk.CTkLabel(self._validate_frame, text=sts, width=FIXED[0], anchor="w", font=ctk.CTkFont(size=11), text_color=sts_color).grid(row=r, column=3, padx=(4,1))
            ctk.CTkLabel(self._validate_frame, text=str(row["ln"]), width=FIXED[1], anchor="w", font=ctk.CTkFont(size=11)).grid(row=r, column=4)
            chk = ctk.CTkCheckBox(self._validate_frame, text="", width=FIXED[2], command=lambda idx=row["_idx"]: self._validate_toggle(idx))
            chk.grid(row=r, column=5)
            page_chks.append(chk)
        self._validate_page_label.configure(text=f"{self._validate_page+1}/{pages}")
        self._validate_info.configure(text=f"{total}/{len(self._validate_all_data)} lines  |  ✓ {sum(1 for d in self._validate_all_data if d['status']=='✓')}  ⚠ {sum(1 for d in self._validate_all_data if d['status']=='⚠')}  ✗ {sum(1 for d in self._validate_all_data if d['status'].startswith('✗'))}")

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
        sel = bool(hdr.get())
        for c, d in zip(chks, self._validate_current_page_data()):
            c.select() if sel else c.deselect()
            if 0 <= d["_idx"] < len(self._validate_all_data):
                self._validate_all_data[d["_idx"]]["checked"] = sel

    def _validate_current_page_data(self):
        filtered = [d for d in self._validate_all_data if d["status"] not in ("H", "-")] if not self._validate_hide_filter.get() else self._validate_all_data
        per_page = self._validate_per_page.get()
        if per_page < 1:
            per_page = 1
        start = self._validate_page * per_page
        return filtered[start:start + per_page]

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
        if 0 <= idx < len(self._validate_all_data):
            self._validate_all_data[idx]["checked"] = not self._validate_all_data[idx].get("checked", False)

    def _auto_load_validate(self):
        self._validate_load_files()

    def _validate_retry_selected(self):
        checked = [d for d in self._validate_all_data if d.get("checked", False) or d["status"].startswith("✗")]
        if not checked:
            return
        src = self.source_lang.get()
        tgt = self.target_lang.get()
        model = self.ollama_model.get()
        self._validate_retry_btn.configure(state="disabled", text="Retrying...")
        self._validate_save_btn.configure(state="disabled")
        self.engine.stop_event.clear()
        def _worker():
            try:
                for row in checked:
                    if self.engine.stop_event.is_set():
                        self.after(0, lambda: self.log("[STOP] Validate retry interrupted"))
                        break
                    if row["status"] in ("✓", "H", "-"):
                        continue
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
                        ln = row["ln"]
                        def _update(r2=result, rw2=row, ln2=ln):
                            rw2["tgt"] = r2
                            rw2["status"] = "✓"
                            self._validate_modified[ln2] = r2
                            self._validate_save_btn.configure(text="Save Changes *")
                        self.after(0, _update)
                self.after(0, self._validate_retry_done)
            finally:
                pass
        threading.Thread(target=_worker, daemon=True).start()

    def _validate_retry_done(self):
        self._validate_retry_btn.configure(state="normal", text="Retry Selected")
        self._validate_save_btn.configure(state="normal")
        self._validate_render()

    def _validate_save(self):
        if not self._validate_file_pairs or not self._validate_all_data:
            return
        base_name = self._validate_combo.get().replace("⚠ ", "").replace("✗ ", "")
        fp = next((f for f in self._validate_file_pairs if f["base"] == base_name), None)
        if not fp:
            self.log("[ERROR] Save: no matching file pair for current selection")
            return
        try:
            with open(fp["out"], "r", encoding="utf-8-sig") as f:
                lines = f.readlines()
            count = 0
            for ln, new_val in self._validate_modified.items():
                if ln < len(lines):
                    lines[ln] = re.sub(r'"[^"]*"', f'"{new_val}"', lines[ln], count=1)
                    count += 1
            with open(fp["out"], "w", encoding="utf-8-sig") as f:
                f.writelines(lines)
            self._validate_modified.clear()
            self._validate_save_btn.configure(text="Save Changes")
            self.log(f"[SAVE] Updated {count} lines in {fp['out']}")
        except Exception as e:
            self.log(f"[ERROR] Save failed: {e}")

    # ── Glossary ──
    # All glossary methods moved to ollama_translator/tabs/glossary_tab.py (GlossaryTabMixin)

    # ── Folders and Logging ──

    def _browse_input(self):
        d = filedialog.askdirectory()
        if d:
            src_code = OllamaTranslator._LANG_CODE.get(self.source_lang.get(), "").lower()
            tgt_code = OllamaTranslator._LANG_CODE.get(self.target_lang.get(), "").lower()
            for sub in ("localisation", "localization"):
                p = os.path.join(d, sub)
                if os.path.isdir(p):
                    d = p
                    break
            sf = os.path.join(d, src_code)
            if os.path.isdir(sf):
                d = sf
            self.input_dir.set(d)
            if hasattr(self, '_m_folder_var'):
                self._m_folder_var.set(d)
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

    def update_progress(self, current, total, file_index=1, total_files=1):
        def _u():
            overall = ((file_index - 1) + current / total) / total_files if total > 0 else 0
            self.progress_bar.set(min(overall, 1.0))
            self.progress_text.configure(text=f"File {file_index}/{total_files} | {current} / {total} lines")
        self.after(0, _u)

    def set_status(self, status):
        def _u():
            self.status_label.configure(text=status)
        self.after(0, _u)

    # ── Ollama Connection ──

    def _start_ollama(self):
        def _run():
            self._start_ollama_btn.configure(state="disabled")
            self.log("[OLLAMA] Starting server...")
            models = self.engine.start_server()
            if models:
                self.log(f"[OLLAMA] Models: {', '.join(models)}")
                self.ollama_combo.configure(values=models)
                if self.ollama_model.get() not in models:
                    self.ollama_model.set(models[0] if models else "")
                self._start_ollama_btn.configure(state="normal")
            else:
                self.log("[OLLAMA] Failed to start server")
                self._start_ollama_btn.configure(state="normal")
        threading.Thread(target=_run, daemon=True).start()

    def _connect_ollama(self):
        url = self.ollama_url.get().strip()
        if not url:
            self.log("[CONNECT] Enter Ollama URL first")
            return

        def _set_llm_buttons(state):
            self._connect_btn.configure(state=state)
            self.start_btn.configure(state=state)
            if hasattr(self, '_validate_retry_btn'):
                self._validate_retry_btn.configure(state=state)
            if hasattr(self, '_m_translate_btn'):
                self._m_translate_btn.configure(state=state)
            if hasattr(self, '_m_retry_btn'):
                self._m_retry_btn.configure(state=state)
            if hasattr(self, '_g_validate_retry_btn'):
                self._g_validate_retry_btn.configure(state=state)

        def _run():
            self.after(0, lambda: _set_llm_buttons("disabled"))
            self.log("[CONNECT] Starting server...")
            self.engine.set_base_url(url)
            try:
                models = self.engine.start_server()
                if not models:
                    self.log("[CONNECT] Failed to start server")
                    self._connected = False
                    return
                self.ollama_combo.configure(values=models)
                model = self.ollama_model.get()
                if not model or model not in models:
                    model = models[0]
                    self.ollama_model.set(model)
                self.log(f"[CONNECT] Loading model {model}...")
                import requests
                resp = requests.post(f"{url}/api/generate",
                    json={"model": model, "prompt": "hello", "stream": False},
                    timeout=120)
                if resp.status_code != 200:
                    self.log(f"[CONNECT] Model load failed (HTTP {resp.status_code})")
                    self._connected = False
                    return
                self.log("[CONNECT] Running test translation...")
                tgt = self.target_lang.get() or "Korean"
                test = self.engine.test_model(model, tgt, self.selected_game.get())
                if test and not test.startswith("[OLLAMA_"):
                    self._connected = True
                    self.log("[CONNECT] Translator Ready")
                else:
                    self._connected = False
                    self.log(f"[CONNECT] Model test failed: {test}")
            except Exception as e:
                self.log(f"[CONNECT] Error: {e}")
                self._connected = False
            finally:
                self.after(0, lambda: _set_llm_buttons("normal"))
                self.after(0, self._validate_fields)

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
            self.engine._checkpoint_restored = False
            if os.path.isdir(cp_dir):
                for fn in os.listdir(cp_dir):
                    if not fn.endswith((".yml", ".yaml")):
                        continue
                    cp = os.path.join(cp_dir, fn)
                    if not os.path.isfile(cp):
                        continue
                    out = self.output_dir.get()
                    os.makedirs(out, exist_ok=True)
                    out_path = os.path.join(out, fn)
                    with open(cp, "r", encoding="utf-8-sig") as f:
                        data = f.read()
                    with open(out_path, "w", encoding="utf-8-sig") as f:
                        f.write(data)
                    self.log(f"[RESUME] Restored checkpoint: {fn} ({len(data.splitlines())} lines)")
                    self.engine._checkpoint_restored = True
        self.engine.set_base_url(self.ollama_url.get())
        self._sync_prompt()
        self.engine.prompt_template = self.prompt_template_var.get()
        self.start_btn.configure(state="disabled")
        # self.reset_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log("[START] Translation started")
        self.engine.glossary_limit = int(self._m_glossary_limit_var.get())
        self.engine.start(self.input_dir.get(), self.output_dir.get(),
                          self.source_lang.get(), self.target_lang.get(),
                          self.ollama_model.get(), self.temperature.get(),
                          self.max_tokens.get(), self.batch_size.get(),
                          num_ctx=self.num_ctx.get(), game=self.selected_game.get(),
                           max_retries=self.max_retries.get(),
                           timeout=self.timeout.get(),
                           mod_folder=self._m_folder_var.get() or self.input_dir.get())
        self.after(500, self._poll_done)

    def _poll_done(self):
        if self.engine.busy:
            self.after(500, self._poll_done)
        else:
            self._done()

    def _done(self):
        self.engine.busy = False
        self._poll_count = 0
        self.start_btn.configure(state="normal")
        # self.reset_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.log("[DONE] Translation complete")
        self._auto_load_validate()

    def _stop(self):
        self.log("[STOP] Stopping translation...")
        self.stop_btn.configure(state="disabled")
        self.engine.stop()

    def _reset_ui(self):
        self.log_text.delete("1.0", "end")
        self.progress_bar.set(0)
        self.progress_text.configure(text="0 / 0 lines")
        self.start_btn.configure(state="normal" if self._connected else "disabled")
        # self.reset_btn.configure(state="disabled")
        self.stop_btn.configure(state="disabled")

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
