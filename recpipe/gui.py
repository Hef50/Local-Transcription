"""Tkinter window: drop files in, get a speaker-labelled transcript out.

The GUI never imports torch or whisperx. One long-lived worker process
(`app.py --worker --jsonl --serve`) takes JSON commands on stdin and streams
JSON progress events back. It is started when the window opens and loads the
libraries and models straight away, so a dropped file starts transcribing
immediately instead of waiting ~30 s for imports. The window stays responsive,
Cancel just kills the worker, and a crash in native code can't take the window
down with it.
"""
from __future__ import annotations

import ctypes
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from collections import deque
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import APP_NAME, config, deps, env
from .errors import DIARIZATION_URL, TOKENS_URL
from .formats import clock, human_duration
from .pipeline import DEFAULT_MODEL, DEFAULT_QUALITY, DIARIZATION, MODELS, QUALITY, SUPPORTED_EXTS, missing_components

try:
    from tkinterdnd2 import COPY, DND_FILES, TkinterDnD
except Exception:  # still usable (click to browse) without drag-and-drop
    TkinterDnD = None

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
AUTO_SPEAKERS = "Auto-detect"
NO_LABELS = "No speaker labels"
SPEAKER_CHOICES = [AUTO_SPEAKERS, NO_LABELS, "2", "3", "4", "5", "6", "7", "8"]
MODEL_HINTS = {"small.en": "fastest", "medium.en": "balanced", "large-v3": "most accurate, slowest"}
QUALITY_LABELS = {"fast": "Fastest", "balanced": "Balanced (recommended)", "best": "Most accurate"}

BG, CARD, BORDER = "#f5f6f8", "#ffffff", "#cfd6df"
ACCENT, ACCENT_SOFT = "#2f6fdf", "#eaf1fd"
TEXT, MUTED = "#1d2330", "#667085"
BANNERS = {  # background, foreground, icon
    "error": ("#fdeceb", "#a4262c", "✖"),
    "warning": ("#fff5dc", "#7a4a00", "⚠"),
    "info": ("#e9f2fe", "#1f4f99", "ℹ"),
    "success": ("#e8f5ec", "#1e6b36", "✔"),
}
SPEAKER_COLORS = ["#2f6fdf", "#c2410c", "#15803d", "#7e22ce", "#b91c1c", "#0e7490", "#a16207", "#be185d"]
HEADING_RE = re.compile(r"^(?:(?P<speaker>.+?) )?(?P<stamp>\[\d{2}:\d{2}(?::\d{2})? - \d{2}:\d{2}(?::\d{2})?\])$")


def human_eta(seconds: float) -> str:
    if seconds < 45:
        return "less than a minute left"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"about {minutes} min left"
    return f"about {minutes // 60} h {minutes % 60:02d} min left"


def short_path(path, limit: int = 60) -> str:
    text = str(path)
    return text if len(text) <= limit else f"{text[:18]}…{text[-(limit - 19):]}"


def worker_command(args: list[str]) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--worker", *args]
    return [deps.console_python(), str(env.APP_DIR / "app.py"), "--worker", *args]


class Worker:
    """One long-lived pipeline process. Commands go in as JSON lines; events come back tagged with the command id."""

    def __init__(self, events: queue.Queue, threads: int):
        self.events = events
        self.log: deque[str] = deque(maxlen=400)
        self.alive = True
        args = ["--jsonl", "--serve"] + (["--threads", str(threads)] if threads else [])
        environ = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self.proc = subprocess.Popen(
            worker_command(args), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=env.APP_DIR, env=environ, creationflags=NO_WINDOW)
        self._stderr = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr.start()
        threading.Thread(target=self._read_stdout, daemon=True).start()

    def send(self, command: dict) -> bool:
        try:
            self.proc.stdin.write((json.dumps(command) + "\n").encode("utf-8"))
            self.proc.stdin.flush()
            return True
        except (OSError, ValueError):
            self.alive = False
            return False

    def _read_stdout(self):
        for raw in self.proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                self.log.append(line)
                continue
            self.events.put((event.get("id"), event))
        code = self.proc.wait()
        self.alive = False
        self._stderr.join(timeout=2)
        self.events.put((None, {"event": "exit", "code": code}))

    def _read_stderr(self):
        for raw in self.proc.stderr:
            self.log.append(raw.decode("utf-8", "replace").rstrip())

    def kill(self):
        self.alive = False
        try:
            self.proc.kill()
        except OSError:
            pass


def _round_rect(canvas: tk.Canvas, x1, y1, x2, y2, r, **kwargs):
    points = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
              x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return canvas.create_polygon(points, smooth=True, **kwargs)


class App:
    def __init__(self, root: tk.Tk, dnd: bool):
        self.root, self.dnd = root, dnd
        self.cfg = config.load()
        self.events: queue.Queue = queue.Queue()
        self.pending: deque[Path] = deque()
        self.worker: Worker | None = None
        self.command_id = 0
        self.current: dict | None = None  # the job in flight
        self.warm_id = None
        self.warm_timer = None
        self.eta: tuple[float, float] | None = None
        self.outputs: dict = {}
        self.transcript = ""
        self.setup_ok = False
        self.banner_kind = None
        self.install_window = None
        self.status = ""
        self._build()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.report_callback_exception = self._on_tk_error
        root.after(150, self._poll)
        root.after(300, self.startup_checks)

    # ------------------------------------------------------------ layout

    def _build(self):
        root = self.root
        s = self.scale = root.winfo_fpixels("1i") / 96
        root.title(APP_NAME)
        root.configure(bg=BG)
        root.geometry(f"{int(900 * s)}x{int(780 * s)}")
        root.minsize(int(660 * s), int(600 * s))

        style = ttk.Style(root)
        style.configure(".", font=("Segoe UI", 10))
        for name in ("TFrame", "TLabel", "TCheckbutton", "TRadiobutton"):
            style.configure(name, background=BG)
        style.configure("TLabel", foreground=TEXT)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 16))
        style.configure("Section.TLabel", font=("Segoe UI Semibold", 11))

        outer = ttk.Frame(root, padding=(int(18 * s), int(14 * s)))
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(6, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text=APP_NAME, style="Title.TLabel").pack(side="left")
        ttk.Button(header, text="Settings…", command=self.open_settings).pack(side="right", padx=(14, 0))
        self.speakers_var = tk.StringVar(
            value=self.cfg["speakers"] if self.cfg["speakers"] in SPEAKER_CHOICES else AUTO_SPEAKERS)
        speakers = ttk.Combobox(header, textvariable=self.speakers_var, values=SPEAKER_CHOICES,
                                state="readonly", width=17)
        speakers.pack(side="right")
        ttk.Label(header, text="Speakers").pack(side="right", padx=(18, 6))
        self.model_hint = ttk.Label(header, style="Muted.TLabel")
        self.model_hint.pack(side="right", padx=(6, 0))
        self.model_var = tk.StringVar(value=self.cfg["model"] if self.cfg["model"] in MODELS else DEFAULT_MODEL)
        model = ttk.Combobox(header, textvariable=self.model_var, values=MODELS, state="readonly", width=10)
        model.pack(side="right")
        ttk.Label(header, text="Model").pack(side="right", padx=(0, 6))
        model.bind("<<ComboboxSelected>>", lambda e: self._model_changed())
        speakers.bind("<<ComboboxSelected>>", lambda e: self._remember_choices())
        self._update_model_hint()

        ttk.Label(outer, text="Runs entirely on this PC — audio and transcripts never leave your computer.",
                  style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(2, int(10 * s)))

        self.drop = tk.Canvas(outer, height=int(140 * s), bg=BG, highlightthickness=0, cursor="hand2")
        self.drop.grid(row=2, column=0, sticky="ew")
        self.drop.bind("<Configure>", lambda e: self._draw_drop())
        self.drop.bind("<Button-1>", lambda e: self.browse())

        self._build_banner(outer)
        self._build_progress(outer)

        heading = ttk.Frame(outer)
        heading.grid(row=5, column=0, sticky="ew", pady=(int(16 * s), int(6 * s)))
        ttk.Label(heading, text="Transcript", style="Section.TLabel").pack(side="left")
        self.file_label = ttk.Label(heading, style="Muted.TLabel")
        self.file_label.pack(side="left", padx=10)

        frame = tk.Frame(outer, bg=BORDER, padx=1, pady=1)
        frame.grid(row=6, column=0, sticky="nsew")
        self.text = tk.Text(frame, wrap="word", relief="flat", bg=CARD, fg=TEXT, font=("Segoe UI", 10),
                            padx=int(14 * s), pady=int(10 * s), spacing1=2, spacing3=2,
                            highlightthickness=0, insertwidth=0, cursor="arrow")
        scroll = ttk.Scrollbar(frame, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        self.text.tag_configure("meta", foreground=MUTED)
        self.text.tag_configure("stamp", foreground=MUTED, font=("Segoe UI", 9))
        self.text.tag_configure("placeholder", foreground=MUTED, justify="center", spacing1=int(40 * s))
        for i, color in enumerate(SPEAKER_COLORS):
            self.text.tag_configure(f"spk{i}", foreground=color, font=("Segoe UI Semibold", 10))
        self._set_placeholder("Your transcript will appear here.")

        actions = ttk.Frame(outer)
        actions.grid(row=7, column=0, sticky="ew", pady=(int(10 * s), 0))
        self.copy_btn = ttk.Button(actions, text="Copy to clipboard", command=self.copy_transcript)
        self.open_btn = ttk.Button(actions, text="Open transcript", command=lambda: self.open_output("txt"))
        self.srt_btn = ttk.Button(actions, text="Open .srt", command=lambda: self.open_output("srt"))
        self.folder_btn = ttk.Button(actions, text="Show in folder", command=self.show_in_folder)
        for button in (self.copy_btn, self.open_btn, self.srt_btn, self.folder_btn):
            button.pack(side="left", padx=(0, 8))
            button.state(["disabled"])
        self.queue_label = ttk.Label(actions, style="Muted.TLabel")
        self.queue_label.pack(side="right")

        if self.dnd:
            for widget in (self.drop, self.text):
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_drop)
                widget.dnd_bind("<<DropEnter>>", lambda e: self._drop_hover(e, True))
                widget.dnd_bind("<<DropLeave>>", lambda e: self._drop_hover(e, False))

    def _build_banner(self, parent):
        s = self.scale
        self.banner = tk.Frame(parent, padx=int(12 * s), pady=int(8 * s))
        self.banner.columnconfigure(1, weight=1)
        self.banner_icon = tk.Label(self.banner, font=("Segoe UI", 12))
        self.banner_icon.grid(row=0, column=0, rowspan=3, sticky="n", padx=(0, int(10 * s)))
        self.banner_title = tk.Label(self.banner, font=("Segoe UI Semibold", 10), anchor="w", justify="left")
        self.banner_title.grid(row=0, column=1, sticky="ew")
        self.banner_hint = tk.Label(self.banner, anchor="w", justify="left", fg=TEXT)
        self.banner_hint.grid(row=1, column=1, sticky="ew", pady=(2, 0))
        self.banner_actions = tk.Frame(self.banner)
        self.banner_actions.grid(row=2, column=1, sticky="w", pady=(int(6 * s), 0))
        self.banner_close = tk.Label(self.banner, text="✕", cursor="hand2", font=("Segoe UI", 10))
        self.banner_close.grid(row=0, column=2, sticky="ne", padx=(int(10 * s), 0))
        self.banner_close.bind("<Button-1>", lambda e: self.hide_banner())
        self.banner.bind("<Configure>", lambda e: self._wrap_banner(e.width))

    def _build_progress(self, parent):
        s = self.scale
        self.progress = ttk.Frame(parent)
        self.progress.columnconfigure(0, weight=1)
        self.stage_label = ttk.Label(self.progress, style="Section.TLabel")
        self.stage_label.grid(row=0, column=0, sticky="w")
        self.pct_label = ttk.Label(self.progress, style="Muted.TLabel")
        self.pct_label.grid(row=0, column=1, sticky="e")
        self.bar = ttk.Progressbar(self.progress, maximum=100)
        self.bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=int(6 * s))
        self.time_label = ttk.Label(self.progress, style="Muted.TLabel")
        self.time_label.grid(row=2, column=0, sticky="w")
        self.cancel_btn = ttk.Button(self.progress, text="Cancel", command=self.cancel)
        self.cancel_btn.grid(row=2, column=1, sticky="e")
        self.progress.grid(row=4, column=0, sticky="ew", pady=(int(14 * s), 0))
        self.progress.grid_remove()

    def _draw_drop(self, active: bool = False):
        c, s = self.drop, self.scale
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10:
            return
        _round_rect(c, 2, 2, w - 2, h - 2, int(14 * s), fill=ACCENT_SOFT if active else CARD,
                    outline=ACCENT if active else BORDER, width=2 if active else 1)
        title = "Release to transcribe" if active else "Drop audio or video files here"
        c.create_text(w / 2, h / 2 - 13 * s, text=title, font=("Segoe UI Semibold", 14),
                      fill=ACCENT if active else TEXT)
        how = "or click to browse" if self.dnd else "click to choose files (drag-and-drop unavailable)"
        c.create_text(w / 2, h / 2 + 17 * s, text=f"{how}  ·  wav, mp3, m4a, mp4 and more",
                      font=("Segoe UI", 10), fill=MUTED)

    def _drop_hover(self, event, active: bool):
        self._draw_drop(active)
        return event.action

    def _on_drop(self, event):
        self._draw_drop(False)
        self.add_files(self.root.tk.splitlist(event.data))
        return COPY

    # ------------------------------------------------------------ banner

    def show_banner(self, kind: str, title: str, hint: str = "", actions=()):
        bg, fg, icon = BANNERS[kind]
        self.banner_kind = kind
        for widget in (self.banner, self.banner_icon, self.banner_title, self.banner_hint,
                       self.banner_actions, self.banner_close):
            widget.configure(bg=bg)
        self.banner_icon.configure(text=icon, fg=fg)
        self.banner_title.configure(text=title, fg=fg)
        self.banner_close.configure(fg=fg)
        self.banner_hint.configure(text=hint)
        self.banner_hint.grid() if hint else self.banner_hint.grid_remove()
        for child in self.banner_actions.winfo_children():
            child.destroy()
        for label, callback in actions:
            ttk.Button(self.banner_actions, text=label, command=callback).pack(side="left", padx=(0, 6))
        self.banner_actions.grid() if actions else self.banner_actions.grid_remove()
        self.banner.grid(row=3, column=0, sticky="ew", pady=(int(10 * self.scale), 0))

    def hide_banner(self):
        self.banner_kind = None
        self.banner.grid_remove()

    def _wrap_banner(self, width):
        wrap = max(width - int(90 * self.scale), 200)
        self.banner_title.configure(wraplength=wrap)
        self.banner_hint.configure(wraplength=wrap)

    def show_details(self, title: str, text: str):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry(f"{int(760 * self.scale)}x{int(460 * self.scale)}")
        box = tk.Text(win, wrap="none", font=("Consolas", 9), relief="flat", padx=8, pady=8)
        scroll = ttk.Scrollbar(win, command=box.yview)
        box.configure(yscrollcommand=scroll.set)
        bar = ttk.Frame(win, padding=8)
        bar.pack(side="bottom", fill="x")
        ttk.Button(bar, text="Copy", command=lambda: self._to_clipboard(text)).pack(side="right")
        scroll.pack(side="right", fill="y")
        box.pack(fill="both", expand=True)
        box.insert("1.0", text)
        box.see("end")
        box.configure(state="disabled")

    def _on_tk_error(self, exc_type, exc, tb):
        detail = "".join(traceback.format_exception(exc_type, exc, tb))
        self.show_banner("error", f"Unexpected problem in the app window: {exc}",
                         "The app is still running. Please try again.",
                         [("Details…", lambda: self.show_details("Error details", detail))])

    # ------------------------------------------------------------ setup checks

    def startup_checks(self) -> bool:
        missing = deps.missing_packages()
        if missing:
            self.setup_ok = False
            if deps.in_virtualenv():
                self.show_banner("error", f"Setup needed: missing Python packages ({', '.join(missing)}).",
                                 "Click Install to download them (about 1 GB, one time).",
                                 [("Install", self.install_dependencies)])
            else:
                self.show_banner("error", f"Setup needed: missing Python packages ({', '.join(missing)}).",
                                 "Close this window and double-click run.bat (or run setup.ps1) in the app "
                                 "folder. It creates a private Python environment and installs everything.")
            return False
        if deps.ffmpeg_missing():
            self.setup_ok = False
            command = "winget install Gyan.FFmpeg"
            self.show_banner("error", "ffmpeg wasn't found. It's needed to read audio and video files.",
                             f"Install it from a terminal with:   {command}   then click Check again "
                             "(you may need to restart the app so it sees the new PATH).",
                             [("Copy command", lambda: self._to_clipboard(command)),
                              ("Check again", self.startup_checks)])
            return False
        was_blocked = not self.setup_ok and self.banner_kind == "error"
        self.setup_ok = True
        if was_blocked:
            self.show_banner("success", "Setup complete. Drop a file to start.")
        if not self.cfg["token_prompt_done"] and not self._diarization_ready():
            self.open_settings(welcome=True)
        self.warm_up()
        return True

    def install_dependencies(self):
        if self.install_window is not None:
            return
        win = self.install_window = tk.Toplevel(self.root)
        win.title("Installing dependencies")
        win.geometry(f"{int(760 * self.scale)}x{int(420 * self.scale)}")
        self.install_log = tk.Text(win, wrap="none", font=("Consolas", 9), relief="flat", padx=8, pady=8)
        self.install_log.pack(fill="both", expand=True)
        command = deps.install_command()
        self.install_log.insert("end", "> " + " ".join(command) + "\n")

        def run():
            try:
                proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        cwd=env.APP_DIR, env=deps.install_environment(),
                                        creationflags=NO_WINDOW)
                for raw in proc.stdout:
                    self.events.put((-1, {"event": "install_log", "line": raw.decode("utf-8", "replace")}))
                code = proc.wait()
            except OSError as exc:
                self.events.put((-1, {"event": "install_log", "line": f"{exc}\n"}))
                code = -1
            self.events.put((-1, {"event": "install_exit", "code": code}))

        threading.Thread(target=run, daemon=True).start()
        self.show_banner("info", "Installing dependencies…", "This can take several minutes.")

    def _install_event(self, event: dict):
        if event["event"] == "install_log" and self.install_window is not None:
            self.install_log.insert("end", event["line"])
            self.install_log.see("end")
        elif event["event"] == "install_exit":
            if self.install_window is not None:
                self.install_window.destroy()
                self.install_window = None
            if event["code"] == 0 and not deps.missing_packages():
                self.startup_checks()
            else:
                self.show_banner("error", "Installing the dependencies failed.",
                                 "Try running setup.ps1 from a terminal to see the full output.")

    def _diarization_ready(self) -> bool:
        return DIARIZATION not in missing_components(config.models_dir(self.cfg), self.model_var.get(), True)

    # ------------------------------------------------------------ worker

    def _get_worker(self) -> Worker | None:
        if self.worker is None or not self.worker.alive:
            try:
                self.worker = Worker(self.events, int(self.cfg["threads"] or 0))
            except OSError as exc:
                self.worker = None
                self.show_banner("error", "Couldn't start the transcription process.", str(exc))
                return None
        return self.worker

    def _dispatch(self, command: dict, meta: dict | None):
        """Send a command; `meta` is set for jobs the progress panel should track."""
        worker = self._get_worker()
        if worker is None:
            return None
        self.command_id += 1
        command["id"] = self.command_id
        if meta is not None:
            self.current = dict(meta, id=self.command_id, started=time.monotonic(), duration=None, cancelled=False,
                                tracks={})
            self.eta = None
            self._indeterminate(False)
            self.bar.configure(value=0)
            self.stage_label.configure(text="Starting…")
            self.pct_label.configure(text="")
            self.time_label.configure(text="")
            self.cancel_btn.grid()
            self.progress.grid()
        if not worker.send(command):
            self.worker = None
            if meta is not None:
                self.current = None
                self.progress.grid_remove()
            self.show_banner("error", "The transcription process isn't responding.", "Please try again.")
            return None
        self._update_queue_label()
        return self.command_id

    def _common_fields(self, diarize: bool) -> dict:
        return {"model": self.model_var.get(), "diarize": diarize, "quality": self.cfg.get("quality", DEFAULT_QUALITY),
                "language": self.cfg["language"]}

    def warm_up(self):
        """Load libraries and the current models in the background, so the next drop starts at once."""
        if self.warm_timer is not None:
            self.root.after_cancel(self.warm_timer)
            self.warm_timer = None
        if not self.setup_ok or self.current is not None:
            return
        diarize = self.speakers_var.get() != NO_LABELS and (bool(self.cfg["hf_token"]) or self._diarization_ready())
        self.warm_id = self._dispatch({"cmd": "warm", **self._common_fields(diarize)}, None)
        if self.warm_id is not None:
            self._set_status("Loading models in the background…")

    def _schedule_warm_up(self):
        if self.warm_timer is not None:
            self.root.after_cancel(self.warm_timer)
        self.warm_timer = self.root.after(1500, self.warm_up)

    def _restart_worker(self):
        """Settings that only apply at start-up (threads) changed: drop the worker, warm a new one."""
        if self.current is not None:
            return
        if self.worker is not None:
            self.worker.kill()
            self.worker = None
        self.warm_id = None
        self.warm_up()

    # ------------------------------------------------------------ queue + jobs

    def browse(self):
        patterns = " ".join(f"*{ext}" for ext in SUPPORTED_EXTS)
        files = filedialog.askopenfilenames(title="Choose recordings to transcribe",
                                            filetypes=[("Audio and video", patterns), ("All files", "*.*")])
        if files:
            self.add_files(files)

    def add_files(self, names):
        good, bad = [], []
        for name in names:
            path = Path(name)
            (good if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS else bad).append(path)
        if bad:
            more = f" (and {len(bad) - 1} more)" if len(bad) > 1 else ""
            self.show_banner("error", f"Can't transcribe “{bad[0].name}”{more}: not a supported audio or video file.",
                             "Supported: " + ", ".join(e.lstrip(".") for e in SUPPORTED_EXTS) + ".")
        self.pending.extend(good)
        self.start_next()

    def start_next(self):
        self._update_queue_label()
        if self.current is not None or not self.pending:
            return
        if not self.setup_ok and not self.startup_checks():
            return
        diarize = self.speakers_var.get() != NO_LABELS
        if diarize and not self.cfg["hf_token"] and not self._diarization_ready():
            self.show_banner(
                "warning", "Speaker labels need a one-time model download with a free Hugging Face token.",
                "Add your token in Settings, or transcribe without speaker labels for now.",
                [("Add token…", self.open_settings),
                 ("Transcribe without labels", self._continue_without_labels),
                 ("Clear queue", self._clear_queue)])
            return
        self._start_transcription(self.pending.popleft(), diarize)

    def _continue_without_labels(self):
        self.speakers_var.set(NO_LABELS)
        self._remember_choices()
        self.hide_banner()
        self.start_next()

    def _clear_queue(self):
        self.pending.clear()
        self.hide_banner()
        self._update_queue_label()

    def _start_transcription(self, source: Path, diarize: bool):
        command = {"cmd": "job", "file": str(source), **self._common_fields(diarize)}
        speakers = self.speakers_var.get()
        if diarize and speakers.isdigit():
            command["speakers"] = int(speakers)
        if self.cfg["output_mode"] == "folder" and self.cfg["output_dir"]:
            command["output_dir"] = self.cfg["output_dir"]
        if self.banner_kind in ("warning", "info", "success"):
            self.hide_banner()  # errors about an earlier file stay visible until dismissed
        self.file_label.configure(text=source.name)
        self._set_result(None, "")
        self._set_placeholder(f"Working on {source.name}…\nThe transcript will appear here when it's done.")
        self._dispatch(command, {"kind": "transcribe", "source": source})

    def start_prepare(self):
        if self.current is not None:
            messagebox.showinfo(APP_NAME, "Please wait until the current transcription finishes.", parent=self.root)
            return
        if not self.setup_ok and not self.startup_checks():
            return
        diarize = bool(self.cfg["hf_token"]) or self._diarization_ready()
        self._dispatch({"cmd": "prepare", **self._common_fields(diarize)}, {"kind": "prepare", "source": None})

    def cancel(self):
        if self.current is None or self.worker is None:
            return
        name = self.current["source"].name if self.current.get("source") else "the model download"
        extra = f"\n\nThe {len(self.pending)} queued file(s) will be removed too." if self.pending else ""
        if not messagebox.askyesno(APP_NAME, f"Stop working on {name}?{extra}", parent=self.root):
            return
        self.current["cancelled"] = True
        self.pending.clear()
        self.stage_label.configure(text="Cancelling…")
        self.worker.kill()  # the exit event finishes the job; the next job starts a fresh worker

    # ------------------------------------------------------------ events

    def _poll(self):
        try:
            while True:
                ident, event = self.events.get_nowait()
                if ident == -1:
                    self._install_event(event)
                elif event.get("event") == "exit":
                    self._on_worker_exit(event.get("code"))
                elif self.current is not None and ident == self.current["id"]:
                    self._handle(event)
                elif ident is not None and ident == self.warm_id:
                    self._handle_warm(event)
        except queue.Empty:
            pass
        self._tick()
        self.root.after(200, self._poll)

    def _handle(self, event: dict):
        kind = event.get("event")
        if kind == "stage":
            self.current["tracks"][event.get("track", "main")] = event.get("label", "")
            labels = [label for track in ("main", "diarize") if (label := self.current["tracks"].get(track))]
            self.stage_label.configure(text="   +   ".join(labels))
            if event.get("stage") in ("download", "decode"):
                self._indeterminate(True)
                self.pct_label.configure(text="")
        elif kind == "progress":
            value = event.get("overall_pct")
            value = event.get("stage_pct") if value is None else value
            if value is None:
                self._indeterminate(True)
            else:
                self._indeterminate(False)
                self.bar.configure(value=value)
                detail = event.get("detail")
                self.pct_label.configure(text=f"{detail}  ·  {value:.0f}%" if detail else f"{value:.0f}%")
            eta = event.get("eta_s")
            self.eta = (eta, time.monotonic()) if eta is not None else None
        elif kind == "info":
            self.current["duration"] = event.get("duration_s")
        elif kind == "warning":
            self.show_banner("warning", event.get("message", ""), event.get("hint", ""))
        elif kind == "done":
            self._finish(done=event)
        elif kind == "error":
            self._finish(error=event)

    def _handle_warm(self, event: dict):
        kind = event.get("event")
        if kind == "stage":
            self._set_status(event.get("label", ""))
        elif kind in ("warm_done", "error"):
            self.warm_id = None
            self._set_status("Ready" if kind == "warm_done" else "")

    def _tick(self):
        if self.current is None:
            return
        parts = [f"Elapsed {clock(time.monotonic() - self.current['started'], False)}"]
        if self.eta is not None:
            eta, received = self.eta
            left = eta - (time.monotonic() - received)
            parts.append(human_eta(left) if left > 0 else "finishing up…")
        if self.current.get("duration"):
            parts.append(f"{human_duration(self.current['duration'])} of audio")
        self.time_label.configure(text="   ·   ".join(parts))

    def _on_worker_exit(self, code):
        self.worker = None
        self.warm_id = None
        self._set_status("")
        if self.current is not None:
            self._finish(crashed=code)

    def _finish(self, done: dict | None = None, error: dict | None = None, crashed=None):
        job, self.current, self.eta = self.current, None, None
        self._indeterminate(False)
        self.cancel_btn.grid_remove()
        log = "\n".join(self.worker.log) if self.worker is not None else ""
        if job["cancelled"]:
            self.progress.grid_remove()
            self._set_placeholder("Cancelled.")
            self.show_banner("info", "Cancelled.")
        elif done is not None and job["kind"] == "prepare":
            self.progress.grid_remove()
            got = done.get("downloaded") or []
            self.show_banner("success", "Models are ready: the app now works without an internet connection."
                             if got else "Everything for these settings was already downloaded.")
        elif done is not None:
            self._on_success(job, done)
        else:
            self.progress.grid_remove()
            if job["kind"] == "transcribe":
                self._set_placeholder("No transcript: see the message above.")
            self._show_error(error or {
                "code": "WORKER_CRASHED",
                "message": f"The transcription process stopped unexpectedly (exit code {crashed}).",
                "hint": "If the computer ran low on memory, try the small.en model. Details has the full log.",
            }, log)
        self.start_next()

    def _on_success(self, job: dict, done: dict):
        try:
            text = Path(done["txt"]).read_text(encoding="utf-8")
        except OSError as exc:
            self.progress.grid_remove()
            self.show_banner("error", "The transcript was written but couldn't be read back.", str(exc))
            return
        self.bar.configure(value=100)
        self.pct_label.configure(text="100%")
        speakers = (f"{done['speakers']} speaker{'s' if done['speakers'] != 1 else ''}"
                    if done.get("diarized") else "no speaker labels")
        self.stage_label.configure(text=f"Done — {job['source'].name}")
        self.time_label.configure(text=f"{human_duration(done['duration_s'])} of audio transcribed in "
                                       f"{human_duration(done['elapsed_s'])}   ·   {speakers}   ·   "
                                       f"saved in {short_path(Path(done['txt']).parent)}")
        self._set_result(done, text)

    def _show_error(self, err: dict, log: str):
        code = err.get("code")
        actions = []
        if code in ("HF_TOKEN_MISSING", "HF_TOKEN_INVALID"):
            actions += [("Open Settings", self.open_settings), ("Get a token", lambda: webbrowser.open(TOKENS_URL))]
        if code in ("HF_GATED", "HF_TOKEN_MISSING"):
            actions.append(("Open model page", lambda: webbrowser.open(DIARIZATION_URL)))
        if code == "OUTPUT_NOT_WRITABLE":
            actions.append(("Open Settings", self.open_settings))
        details = "\n\n".join(part for part in (err.get("detail"), log) if part)
        if details:
            actions.append(("Details…", lambda: self.show_details("Error details", details)))
        self.show_banner("error", err.get("message", "Something went wrong."), err.get("hint", ""), actions)

    # ------------------------------------------------------------ transcript display

    def _set_placeholder(self, message: str):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", message, "placeholder")
        self.text.configure(state="disabled")

    def _set_result(self, done: dict | None, text: str):
        self.outputs, self.transcript = (done or {}), text
        for button in (self.copy_btn, self.open_btn, self.srt_btn, self.folder_btn):
            button.state(["!disabled"] if done else ["disabled"])
        if not done:
            return
        box = self.text
        box.configure(state="normal")
        box.delete("1.0", "end")
        colors: dict[str, str] = {}
        in_header = True
        for line in text.splitlines():
            if in_header:
                if line:
                    box.insert("end", line + "\n", "meta")
                    continue
                in_header = False
            match = HEADING_RE.match(line)
            if match:
                name = match.group("speaker")
                if name:
                    tag = colors.setdefault(name, f"spk{len(colors) % len(SPEAKER_COLORS)}")
                    box.insert("end", name + "  ", tag)
                box.insert("end", match.group("stamp") + "\n", "stamp")
            else:
                box.insert("end", line + "\n")
        box.configure(state="disabled")
        box.yview_moveto(0)

    def _to_clipboard(self, text: str):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()

    def copy_transcript(self):
        if not self.transcript:
            return
        self._to_clipboard(self.transcript)
        self.copy_btn.configure(text="Copied ✔")
        self.root.after(1500, lambda: self.copy_btn.configure(text="Copy to clipboard"))

    def _output_path(self, kind: str) -> Path | None:
        path = Path(self.outputs.get(kind, ""))
        if not self.outputs.get(kind) or not path.exists():
            self.show_banner("error", "That file isn't there any more.", "It may have been moved or deleted.")
            return None
        return path

    def open_output(self, kind: str):
        path = self._output_path(kind)
        if path is None:
            return
        try:
            os.startfile(path)
        except OSError:  # e.g. no app registered for .srt
            subprocess.Popen(["notepad.exe", str(path)])

    def show_in_folder(self):
        path = self._output_path("txt")
        if path is not None:
            subprocess.Popen(f'explorer /select,"{path}"')

    # ------------------------------------------------------------ misc

    def _indeterminate(self, on: bool):
        if on and str(self.bar.cget("mode")) != "indeterminate":
            self.bar.configure(mode="indeterminate")
            self.bar.start(15)
        elif not on and str(self.bar.cget("mode")) == "indeterminate":
            self.bar.stop()
            self.bar.configure(mode="determinate")

    def _set_status(self, text: str):
        self.status = text
        self._update_queue_label()

    def _update_queue_label(self):
        count = len(self.pending)
        if count:
            text = (f"{count} more file{'s' if count != 1 else ''} queued" if self.current is not None
                    else f"{count} file{'s' if count != 1 else ''} waiting")
        else:
            text = self.status
        self.queue_label.configure(text=text)

    def _update_model_hint(self):
        self.model_hint.configure(text=MODEL_HINTS.get(self.model_var.get(), ""))

    def _model_changed(self):
        self._remember_choices()
        self._schedule_warm_up()

    def _remember_choices(self):
        self._update_model_hint()
        self.cfg["model"] = self.model_var.get()
        self.cfg["speakers"] = self.speakers_var.get()
        self._save_config()

    def _save_config(self) -> bool:
        try:
            config.save(self.cfg)
            return True
        except OSError as exc:
            self.show_banner("error", "Couldn't save settings.", str(exc))
            return False

    def open_settings(self, welcome: bool = False):
        SettingsDialog(self, welcome)

    def on_settings_saved(self, threads_changed: bool = False):
        if self.banner_kind == "warning" and self.pending:
            self.hide_banner()
        if threads_changed:
            self._restart_worker()
        elif not self.pending:
            self._schedule_warm_up()
        self.start_next()

    def on_close(self):
        if self.current is not None:
            if not messagebox.askyesno(APP_NAME, "A transcription is still running. Stop it and quit?",
                                       parent=self.root):
                return
        if self.worker is not None:
            self.worker.kill()
        self.root.destroy()


class SettingsDialog(tk.Toplevel):
    def __init__(self, app: App, welcome: bool):
        super().__init__(app.root)
        self.app, self.welcome = app, welcome
        cfg, s = app.cfg, app.scale
        self.title(f"Welcome to {APP_NAME}" if welcome else "Settings")
        self.configure(bg=BG)
        self.transient(app.root)
        self.resizable(False, False)
        wrap = int(500 * s)

        body = ttk.Frame(self, padding=int(20 * s))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        row = 0

        def section(title, top=16):
            nonlocal row
            ttk.Label(body, text=title, style="Section.TLabel").grid(
                row=row, column=0, columnspan=3, sticky="w", pady=(int(top * s), int(4 * s)))
            row += 1

        def note(text, style="Muted.TLabel"):
            nonlocal row
            ttk.Label(body, text=text, style=style, wraplength=wrap, justify="left").grid(
                row=row, column=0, columnspan=3, sticky="w", pady=(0, int(4 * s)))
            row += 1

        if welcome:
            ttk.Label(body, text="Welcome!", style="Title.TLabel").grid(row=row, column=0, columnspan=3, sticky="w")
            row += 1
            note("Transcription works right away. Speaker labels (Speaker 1, Speaker 2…) use the free "
                 "pyannote model, which needs a Hugging Face access token for a one-time download.", "TLabel")

        section("Speaker labels", top=8 if welcome else 0)
        for text, url, button in (
                ("1.  Create a free Hugging Face account and a token with “Read” access", TOKENS_URL, "Token page"),
                ("2.  Accept the conditions of pyannote/speaker-diarization-community-1", DIARIZATION_URL, "Model page")):
            ttk.Label(body, text=text, wraplength=int(390 * s), justify="left").grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
            ttk.Button(body, text=button, command=lambda u=url: webbrowser.open(u)).grid(row=row, column=2, sticky="e", padx=(8, 0))
            row += 1
        ttk.Label(body, text="3.  Paste the token:").grid(row=row, column=0, sticky="w", pady=(6, 2))
        self.token_var = tk.StringVar(value=cfg["hf_token"])
        self.token_entry = ttk.Entry(body, textvariable=self.token_var, show="•", width=44)
        self.token_entry.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=(6, 2))
        self.show_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(body, text="Show", variable=self.show_var,
                        command=lambda: self.token_entry.configure(show="" if self.show_var.get() else "•")
                        ).grid(row=row, column=2, sticky="e", padx=(8, 0), pady=(6, 2))
        row += 1
        status = ("Speaker model: downloaded ✔ (the token is no longer needed for it)"
                  if app._diarization_ready() else "Speaker model: not downloaded yet")
        note("The token is stored only in config.json in the app folder and is sent only to huggingface.co "
             f"to download the model. Your audio never leaves this PC.\n{status}")

        if not welcome:
            section("Transcript files")
            self.mode_var = tk.StringVar(value=cfg["output_mode"])
            ttk.Radiobutton(body, text="Save next to the original file", value="source",
                            variable=self.mode_var).grid(row=row, column=0, columnspan=3, sticky="w")
            row += 1
            ttk.Radiobutton(body, text="Save in this folder:", value="folder",
                            variable=self.mode_var).grid(row=row, column=0, sticky="w")
            self.dir_var = tk.StringVar(value=cfg["output_dir"])
            ttk.Entry(body, textvariable=self.dir_var).grid(row=row, column=1, sticky="ew", padx=(8, 0))
            ttk.Button(body, text="Browse…", command=self._browse_output).grid(row=row, column=2, sticky="e", padx=(8, 0))
            row += 1

            section("Transcription")
            ttk.Label(body, text="Speed vs accuracy").grid(row=row, column=0, sticky="w", pady=2)
            quality = cfg.get("quality") if cfg.get("quality") in QUALITY else DEFAULT_QUALITY
            self.quality_var = tk.StringVar(value=QUALITY_LABELS[quality])
            ttk.Combobox(body, textvariable=self.quality_var, values=list(QUALITY_LABELS.values()),
                         state="readonly", width=24).grid(row=row, column=1, sticky="w", padx=(8, 0), pady=2)
            row += 1
            note("Fastest: greedy decoding and coarser speaker scanning. Most accurate: the slow library "
                 "defaults. Balanced keeps the accuracy of the defaults where it measurably matters.")
            ttk.Label(body, text="Language (multilingual models)").grid(row=row, column=0, sticky="w", pady=2)
            self.language_var = tk.StringVar(value="Auto-detect" if cfg["language"] == "auto" else "English")
            ttk.Combobox(body, textvariable=self.language_var, values=["English", "Auto-detect"],
                         state="readonly", width=14).grid(row=row, column=1, sticky="w", padx=(8, 0), pady=2)
            row += 1
            ttk.Label(body, text="CPU threads").grid(row=row, column=0, sticky="w", pady=2)
            self.threads_var = tk.StringVar(value=str(cfg["threads"] or 0))
            ttk.Spinbox(body, from_=0, to=os.cpu_count() or 16, textvariable=self.threads_var, width=6
                        ).grid(row=row, column=1, sticky="w", padx=(8, 0), pady=2)
            row += 1
            note(f"0 = automatic ({max(4, (os.cpu_count() or 6) - 2)} on this PC).")

            section("Models")
            models_dir = config.models_dir(cfg)
            note(f"Stored in {models_dir}")
            buttons = ttk.Frame(body)
            buttons.grid(row=row, column=0, columnspan=3, sticky="w")
            row += 1
            ttk.Button(buttons, text="Open models folder",
                       command=lambda: (models_dir.mkdir(parents=True, exist_ok=True), os.startfile(models_dir))
                       ).pack(side="left", padx=(0, 8))
            ttk.Button(buttons, text="Download models now", command=lambda: self._save(then=app.start_prepare)
                       ).pack(side="left")
            note("Downloads the selected speech model plus the timing and speaker models, so the app "
                 "works with no internet connection afterwards.")

        footer = ttk.Frame(body)
        footer.grid(row=row, column=0, columnspan=3, sticky="e", pady=(int(18 * s), 0))
        ttk.Button(footer, text="Save", command=self._save).pack(side="right")
        ttk.Button(footer, text="Skip for now" if welcome else "Cancel", command=self._dismiss
                   ).pack(side="right", padx=(0, 8))
        self.protocol("WM_DELETE_WINDOW", self._dismiss)
        self.bind("<Escape>", lambda e: self._dismiss())

        self.update_idletasks()
        x = app.root.winfo_rootx() + (app.root.winfo_width() - self.winfo_width()) // 2
        y = app.root.winfo_rooty() + max((app.root.winfo_height() - self.winfo_height()) // 3, 0)
        self.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        self.grab_set()
        self.token_entry.focus_set()

    def _browse_output(self):
        folder = filedialog.askdirectory(parent=self, title="Save transcripts in")
        if folder:
            self.dir_var.set(folder)
            self.mode_var.set("folder")

    def _dismiss(self):
        if self.welcome:
            self.app.cfg["token_prompt_done"] = True
            self.app._save_config()
        self.destroy()

    def _save(self, then=None):
        cfg = dict(self.app.cfg)
        token = self.token_var.get().strip()
        if token and not token.startswith("hf_") and not messagebox.askyesno(
                APP_NAME, "Hugging Face access tokens start with “hf_”. Save this value anyway?", parent=self):
            return
        cfg["hf_token"] = token
        cfg["token_prompt_done"] = True
        if not self.welcome:
            if self.mode_var.get() == "folder" and not self.dir_var.get().strip():
                messagebox.showerror(APP_NAME, "Choose a folder for transcripts, or save next to the original file.",
                                     parent=self)
                return
            cfg["output_mode"] = self.mode_var.get()
            cfg["output_dir"] = self.dir_var.get().strip()
            cfg["language"] = "auto" if self.language_var.get() == "Auto-detect" else "en"
            cfg["quality"] = next((k for k, v in QUALITY_LABELS.items() if v == self.quality_var.get()), DEFAULT_QUALITY)
            try:
                cfg["threads"] = max(0, int(self.threads_var.get() or 0))
            except ValueError:
                messagebox.showerror(APP_NAME, "CPU threads must be a whole number (0 = automatic).", parent=self)
                return
        threads_changed = cfg["threads"] != self.app.cfg["threads"]
        self.app.cfg.update(cfg)
        if not self.app._save_config():
            return
        self.destroy()
        self.app.on_settings_saved(threads_changed)
        if then:
            then()


def _enable_dpi_awareness():
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # crisp text on scaled displays
        except (AttributeError, OSError):
            pass


def main():
    _enable_dpi_awareness()
    root, dnd = None, False
    if TkinterDnD is not None:
        try:
            root, dnd = TkinterDnD.Tk(), True
        except (tk.TclError, RuntimeError):
            root = None
    if root is None:
        root = tk.Tk()
    icon = env.APP_DIR / "assets" / "app.ico"
    if icon.exists():
        try:
            root.iconbitmap(default=str(icon))  # also used by the dialogs
        except tk.TclError:
            pass
    App(root, dnd)
    root.mainloop()
