"""Tkinter window: drop files in, get a speaker-labelled transcript out.

Look: the Sun Valley ttk theme (Windows 11 style) following the Windows light/dark
setting, Segoe Fluent icons, and Pillow-rendered antialiased artwork for the drop
zone. Every piece degrades gracefully: without sv-ttk the stock theme is used,
without Pillow the icons are left out.

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
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from collections import deque
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

from . import APP_NAME, config, deps, env
from .errors import DIARIZATION_URL, TOKENS_URL
from .formats import (SpeakerSummary, clock, human_duration, name_problem, rename_outputs, speakers_in,
                      transcript_siblings)
from .pipeline import (DEFAULT_MODEL, DEFAULT_QUALITY, DIARIZATION, MODELS, QUALITY, SUPPORTED_EXTS, Job,
                       load_calibration, missing_components, plan_stages)

try:
    from tkinterdnd2 import COPY, DND_FILES, TkinterDnD
except Exception:  # still usable (click to browse) without drag-and-drop
    TkinterDnD = None
try:
    import sv_ttk
except Exception:  # falls back to the stock Windows look
    sv_ttk = None
try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except Exception:  # icons and antialiased artwork are optional
    Image = None
try:
    import winsound  # plays voice samples in the Name speakers dialog
except ImportError:
    winsound = None

# Taskbar identity. create_shortcut.ps1 stamps the same ID on the shortcut, so the
# running window groups under the pinned icon instead of a generic Python button.
APP_ID = "LocalTranscriber.App"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
AUTO_SPEAKERS = "Auto-detect"
NO_LABELS = "No speaker labels"
SPEAKER_CHOICES = [AUTO_SPEAKERS, NO_LABELS, "2", "3", "4", "5", "6", "7", "8"]
MODEL_HINTS = {"small.en": "Fastest, fine for clear audio", "medium.en": "Recommended balance",
               "large-v3": "Most accurate, slowest"}
QUALITY_LABELS = {"fast": "Fastest", "balanced": "Balanced", "best": "Most accurate"}
QUALITY_HINTS = {"fast": "Quickest speaker detection", "balanced": "Recommended",
                 "best": "Slowest, library defaults"}
THEME_LABELS = {"system": "Match Windows", "light": "Light", "dark": "Dark"}

PALETTES = {
    "dark": {
        "bg": "#1c1c1c", "surface": "#242424", "surface_active": "#1d2a35", "border": "#353535",
        "text": "#fafafa", "muted": "#a3a3a3", "subtle": "#737373", "accent": "#57c8ff", "on_accent": "#000000",
        "select": "#264f78",
        "speakers": ["#6cb8ff", "#ffb06a", "#7fd88f", "#d7a6ff", "#ff9b9b", "#66d9e8", "#f2cc60", "#ff94c9"],
        "banners": {"error": ("#442726", "#ffb4ab"), "warning": ("#433519", "#fcd581"),
                    "info": ("#1d3349", "#9ccfff"), "success": ("#1e3a26", "#8fd9a2")},
        "status": {"ok": "#6ccb5f", "busy": "#fcd581", "work": "#57c8ff", "off": "#737373", "error": "#ff99a4"},
    },
    "light": {
        "bg": "#fafafa", "surface": "#ffffff", "surface_active": "#eaf3fc", "border": "#e0e0e0",
        "text": "#1c1c1c", "muted": "#5c5c5c", "subtle": "#999999", "accent": "#005fb8", "on_accent": "#ffffff",
        "select": "#cce4f7",
        "speakers": ["#0f62c4", "#c2410c", "#15803d", "#7e22ce", "#b91c1c", "#0e7490", "#a16207", "#be185d"],
        "banners": {"error": ("#fde7e9", "#a4262c"), "warning": ("#fff4ce", "#7a4f00"),
                    "info": ("#e5f1fb", "#0f5499"), "success": ("#dff6dd", "#0e700e")},
        "status": {"ok": "#0e7a0d", "busy": "#c18a00", "work": "#005fb8", "off": "#999999", "error": "#c42b1c"},
    },
}
ICONS = {  # Segoe Fluent Icons / Segoe MDL2 Assets code points
    "settings": "\uE713", "copy": "\uE8C8", "open": "\uE7C3", "captions": "\uE7F0", "folder": "\uE838",
    "cancel": "\uE711", "check": "\uE73E", "warning": "\uE7BA", "info": "\uE946", "error": "\uEA39",
    "success": "\uE930", "link": "\uE8A7", "download": "\uE896", "people": "\uE716", "play": "\uE768",
    "stop": "\uE71A",
}
SPEAKER_TAGS = len(PALETTES["dark"]["speakers"])  # same length in every palette
ICON_FONTS = [r"C:\Windows\Fonts\SegoeIcons.ttf", r"C:\Windows\Fonts\segmdl2.ttf"]
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
    return text if len(text) <= limit else f"{text[:18]}\u2026{text[-(limit - 19):]}"


def short_name(name: str, limit: int = 44) -> str:
    return name if len(name) <= limit else f"{name[:limit - 14]}\u2026{name[-13:]}"


def speaker_hint(choice: str) -> str:
    if choice == AUTO_SPEAKERS:
        return "Finds how many voices there are"
    if choice == NO_LABELS:
        return "Plain transcript, a bit faster"
    return f"Exactly {choice} voices"


def worker_command(args: list[str]) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--worker", *args]
    return [deps.console_python(), str(env.APP_DIR / "app.py"), "--worker", *args]


def system_theme() -> str:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as key:
            return "light" if winreg.QueryValueEx(key, "AppsUseLightTheme")[0] else "dark"
    except (ImportError, OSError):
        return "light"


def resolve_theme(preference: str) -> str:
    return preference if preference in ("light", "dark") else system_theme()


def set_title_bar(window: tk.Misc, dark: bool, repaint: bool = False) -> None:
    """Dark or light window title bar to match the theme (Windows 10 20H1+ / 11)."""
    if sys.platform != "win32":
        return
    try:
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        value = ctypes.c_int(1 if dark else 0)
        for attribute in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE, and its pre-20H1 number
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(value),
                                                         ctypes.sizeof(value)) == 0:
                break
        if repaint:  # Windows only redraws the caption on the next frame change
            window.wm_attributes("-alpha", 0.99)
            window.after(30, lambda: window.wm_attributes("-alpha", 1.0))
    except (AttributeError, OSError, tk.TclError):
        pass


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


class Look:
    """Fonts, palette, icons and theme switching shared by the main window and its dialogs."""

    def __init__(self, root: tk.Tk, theme: str):
        self.root = root
        self.scale = root.winfo_fpixels("1i") / 96
        self.theme = theme
        self.p = PALETTES[theme]
        families = set(tkfont.families(root))

        def pick(*names):
            return next((n for n in names if n in families), names[-1])

        self.families = {
            "text": pick("Segoe UI Variable Text", "Segoe UI"),
            "strong": pick("Segoe UI Variable Text Semibold", "Segoe UI Semibold"),
            "display": pick("Segoe UI Variable Display Semib", "Segoe UI Semibold"),
            "small": pick("Segoe UI Variable Small", "Segoe UI"),
            "small_strong": pick("Segoe UI Variable Small Semibol", "Segoe UI Semibold"),
        }
        f = self.families

        def font(family, size):
            return tkfont.Font(root=root, family=family, size=size)

        self.fonts = {
            "title": font(f["display"], 18), "section": font(f["strong"], 11), "body": font(f["text"], 10),
            "strong": font(f["strong"], 10), "caption": font(f["small_strong"], 8), "hint": font(f["small"], 9),
            "reading": font(f["text"], 11), "reading_strong": font(f["strong"], 11),
            "doc_title": font(f["display"], 14), "stamp": font(f["small"], 9), "drop": font(f["display"], 14),
            "estimate": font(f["display"], 16), "placeholder": font(f["display"], 13), "mono": font("Consolas", 9),
        }
        self.icon_font_path = next((path for path in ICON_FONTS if os.path.exists(path)), None)
        self._icon_fonts: dict[int, object] = {}
        self._images: dict[tuple, object] = {}

    def px(self, value: float) -> int:
        return int(round(value * self.scale))

    def apply(self, theme: str) -> None:
        self.theme, self.p = theme, PALETTES[theme]
        p, style = self.p, ttk.Style(self.root)
        if sv_ttk is not None:
            sv_ttk.set_theme(theme, self.root)
            self._scale_theme_fonts()
        else:
            style.theme_use("vista" if "vista" in style.theme_names() else "clam")
            for name in ("TFrame", "TLabel", "TCheckbutton", "TRadiobutton"):
                style.configure(name, background=p["bg"], foreground=p["text"])
            style.configure("Card.TFrame", background=p["bg"], relief="solid", borderwidth=1)
        style.configure("Title.TLabel", font=self.fonts["title"], foreground=p["text"])
        style.configure("Section.TLabel", font=self.fonts["section"], foreground=p["text"])
        style.configure("Strong.TLabel", font=self.fonts["strong"], foreground=p["text"])
        style.configure("Muted.TLabel", font=self.fonts["body"], foreground=p["muted"])
        style.configure("Caption.TLabel", font=self.fonts["caption"], foreground=p["muted"])
        style.configure("Hint.TLabel", font=self.fonts["hint"], foreground=p["muted"])
        style.configure("Estimate.TLabel", font=self.fonts["estimate"], foreground=p["text"])
        for kind, (bg, _fg) in p["banners"].items():
            style.configure(f"{kind.title()}.TButton", background=bg)  # rounded corners blend into the banner
        self.root.configure(bg=p["bg"])

    def _scale_theme_fonts(self) -> None:
        """sv-ttk sizes fonts in pixels, which stay tiny on scaled displays; use points instead."""
        f = self.families
        for name, family, size in (("SunValleyCaptionFont", f["small"], 9), ("SunValleyBodyFont", f["text"], 10),
                                   ("SunValleyBodyStrongFont", f["strong"], 10),
                                   ("SunValleyBodyLargeFont", f["text"], 13),
                                   ("SunValleySubtitleFont", f["display"], 15),
                                   ("SunValleyTitleFont", f["display"], 21),
                                   ("SunValleyTitleLargeFont", f["display"], 30),
                                   ("SunValleyDisplayFont", f["display"], 50)):
            try:
                self.root.tk.call("font", "configure", name, "-family", family, "-size", size)
            except tk.TclError:
                pass

    def icon(self, name: str, color: str | None = None, size: int = 16):
        """A Fluent icon as a PhotoImage, or "" when the icon font or Pillow is unavailable."""
        if Image is None or self.icon_font_path is None or name not in ICONS:
            return ""
        color = color or self.p["text"]
        key = (name, color, size)
        if key not in self._images:
            px = max(8, self.px(size))
            if px not in self._icon_fonts:
                self._icon_fonts[px] = ImageFont.truetype(self.icon_font_path, px)
            img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
            ImageDraw.Draw(img).text((px / 2, px / 2), ICONS[name], font=self._icon_fonts[px], fill=color, anchor="mm")
            self._images[key] = ImageTk.PhotoImage(img, master=self.root)
        return self._images[key]

    def button_image(self, name: str, accent: bool = False):
        """Icon for a button, greyed out when the button is disabled."""
        normal = self.icon(name, self.p["on_accent"] if accent else self.p["text"])
        if not normal:
            return ""
        return (normal, "disabled", self.icon(name, self.p["subtle"]))

    def logo(self, size: int):
        key = ("logo", size)
        if key not in self._images:
            path = env.APP_DIR / "assets" / "app.png"
            if Image is not None and path.exists():
                img = Image.open(path).convert("RGBA").resize((self.px(size), self.px(size)), Image.LANCZOS)
                self._images[key] = ImageTk.PhotoImage(img, master=self.root)
            else:
                self._images[key] = ""
        return self._images[key]

    def style_text(self, box: tk.Text, mono: bool = False) -> None:
        p = self.p
        box.configure(bg=p["bg"], fg=p["text"], insertbackground=p["text"], selectbackground=p["select"],
                      selectforeground=p["text"], inactiveselectbackground=p["select"],
                      font=self.fonts["mono" if mono else "reading"])


def _round_rect(canvas: tk.Canvas, x1, y1, x2, y2, r, **kwargs):
    points = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
              x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return canvas.create_polygon(points, smooth=True, **kwargs)


class App:
    def __init__(self, root: tk.Tk, dnd: bool):
        self.root, self.dnd = root, dnd
        self.cfg = config.load()
        self.look = Look(root, resolve_theme(self.cfg.get("theme", "system")))
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
        self.banner_state: tuple | None = None  # kept so a theme switch can redraw the banner
        self.install_window = None
        self.status = ("Starting\u2026", "busy")
        self.drop_state = "idle"
        self._drop_after = None
        self._icon_buttons: list[tuple[ttk.Button, str, bool]] = []
        self._build()
        self.apply_theme(self.look.theme)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.report_callback_exception = self._on_tk_error
        root.bind("<Control-o>", lambda e: self.browse())
        root.after(150, self._poll)
        root.after(300, self.startup_checks)

    @property
    def p(self) -> dict:
        return self.look.p

    @property
    def banner_kind(self) -> str | None:
        return self.banner_state[0] if self.banner_state else None

    # ------------------------------------------------------------ layout

    def _build(self):
        root, P = self.root, self.look.px
        root.title(APP_NAME)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(P(980), int(sw * 0.92)), min(P(860), sh - P(90))
        root.geometry(f"{w}x{h}+{max((sw - w) // 2, 0)}+{max((sh - h) // 2 - P(20), 0)}")
        root.minsize(P(760), P(640))

        outer = ttk.Frame(root, padding=(P(28), P(22), P(28), P(12)))
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(6, weight=1)

        # -- header
        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        self.logo_label = ttk.Label(header)
        self.logo_label.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, P(14)))
        ttk.Label(header, text=APP_NAME, style="Title.TLabel").grid(row=0, column=1, sticky="sw")
        ttk.Label(header, text="Private transcription with speaker labels. Everything stays on this PC.",
                  style="Muted.TLabel").grid(row=1, column=1, sticky="nw")
        self.settings_btn = ttk.Button(header, text="Settings", compound="left", command=self.open_settings)
        self.settings_btn.grid(row=0, column=2, rowspan=2, sticky="e")
        self._icon_buttons.append((self.settings_btn, "settings", False))

        # -- options card
        opts = ttk.Frame(outer, style="Card.TFrame", padding=(P(18), P(12), P(18), P(14)))
        opts.grid(row=1, column=0, sticky="ew", pady=(P(18), 0))
        opts.columnconfigure(3, weight=1)
        self.model_var = tk.StringVar(value=self.cfg["model"] if self.cfg["model"] in MODELS else DEFAULT_MODEL)
        self.speakers_var = tk.StringVar(
            value=self.cfg["speakers"] if self.cfg["speakers"] in SPEAKER_CHOICES else AUTO_SPEAKERS)
        quality = self.cfg.get("quality") if self.cfg.get("quality") in QUALITY else DEFAULT_QUALITY
        self.quality_var = tk.StringVar(value=QUALITY_LABELS[quality])

        def field(column, caption, variable, values, width):
            ttk.Label(opts, text=caption, style="Caption.TLabel").grid(row=0, column=column, sticky="w", padx=(0, P(22)))
            box = ttk.Combobox(opts, textvariable=variable, values=values, state="readonly", width=width)
            box.grid(row=1, column=column, sticky="w", padx=(0, P(22)), pady=(P(4), P(4)))
            hint = ttk.Label(opts, style="Hint.TLabel")
            hint.grid(row=2, column=column, sticky="w", padx=(0, P(22)))
            return box, hint

        model_box, self.model_hint = field(0, "MODEL", self.model_var, MODELS, 13)
        speakers_box, self.speakers_hint = field(1, "SPEAKERS", self.speakers_var, SPEAKER_CHOICES, 18)
        quality_box, self.quality_hint = field(2, "SPEED", self.quality_var, list(QUALITY_LABELS.values()), 14)
        model_box.bind("<<ComboboxSelected>>", lambda e: self._choices_changed(rewarm=True))
        speakers_box.bind("<<ComboboxSelected>>", lambda e: self._choices_changed(rewarm=False))
        quality_box.bind("<<ComboboxSelected>>", lambda e: self._choices_changed(rewarm=True))
        estimate = ttk.Frame(opts)
        estimate.grid(row=0, column=3, rowspan=3, sticky="e")
        ttk.Label(estimate, text="ESTIMATED TIME", style="Caption.TLabel").pack(anchor="e")
        self.estimate_value = ttk.Label(estimate, style="Estimate.TLabel")
        self.estimate_value.pack(anchor="e")
        ttk.Label(estimate, text="per hour of audio on this PC", style="Hint.TLabel").pack(anchor="e")

        # -- drop zone
        self.drop = tk.Canvas(outer, height=P(150), highlightthickness=0, bd=0, cursor="hand2")
        self.drop.grid(row=2, column=0, sticky="ew", pady=(P(16), 0))
        self.drop.bind("<Configure>", lambda e: self._schedule_drop_redraw())
        self.drop.bind("<Button-1>", lambda e: self.browse())
        self.drop.bind("<Enter>", lambda e: self._set_drop_state("hover"))
        self.drop.bind("<Leave>", lambda e: self._set_drop_state("idle"))

        self._build_banner(outer)
        self._build_progress(outer)

        # -- transcript header with actions
        heading = ttk.Frame(outer)
        heading.grid(row=5, column=0, sticky="ew", pady=(P(20), P(8)))
        heading.columnconfigure(1, weight=1)
        ttk.Label(heading, text="Transcript", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.file_label = ttk.Label(heading, style="Muted.TLabel")
        self.file_label.grid(row=0, column=1, sticky="w", padx=P(10))
        actions = ttk.Frame(heading)
        actions.grid(row=0, column=2, sticky="e")
        self.names_btn = ttk.Button(actions, text="Name speakers", compound="left", command=self.open_speaker_names)
        self.open_btn = ttk.Button(actions, text="Open", compound="left", command=lambda: self.open_output("txt"))
        self.srt_btn = ttk.Button(actions, text="Subtitles", compound="left", command=lambda: self.open_output("srt"))
        self.folder_btn = ttk.Button(actions, text="Show in folder", compound="left", command=self.show_in_folder)
        self.copy_btn = ttk.Button(actions, text="Copy", style="Accent.TButton", compound="left",
                                   command=self.copy_transcript)
        for button, icon, accent in ((self.names_btn, "people", False), (self.open_btn, "open", False),
                                     (self.srt_btn, "captions", False), (self.folder_btn, "folder", False),
                                     (self.copy_btn, "copy", True)):
            button.pack(side="left", padx=(P(6), 0))
            button.state(["disabled"])
            self._icon_buttons.append((button, icon, accent))

        # -- transcript card
        card = ttk.Frame(outer, style="Card.TFrame", padding=P(2))
        card.grid(row=6, column=0, sticky="nsew")
        card.columnconfigure(0, weight=1)
        card.rowconfigure(0, weight=1)
        self.text = tk.Text(card, wrap="word", relief="flat", bd=0, highlightthickness=0, padx=P(22), pady=P(18),
                            spacing2=P(3), cursor="arrow", insertwidth=0, takefocus=False)
        scroll = ttk.Scrollbar(card, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.grid(row=0, column=0, sticky="nsew", padx=(P(2), 0), pady=P(2))
        scroll.grid(row=0, column=1, sticky="ns", padx=(0, P(2)), pady=P(6))

        # -- status bar
        bar = ttk.Frame(outer)
        bar.grid(row=7, column=0, sticky="ew", pady=(P(10), 0))
        bar.columnconfigure(1, weight=1)
        self.status_dot = tk.Canvas(bar, width=P(9), height=P(9), highlightthickness=0, bd=0)
        self.status_dot.grid(row=0, column=0, padx=(P(2), P(8)))
        self.status_label = ttk.Label(bar, style="Hint.TLabel")
        self.status_label.grid(row=0, column=1, sticky="w")
        self.queue_label = ttk.Label(bar, style="Hint.TLabel")
        self.queue_label.grid(row=0, column=2, sticky="e")

        if self.dnd:
            for widget in (self.drop, self.text):
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_drop)
                widget.dnd_bind("<<DropEnter>>", lambda e: self._drop_hover(e, True))
                widget.dnd_bind("<<DropLeave>>", lambda e: self._drop_hover(e, False))

    def _build_banner(self, parent):
        P = self.look.px
        self.banner = tk.Frame(parent, highlightthickness=1, bd=0, padx=P(14), pady=P(10))
        self.banner.columnconfigure(1, weight=1)
        self.banner_icon = tk.Label(self.banner, bd=0)
        self.banner_icon.grid(row=0, column=0, rowspan=3, sticky="n", padx=(0, P(12)), pady=(P(2), 0))
        self.banner_title = tk.Label(self.banner, font=self.look.fonts["strong"], anchor="w", justify="left", bd=0)
        self.banner_title.grid(row=0, column=1, sticky="ew")
        self.banner_hint = tk.Label(self.banner, font=self.look.fonts["body"], anchor="w", justify="left", bd=0)
        self.banner_hint.grid(row=1, column=1, sticky="ew", pady=(P(2), 0))
        self.banner_actions = tk.Frame(self.banner, bd=0)
        self.banner_actions.grid(row=2, column=1, sticky="w", pady=(P(8), 0))
        self.banner_close = tk.Label(self.banner, cursor="hand2", bd=0)
        self.banner_close.grid(row=0, column=2, sticky="ne", padx=(P(10), 0))
        self.banner_close.bind("<Button-1>", lambda e: self.hide_banner())
        self.banner.bind("<Configure>", lambda e: self._wrap_banner(e.width))

    def _build_progress(self, parent):
        P = self.look.px
        self.progress = ttk.Frame(parent, style="Card.TFrame", padding=(P(18), P(14), P(18), P(14)))
        self.progress.columnconfigure(0, weight=1)
        self.stage_label = ttk.Label(self.progress, style="Strong.TLabel", compound="left")
        self.stage_label.grid(row=0, column=0, sticky="w")
        self.pct_label = ttk.Label(self.progress, style="Strong.TLabel")
        self.pct_label.grid(row=0, column=1, sticky="e")
        self.bar = ttk.Progressbar(self.progress, maximum=100)
        self.bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(P(10), P(8)))
        self.time_label = ttk.Label(self.progress, style="Muted.TLabel")
        self.time_label.grid(row=2, column=0, sticky="w")
        self.cancel_btn = ttk.Button(self.progress, text="Cancel", compound="left", command=self.cancel)
        self.cancel_btn.grid(row=2, column=1, sticky="e")
        self._icon_buttons.append((self.cancel_btn, "cancel", False))
        self.progress.grid(row=4, column=0, sticky="ew", pady=(P(14), 0))
        self.progress.grid_remove()

    # ------------------------------------------------------------ theme

    def apply_theme(self, theme: str):
        look = self.look
        look.apply(theme)
        p = self.p
        self.logo_label.configure(image=look.logo(40))
        for button, icon, accent in self._icon_buttons:
            button.configure(image=look.button_image(icon, accent))
        look.style_text(self.text)
        self._style_transcript_tags()
        self.drop.configure(bg=p["bg"])
        self.status_dot.configure(bg=p["bg"])
        self._draw_drop()
        self._set_status(*self.status)
        self._update_hints()
        if self.banner_state:
            self.show_banner(*self.banner_state)
        if self.outputs:
            self._render_transcript(self.transcript)
        set_title_bar(self.root, theme == "dark", repaint=self.root.winfo_viewable())

    def _style_transcript_tags(self):
        fonts, p, P = self.look.fonts, self.p, self.look.px
        t = self.text
        t.tag_configure("doc_title", font=fonts["doc_title"], foreground=p["text"], spacing3=P(2))
        t.tag_configure("meta", font=fonts["hint"], foreground=p["muted"], spacing3=P(14))
        t.tag_configure("heading", spacing1=P(4))
        t.tag_configure("stamp", font=fonts["stamp"], foreground=p["subtle"])
        t.tag_configure("para", font=fonts["reading"], foreground=p["text"], spacing3=P(14))
        t.tag_configure("placeholder_title", font=fonts["placeholder"], foreground=p["muted"], justify="center",
                        spacing1=P(36), spacing3=P(6))
        t.tag_configure("placeholder", font=fonts["body"], foreground=p["subtle"], justify="center")
        for i, color in enumerate(p["speakers"]):
            t.tag_configure(f"spk{i}", font=fonts["reading_strong"], foreground=color)

    # ------------------------------------------------------------ drop zone

    def _schedule_drop_redraw(self):
        if self._drop_after is not None:
            self.root.after_cancel(self._drop_after)
        self._drop_after = self.root.after(30, self._draw_drop)

    def _set_drop_state(self, state: str):
        if state != self.drop_state:
            self.drop_state = state
            self._draw_drop()

    def _draw_drop(self):
        self._drop_after = None
        c, p, look = self.drop, self.p, self.look
        P = look.px
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 20 or h < 20:
            return
        state = self.drop_state
        fill = p["surface_active"] if state == "drag" else p["surface"]
        outline = p["accent"] if state in ("drag", "hover") else p["border"]
        line = 2 if state == "drag" else 1
        title = "Release to add" if state == "drag" else "Drop audio or video files"
        how = "or click to browse" if self.dnd else "Click to choose files"
        subtitle = f"{how}  \u00b7  MP3, WAV, M4A, MP4 and more  \u00b7  Ctrl+O"
        title_font, sub_font = look.fonts["drop"], look.fonts["body"]

        # Roomy and centred while the window is empty; a slim bar (glyph beside the text)
        # once a transcript or a running job needs the space.
        bars = (0.36, 0.66, 1.0, 0.56, 0.82, 0.42)
        bar_w, gap = P(5), P(5)
        compact = h < P(120)
        tall = P(26 if compact else 30)
        glyph_w = len(bars) * bar_w + (len(bars) - 1) * gap
        if compact:
            text_w = max(title_font.measure(title), sub_font.measure(subtitle))
            left = (w - (glyph_w + P(22) + text_w)) / 2
            glyph_x, glyph_y = left, h / 2
            text_x, anchor = left + glyph_w + P(22), "w"
            title_y, sub_y = h / 2 - P(12), h / 2 + P(13)
        else:
            glyph_x, glyph_y = (w - glyph_w) / 2, h / 2 - P(28)
            text_x, anchor = w / 2, "center"
            title_y, sub_y = h / 2 + P(12), h / 2 + P(38)

        if Image is not None:
            ss = 2  # supersample for smooth corners
            img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((ss, ss, w * ss - ss - 1, h * ss - ss - 1), radius=P(14) * ss, fill=fill,
                                outline=outline, width=max(1, P(line)) * ss)
            x = glyph_x * ss
            for frac in bars:
                half = tall * ss * frac / 2
                d.rounded_rectangle((x, glyph_y * ss - half, x + bar_w * ss, glyph_y * ss + half),
                                    radius=bar_w * ss / 2, fill=p["accent"])
                x += (bar_w + gap) * ss
            self._drop_image = ImageTk.PhotoImage(img.resize((w, h), Image.LANCZOS), master=self.root)
            c.create_image(0, 0, image=self._drop_image, anchor="nw")
        else:
            _round_rect(c, 2, 2, w - 2, h - 2, P(14), fill=fill, outline=outline, width=line)
        c.create_text(text_x, title_y, text=title, font=title_font, anchor=anchor,
                      fill=p["accent"] if state == "drag" else p["text"])
        c.create_text(text_x, sub_y, text=subtitle, font=sub_font, anchor=anchor, fill=p["muted"])

    def _update_drop_size(self):
        compact = self.current is not None or bool(self.outputs)
        height = self.look.px(92 if compact else 150)
        if int(self.drop.cget("height")) != height:
            self.drop.configure(height=height)  # <Configure> redraws it

    def _drop_hover(self, event, active: bool):
        self._set_drop_state("drag" if active else "idle")
        return event.action

    def _on_drop(self, event):
        self._set_drop_state("idle")
        self.add_files(self.root.tk.splitlist(event.data))
        return COPY

    # ------------------------------------------------------------ banner

    def show_banner(self, kind: str, title: str, hint: str = "", actions=()):
        bg, fg = self.p["banners"][kind]
        self.banner_state = (kind, title, hint, tuple(actions))
        for widget in (self.banner, self.banner_icon, self.banner_title, self.banner_hint,
                       self.banner_actions, self.banner_close):
            widget.configure(bg=bg)
        self.banner.configure(highlightbackground=fg, highlightcolor=fg)
        self.banner_icon.configure(image=self.look.icon(kind, fg, 18))
        close = self.look.icon("cancel", fg, 12)
        self.banner_close.configure(image=close, text="" if close else "\u2715", fg=fg)
        self.banner_title.configure(text=title, fg=fg)
        self.banner_hint.configure(text=hint, fg=self.p["text"])
        self.banner_hint.grid() if hint else self.banner_hint.grid_remove()
        for child in self.banner_actions.winfo_children():
            child.destroy()
        for label, callback in actions:
            ttk.Button(self.banner_actions, text=label, style=f"{kind.title()}.TButton",
                       command=callback).pack(side="left", padx=(0, self.look.px(8)))
        self.banner_actions.grid() if actions else self.banner_actions.grid_remove()
        self.banner.grid(row=3, column=0, sticky="ew", pady=(self.look.px(14), 0))

    def hide_banner(self):
        self.banner_state = None
        self.banner.grid_remove()

    def _wrap_banner(self, width):
        wrap = max(width - self.look.px(110), 200)
        self.banner_title.configure(wraplength=wrap)
        self.banner_hint.configure(wraplength=wrap)

    def _text_window(self, title: str, text: str = "", copy_text: str | None = None) -> tuple[tk.Toplevel, tk.Text]:
        look, P = self.look, self.look.px
        win = tk.Toplevel(self.root)
        win.withdraw()
        win.title(title)
        win.configure(bg=self.p["bg"])
        win.geometry(f"{P(780)}x{P(480)}")
        frame = ttk.Frame(win, padding=P(14))
        frame.pack(fill="both", expand=True)
        footer = ttk.Frame(frame)
        footer.pack(side="bottom", fill="x", pady=(P(10), 0))
        if copy_text is not None:
            ttk.Button(footer, text="Copy", style="Accent.TButton", image=look.button_image("copy", True),
                       compound="left", command=lambda: self._to_clipboard(copy_text)).pack(side="right")
        card = ttk.Frame(frame, style="Card.TFrame", padding=P(2))
        card.pack(fill="both", expand=True)
        box = tk.Text(card, wrap="none", relief="flat", bd=0, highlightthickness=0, padx=P(10), pady=P(8))
        look.style_text(box, mono=True)
        scroll = ttk.Scrollbar(card, command=box.yview)
        box.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        box.pack(fill="both", expand=True)
        box.insert("1.0", text)
        box.see("end")
        set_title_bar(win, look.theme == "dark")
        win.deiconify()
        return win, box

    def show_details(self, title: str, text: str):
        _win, box = self._text_window(title, text, copy_text=text)
        box.configure(state="disabled")

    def _on_tk_error(self, exc_type, exc, tb):
        detail = "".join(traceback.format_exception(exc_type, exc, tb))
        self.show_banner("error", f"Unexpected problem in the app window: {exc}",
                         "The app is still running. Please try again.",
                         [("Details\u2026", lambda: self.show_details("Error details", detail))])

    # ------------------------------------------------------------ setup checks

    def startup_checks(self) -> bool:
        missing = deps.missing_packages()
        if missing:
            self.setup_ok = False
            self._set_status("Setup needed", "error")
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
            self._set_status("Setup needed", "error")
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
        command = deps.install_command()
        self.install_window, self.install_log = self._text_window("Installing dependencies",
                                                                  "> " + " ".join(command) + "\n")

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
        self.show_banner("info", "Installing dependencies\u2026", "This can take several minutes.")

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
            self.stage_label.configure(text="Starting\u2026", image="")
            self.pct_label.configure(text="")
            self.time_label.configure(text="")
            self.cancel_btn.grid()
            self.progress.grid()
            self._update_drop_size()
        if not worker.send(command):
            self.worker = None
            if meta is not None:
                self.current = None
                self.progress.grid_remove()
            self.show_banner("error", "The transcription process isn't responding.", "Please try again.")
            return None
        self._update_queue_label()
        return self.command_id

    def _quality(self) -> str:
        label = self.quality_var.get()
        return next((key for key, value in QUALITY_LABELS.items() if value == label), DEFAULT_QUALITY)

    def _common_fields(self, diarize: bool) -> dict:
        return {"model": self.model_var.get(), "diarize": diarize, "quality": self._quality(),
                "language": self.cfg["language"], "subtitles": bool(self.cfg.get("subtitles", True))}

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
            self._set_status("Starting the speech engine\u2026", "busy")

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

    def _ready_status(self):
        model = self.model_var.get()
        diarize = self.speakers_var.get() != NO_LABELS
        offline = not missing_components(config.models_dir(self.cfg), model, diarize and bool(
            self.cfg["hf_token"] or self._diarization_ready()))
        where = "works offline" if offline else "the first file downloads the models"
        self._set_status(f"Ready  \u00b7  {model} loaded  \u00b7  {where}", "ok")

    # ------------------------------------------------------------ queue + jobs

    def browse(self):
        patterns = " ".join(f"*{ext}" for ext in SUPPORTED_EXTS)
        files = filedialog.askopenfilenames(title="Choose recordings to transcribe (or a transcript to open)",
                                            filetypes=[("Audio and video", patterns),
                                                       ("Transcripts", "*.transcript.txt"), ("All files", "*.*")])
        if files:
            self.add_files(files)

    def add_files(self, names):
        good, bad, transcripts = [], [], []
        for name in names:
            path = Path(name)
            if path.is_file() and path.name.lower().endswith(".transcript.txt"):
                transcripts.append(path)
            else:
                (good if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS else bad).append(path)
        if transcripts:
            self.open_transcript(transcripts[-1])
        if bad:
            more = f" (and {len(bad) - 1} more)" if len(bad) > 1 else ""
            self.show_banner("error", f"Can't transcribe \u201c{bad[0].name}\u201d{more}: not a supported audio or video file.",
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
                [("Add token\u2026", self.open_settings),
                 ("Transcribe without labels", self._continue_without_labels),
                 ("Clear queue", self._clear_queue)])
            return
        self._start_transcription(self.pending.popleft(), diarize)

    def _continue_without_labels(self):
        self.speakers_var.set(NO_LABELS)
        self._choices_changed(rewarm=False)
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
        self.file_label.configure(text=f"\u00b7  {short_name(source.name)}")
        self._set_result(None, "")
        self._set_placeholder(f"Working on {source.name}\u2026", "The transcript will appear here when it's done.")
        if self._dispatch(command, {"kind": "transcribe", "source": source}) is not None:
            self._set_status(f"Transcribing {source.name}", "work")

    def start_prepare(self):
        if self.current is not None:
            messagebox.showinfo(APP_NAME, "Please wait until the current transcription finishes.", parent=self.root)
            return
        if not self.setup_ok and not self.startup_checks():
            return
        diarize = bool(self.cfg["hf_token"]) or self._diarization_ready()
        if self._dispatch({"cmd": "prepare", **self._common_fields(diarize)}, {"kind": "prepare", "source": None}):
            self._set_status("Downloading models", "work")

    def cancel(self):
        if self.current is None or self.worker is None:
            return
        name = self.current["source"].name if self.current.get("source") else "the model download"
        extra = f"\n\nThe {len(self.pending)} queued file(s) will be removed too." if self.pending else ""
        if not messagebox.askyesno(APP_NAME, f"Stop working on {name}?{extra}", parent=self.root):
            return
        self.current["cancelled"] = True
        self.pending.clear()
        self.stage_label.configure(text="Cancelling\u2026")
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
                self.pct_label.configure(text=f"{detail}  \u00b7  {value:.0f}%" if detail else f"{value:.0f}%")
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
        if kind == "stage" and self.current is None:
            self._set_status(event.get("label", ""), "busy")
        elif kind in ("warm_done", "error"):
            self.warm_id = None
            if self.current is None:
                if kind == "warm_done":
                    self._ready_status()
                else:
                    self._set_status("Couldn't preload the models. They load with the first file instead.", "off")

    def _tick(self):
        if self.current is None:
            return
        parts = [f"Elapsed {clock(time.monotonic() - self.current['started'], False)}"]
        if self.eta is not None:
            eta, received = self.eta
            left = eta - (time.monotonic() - received)
            parts.append(human_eta(left) if left > 0 else "finishing up\u2026")
        if self.current.get("duration"):
            parts.append(f"{human_duration(self.current['duration'])} of audio")
        self.time_label.configure(text="   \u00b7   ".join(parts))

    def _on_worker_exit(self, code):
        self.worker = None
        self.warm_id = None
        self._set_status("Speech engine stopped. It restarts with the next file.", "off")
        if self.current is not None:
            self._finish(crashed=code)

    def _finish(self, done: dict | None = None, error: dict | None = None, crashed=None):
        job, self.current, self.eta = self.current, None, None
        self._indeterminate(False)
        self.cancel_btn.grid_remove()
        log = "\n".join(self.worker.log) if self.worker is not None else ""
        if self.worker is not None and self.worker.alive:
            self._ready_status()
        if job["cancelled"]:
            self.progress.grid_remove()
            self._set_placeholder("Cancelled", "Drop a file to start again.")
            self._update_drop_size()
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
                self._set_placeholder("No transcript", "See the message above.")
            self._update_drop_size()
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
        self.stage_label.configure(text=f"  Transcribed {job['source'].name}",
                                   image=self.look.icon("success", self.p["status"]["ok"]))
        self.time_label.configure(text=f"{human_duration(done['duration_s'])} of audio in "
                                       f"{human_duration(done['elapsed_s'])}   \u00b7   {speakers}   \u00b7   "
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
            actions.append(("Details\u2026", lambda: self.show_details("Error details", details)))
        self.show_banner("error", err.get("message", "Something went wrong."), err.get("hint", ""), actions)

    # ------------------------------------------------------------ transcript display

    def _set_placeholder(self, title: str, message: str = ""):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", title + "\n", "placeholder_title")
        if message:
            self.text.insert("end", message, "placeholder")
        self.text.configure(state="disabled")

    def _set_result(self, done: dict | None, text: str):
        self.outputs, self.transcript = (done or {}), text
        for button in (self.copy_btn, self.open_btn, self.srt_btn, self.folder_btn):
            button.state(["!disabled"] if done else ["disabled"])
        self.names_btn.state(["!disabled"] if done and speakers_in(text) else ["disabled"])
        # No subtitle files for this transcript: hide the button rather than leave it dead.
        if done and "srt" not in done:
            self.srt_btn.pack_forget()
        elif not self.srt_btn.winfo_manager():
            self.srt_btn.pack(side="left", padx=(self.look.px(6), 0), before=self.folder_btn)
        if done:
            self._render_transcript(text)
        self._update_drop_size()

    def open_transcript(self, path: Path):
        """Show a transcript made earlier (dropped or browsed to), so its speakers can be named."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            self.show_banner("error", f"Couldn't open {path.name}.", str(exc))
            return
        found = {kind: str(value) for kind, value in transcript_siblings(path).items()}
        speakers = speakers_in(text)
        if self.current is None:
            self.progress.grid_remove()
        self.file_label.configure(text=f"·  {short_name(path.name)}")
        self._set_result(dict(found, speakers=len(speakers), diarized=bool(speakers)), text)

    def open_speaker_names(self):
        speakers = speakers_in(self.transcript)
        if not self.outputs.get("txt") or not speakers:
            self.show_banner("info", "This transcript has no speaker labels to name.",
                             "Transcribe with Speakers set to Auto-detect (or a number) to get labelled turns.")
            return
        SpeakerNamesDialog(self, speakers, self.outputs)

    def reload_transcript(self, changed: list[Path]):
        try:
            text = Path(self.outputs["txt"]).read_text(encoding="utf-8")
        except OSError as exc:
            self.show_banner("error", "The names were saved but the transcript couldn't be re-read.", str(exc))
            return
        self._set_result(self.outputs, text)
        kinds = [p.suffix for p in changed]
        listed = ", ".join(kinds[:-1]) + (" and " if len(kinds) > 1 else "") + kinds[-1] if kinds else ""
        self.show_banner("success", f"Speaker names saved in the {listed} file{'s' if len(kinds) != 1 else ''}."
                         if kinds else "No names changed.")

    def _render_transcript(self, text: str):
        box = self.text
        box.configure(state="normal")
        box.delete("1.0", "end")
        colors: dict[str, str] = {}
        lines = text.splitlines()
        body_start = lines.index("") + 1 if "" in lines[:4] else 0
        for i, line in enumerate(lines[:body_start]):
            if line:
                box.insert("end", line + "\n", "doc_title" if i == 0 else "meta")
        for line in lines[body_start:]:
            if not line:
                continue
            match = HEADING_RE.match(line)
            if match:
                name = match.group("speaker")
                if name:
                    tag = colors.setdefault(name, f"spk{len(colors) % SPEAKER_TAGS}")
                    box.insert("end", name, (tag, "heading"))
                    box.insert("end", "   ", "heading")
                box.insert("end", match.group("stamp").strip("[]") + "\n", ("stamp", "heading"))
            else:
                box.insert("end", line + "\n", "para")
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
        self.copy_btn.configure(text="Copied", image=self.look.button_image("check", True))
        self.root.after(1500, lambda: self.copy_btn.configure(text="Copy", image=self.look.button_image("copy", True)))

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

    def _set_status(self, text: str, kind: str = "ok"):
        self.status = (text, kind)
        self.status_label.configure(text=text)
        size = self.look.px(9)
        self.status_dot.delete("all")
        color = self.p["status"].get(kind, self.p["status"]["off"])
        self.status_dot.create_oval(1, 1, size - 1, size - 1, fill=color, outline=color)

    def _update_queue_label(self):
        count = len(self.pending)
        if count:
            text = (f"{count} more file{'s' if count != 1 else ''} queued" if self.current is not None
                    else f"{count} file{'s' if count != 1 else ''} waiting")
        else:
            text = ""
        self.queue_label.configure(text=text)

    def _update_hints(self):
        self.model_hint.configure(text=MODEL_HINTS.get(self.model_var.get(), ""))
        self.speakers_hint.configure(text=speaker_hint(self.speakers_var.get()))
        self.quality_hint.configure(text=QUALITY_HINTS[self._quality()])
        self.estimate_value.configure(text=self._estimate_text())

    def _estimate_text(self) -> str:
        """Processing time for one hour of audio with the current choices, from this PC's measured speeds."""
        try:
            job = Job(source=Path("estimate"), model=self.model_var.get(), quality=self._quality(),
                      diarize=self.speakers_var.get() != NO_LABELS, subtitles=bool(self.cfg.get("subtitles", True)))
            rates = {s.key: s.estimate for s in plan_stages(job, 3600, load_calibration()) if s.per_audio_second}
        except Exception:
            return "\u2014"
        main = rates.get("transcribe", 0) + rates.get("align", 0)
        total = max(main, rates.get("diarize", 0))  # speaker detection runs alongside transcription
        minutes = max(1, round(total / 60))
        return f"\u2248 {minutes} min" if minutes < 60 else f"\u2248 {minutes // 60} h {minutes % 60:02d} min"

    def _choices_changed(self, rewarm: bool):
        self.cfg["model"] = self.model_var.get()
        self.cfg["speakers"] = self.speakers_var.get()
        self.cfg["quality"] = self._quality()
        self._update_hints()
        self._save_config()
        if rewarm or self.speakers_var.get() != NO_LABELS:
            self._schedule_warm_up()

    def _save_config(self) -> bool:
        try:
            config.save(self.cfg)
            return True
        except OSError as exc:
            self.show_banner("error", "Couldn't save settings.", str(exc))
            return False

    def open_settings(self, welcome: bool = False):
        SettingsDialog(self, welcome)

    def on_settings_saved(self, threads_changed: bool = False, theme_changed: bool = False):
        if theme_changed:
            self.apply_theme(resolve_theme(self.cfg.get("theme", "system")))
        if self.banner_kind == "warning" and self.pending:
            self.hide_banner()
        self._update_hints()
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
        self.withdraw()  # build hidden, then show centred: no flicker
        self.app, self.welcome = app, welcome
        look, cfg = app.look, app.cfg
        P = look.px
        self.title(f"Welcome to {APP_NAME}" if welcome else "Settings")
        self.configure(bg=look.p["bg"])
        self.transient(app.root)
        self.resizable(False, False)

        body = ttk.Frame(self, padding=(P(26), P(22), P(26), P(20)))
        body.pack(fill="both", expand=True)
        body.columnconfigure((0, 1), weight=1, uniform="col")
        wrap = P(360)

        hero = ttk.Frame(body)
        hero.grid(row=0, column=0, columnspan=2, sticky="ew")
        hero.columnconfigure(1, weight=1)
        if welcome:
            ttk.Label(hero, image=look.logo(48)).grid(row=0, column=0, rowspan=2, padx=(0, P(14)))
            ttk.Label(hero, text=f"Welcome to {APP_NAME}", style="Title.TLabel").grid(row=0, column=1, sticky="sw")
            ttk.Label(hero, text="Transcription works right away. Speaker labels (Speaker 1, Speaker 2\u2026) use the "
                                 "free pyannote model, which needs a Hugging Face access token for a one-time download.",
                      style="Muted.TLabel", wraplength=P(560), justify="left").grid(row=1, column=1, sticky="nw")
        else:
            ttk.Label(hero, text="Settings", style="Title.TLabel").grid(row=0, column=0, sticky="w")

        def card(column, row, title):
            frame = ttk.Frame(body)
            frame.grid(row=row, column=column, sticky="nsew", padx=(0, P(16)) if column == 0 and not welcome else 0,
                       pady=(P(18), 0))
            frame.columnconfigure(0, weight=1)
            ttk.Label(frame, text=title, style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, P(8)))
            inner = ttk.Frame(frame, style="Card.TFrame", padding=(P(16), P(12), P(16), P(14)))
            inner.grid(row=1, column=0, sticky="nsew")
            inner.columnconfigure(1, weight=1)
            frame.rowconfigure(1, weight=1)
            return inner

        def note(parent, row, text, columnspan=3):
            label = ttk.Label(parent, text=text, style="Hint.TLabel", wraplength=wrap, justify="left")
            label.grid(row=row, column=0, columnspan=columnspan, sticky="w", pady=(P(6), 0))
            return label

        # -- speaker labels (both modes)
        speakers = card(0, 1, "Speaker labels")
        if welcome:
            speakers.master.grid_configure(columnspan=2)
        steps = (("1.  Create a free Hugging Face token with \u201cRead\u201d access", TOKENS_URL, "Token page"),
                 ("2.  Accept the conditions of pyannote/speaker-diarization-community-1", DIARIZATION_URL, "Model page"))
        for row, (text, url, button) in enumerate(steps):
            ttk.Label(speakers, text=text, wraplength=P(270 if not welcome else 420), justify="left").grid(
                row=row, column=0, columnspan=2, sticky="w", pady=P(3))
            ttk.Button(speakers, text=button, image=look.button_image("link"), compound="left",
                       command=lambda u=url: webbrowser.open(u)).grid(row=row, column=2, sticky="e", padx=(P(10), 0))
        ttk.Label(speakers, text="3.  Paste the token").grid(row=2, column=0, columnspan=3, sticky="w", pady=(P(8), P(4)))
        self.token_var = tk.StringVar(value=cfg["hf_token"])
        self.token_entry = ttk.Entry(speakers, textvariable=self.token_var, show="\u2022")
        self.token_entry.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.show_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(speakers, text="Show", variable=self.show_var, style="Switch.TCheckbutton",
                        command=lambda: self.token_entry.configure(show="" if self.show_var.get() else "\u2022")
                        ).grid(row=3, column=2, sticky="e", padx=(P(10), 0))
        ready = app._diarization_ready()
        status = ttk.Label(speakers, text=("  Speaker model downloaded. The token isn't needed any more."
                                           if ready else "  Speaker model not downloaded yet."),
                           image=look.icon("success" if ready else "info",
                                           look.p["status"]["ok"] if ready else look.p["muted"], 14),
                           compound="left", style="Hint.TLabel")
        status.grid(row=4, column=0, columnspan=3, sticky="w", pady=(P(10), 0))
        note(speakers, 5, "Stored only in config.json in the app folder and sent only to huggingface.co, for that "
                          "one download. Your audio never leaves this PC.").configure(wraplength=P(560 if welcome else 360))

        if not welcome:
            # -- appearance (left column, under speaker labels)
            appearance = card(0, 2, "Appearance")
            ttk.Label(appearance, text="Theme").grid(row=0, column=0, sticky="w")
            self.theme_var = tk.StringVar(value=THEME_LABELS.get(cfg.get("theme", "system"), THEME_LABELS["system"]))
            ttk.Combobox(appearance, textvariable=self.theme_var, values=list(THEME_LABELS.values()),
                         state="readonly", width=16).grid(row=0, column=1, sticky="e")
            note(appearance, 1, "Match Windows follows your light/dark setting in Windows Personalization.")

            # -- transcript files (right column)
            files = card(1, 1, "Transcript files")
            self.mode_var = tk.StringVar(value=cfg["output_mode"])
            ttk.Radiobutton(files, text="Save next to the original file", value="source",
                            variable=self.mode_var).grid(row=0, column=0, columnspan=3, sticky="w")
            ttk.Radiobutton(files, text="Save in this folder:", value="folder",
                            variable=self.mode_var).grid(row=1, column=0, columnspan=3, sticky="w", pady=(P(6), P(6)))
            self.dir_var = tk.StringVar(value=cfg["output_dir"])
            ttk.Entry(files, textvariable=self.dir_var).grid(row=2, column=0, columnspan=2, sticky="ew")
            ttk.Button(files, text="Browse\u2026", image=look.button_image("folder"), compound="left",
                       command=self._browse_output).grid(row=2, column=2, sticky="e", padx=(P(10), 0))
            self.subtitles_var = tk.BooleanVar(value=bool(cfg.get("subtitles", True)))
            ttk.Checkbutton(files, text="Also save subtitles (.srt and .vtt)", variable=self.subtitles_var,
                            style="Switch.TCheckbutton").grid(row=3, column=0, columnspan=3, sticky="w", pady=(P(12), 0))
            note(files, 4, "The .txt is always saved. With subtitles off and speaker labels off, the word-timing "
                           "step is skipped: about 25% faster.")

            # -- transcription
            tuning = card(1, 2, "Transcription")
            ttk.Label(tuning, text="Language (large-v3 only)").grid(row=0, column=0, sticky="w")
            self.language_var = tk.StringVar(value="Auto-detect" if cfg["language"] == "auto" else "English")
            ttk.Combobox(tuning, textvariable=self.language_var, values=["English", "Auto-detect"],
                         state="readonly", width=14).grid(row=0, column=1, columnspan=2, sticky="e")
            ttk.Label(tuning, text="CPU threads").grid(row=1, column=0, sticky="w", pady=(P(8), 0))
            self.threads_var = tk.StringVar(value=str(cfg["threads"] or 0))
            ttk.Spinbox(tuning, from_=0, to=os.cpu_count() or 16, textvariable=self.threads_var, width=6
                        ).grid(row=1, column=1, columnspan=2, sticky="e", pady=(P(8), 0))
            note(tuning, 2, f"0 = automatic ({max(4, (os.cpu_count() or 6) - 2)} on this PC).")

            # -- models (full width)
            models = card(0, 3, "Models")
            models.master.grid_configure(columnspan=2, padx=0)
            models_dir = config.models_dir(cfg)
            ttk.Label(models, text=str(models_dir), style="Muted.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Button(models, text="Open folder", image=look.button_image("folder"), compound="left",
                       command=lambda: (models_dir.mkdir(parents=True, exist_ok=True), os.startfile(models_dir))
                       ).grid(row=0, column=1, sticky="e", padx=(P(10), 0))
            ttk.Button(models, text="Download models now", image=look.button_image("download"), compound="left",
                       command=lambda: self._save(then=app.start_prepare)).grid(row=0, column=2, sticky="e",
                                                                                 padx=(P(8), 0))
            note(models, 1, "Downloads the selected speech model plus the timing and speaker models, so the app works "
                            "with no internet connection afterwards.").configure(wraplength=P(720))

        footer = ttk.Frame(body)
        footer.grid(row=9, column=0, columnspan=2, sticky="e", pady=(P(22), 0))
        ttk.Button(footer, text="Save token" if welcome else "Save", style="Accent.TButton",
                   command=self._save).pack(side="right")
        ttk.Button(footer, text="Skip for now" if welcome else "Cancel", command=self._dismiss
                   ).pack(side="right", padx=(0, P(8)))
        self.protocol("WM_DELETE_WINDOW", self._dismiss)
        self.bind("<Escape>", lambda e: self._dismiss())
        self.bind("<Return>", lambda e: self._save())

        set_title_bar(self, look.theme == "dark")
        self.update_idletasks()
        root = app.root
        x = root.winfo_rootx() + (root.winfo_width() - self.winfo_reqwidth()) // 2
        y = root.winfo_rooty() + max((root.winfo_height() - self.winfo_reqheight()) // 3, 0)
        self.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        self.deiconify()
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
                APP_NAME, "Hugging Face access tokens start with \u201chf_\u201d. Save this value anyway?", parent=self):
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
            cfg["theme"] = next((k for k, v in THEME_LABELS.items() if v == self.theme_var.get()), "system")
            cfg["subtitles"] = bool(self.subtitles_var.get())
            try:
                cfg["threads"] = max(0, int(self.threads_var.get() or 0))
            except ValueError:
                messagebox.showerror(APP_NAME, "CPU threads must be a whole number (0 = automatic).", parent=self)
                return
        threads_changed = cfg["threads"] != self.app.cfg["threads"]
        theme_changed = cfg.get("theme") != self.app.cfg.get("theme")
        self.app.cfg.update(cfg)
        if not self.app._save_config():
            return
        self.destroy()
        self.app.on_settings_saved(threads_changed, theme_changed)
        if then:
            then()


class SpeakerNamesDialog(tk.Toplevel):
    """Give the detected voices real names; rewrites the .txt/.srt/.vtt and the on-screen transcript."""

    CLIP_SECONDS = 8

    def __init__(self, app: App, speakers: list[SpeakerSummary], outputs: dict):
        super().__init__(app.root)
        self.withdraw()
        self.app, self.outputs = app, outputs
        look, P = app.look, app.look.px
        self.title("Name speakers")
        self.configure(bg=look.p["bg"])
        self.transient(app.root)
        self.resizable(False, False)
        source = Path(outputs["source"]) if outputs.get("source") else None
        self.source = source if source is not None and source.exists() else None
        self.ffmpeg = shutil.which("ffmpeg")
        can_play = winsound is not None and self.source is not None and self.ffmpeg is not None
        self.playing: int | None = None
        self._stop_after = None
        self._clips: list[Path] = []

        body = ttk.Frame(self, padding=(P(26), P(22), P(26), P(20)))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        ttk.Label(body, text="Name the speakers", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(body, text="Type a name for each voice. It replaces the label everywhere in this transcript and its "
                             "subtitle files. Leave a box empty to keep the current name. Giving two voices the same "
                             "name merges them.",
                  style="Muted.TLabel", wraplength=P(660), justify="left").grid(row=1, column=0, sticky="w",
                                                                                 pady=(P(4), P(14)))
        card = ttk.Frame(body, style="Card.TFrame", padding=(P(16), P(10), P(16), P(10)))
        card.grid(row=2, column=0, sticky="ew")
        card.columnconfigure(1, weight=1)
        style = ttk.Style(self)
        self.entries: dict[str, ttk.Entry] = {}
        self.play_buttons: dict[int, ttk.Button] = {}
        for i, speaker in enumerate(speakers):
            row = i * 2
            if i:
                ttk.Separator(card).grid(row=row - 1, column=0, columnspan=4, sticky="ew", pady=P(8))
            label_style = f"Speaker{i % SPEAKER_TAGS}.TLabel"
            style.configure(label_style, foreground=look.p["speakers"][i % SPEAKER_TAGS],
                            font=look.fonts["reading_strong"])
            who = ttk.Frame(card)
            who.grid(row=row, column=0, sticky="nw", padx=(0, P(18)), pady=P(4))
            ttk.Label(who, text=speaker.name, style=label_style).pack(anchor="w")
            ttk.Label(who, text=f"{speaker.turns} turn{'s' if speaker.turns != 1 else ''}  ·  "
                                f"{human_duration(speaker.seconds)}", style="Hint.TLabel").pack(anchor="w")
            quote = speaker.sample if len(speaker.sample) <= 150 else speaker.sample[:147].rsplit(" ", 1)[0] + "…"
            ttk.Label(card, text=f"“{quote}”", style="Hint.TLabel", wraplength=P(330),
                      justify="left").grid(row=row, column=1, sticky="w", pady=P(4))
            play = ttk.Button(card, text="Play", image=look.button_image("play"), compound="left", width=6,
                              command=lambda i=i, s=speaker: self.toggle_play(i, s))
            play.grid(row=row, column=2, padx=(P(14), P(10)))
            if not can_play:
                play.state(["disabled"])
            self.play_buttons[i] = play
            entry = ttk.Entry(card, width=22)
            entry.grid(row=row, column=3, sticky="e")
            self.entries[speaker.name] = entry

        if not can_play:
            reason = ("the original recording isn't next to the transcript any more" if self.source is None
                      else "ffmpeg wasn't found")
            ttk.Label(body, text=f"Play is unavailable: {reason}.", style="Hint.TLabel").grid(
                row=3, column=0, sticky="w", pady=(P(8), 0))
        footer = ttk.Frame(body)
        footer.grid(row=4, column=0, sticky="e", pady=(P(18), 0))
        ttk.Button(footer, text="Save names", style="Accent.TButton", command=self.save).pack(side="right")
        ttk.Button(footer, text="Cancel", command=self.close).pack(side="right", padx=(0, P(8)))
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Escape>", lambda e: self.close())
        self.bind("<Return>", lambda e: self.save())

        set_title_bar(self, look.theme == "dark")
        self.update_idletasks()
        root = app.root
        x = root.winfo_rootx() + (root.winfo_width() - self.winfo_reqwidth()) // 2
        y = root.winfo_rooty() + max((root.winfo_height() - self.winfo_reqheight()) // 3, 0)
        self.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        self.deiconify()
        self.grab_set()
        if self.entries:
            next(iter(self.entries.values())).focus_set()

    # -- voice samples

    def toggle_play(self, index: int, speaker: SpeakerSummary):
        if self.playing == index:
            self.stop()
            return
        self.stop()
        start = max(speaker.sample_start - 0.2, 0.0)
        length = min(max(speaker.sample_end - speaker.sample_start + 0.4, 2.0), self.CLIP_SECONDS)
        clip = Path(tempfile.gettempdir()) / f"local-transcriber-voice-{os.getpid()}-{index}.wav"
        self.configure(cursor="watch")
        self.update_idletasks()
        try:
            subprocess.run([self.ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-ss", f"{start:.2f}",
                            "-t", f"{length:.2f}", "-i", str(self.source), "-vn", "-ac", "1", "-ar", "22050", str(clip)],
                           capture_output=True, check=True, timeout=60, creationflags=NO_WINDOW)
            winsound.PlaySound(str(clip), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            messagebox.showerror(APP_NAME, f"Couldn't play this voice sample.\n\n{exc}", parent=self)
            return
        finally:
            self.configure(cursor="")
        self._clips.append(clip)
        self.playing = index
        self.play_buttons[index].configure(text="Stop", image=self.app.look.button_image("stop"))
        self._stop_after = self.after(int(length * 1000) + 300, self.stop)

    def stop(self):
        if self._stop_after is not None:
            self.after_cancel(self._stop_after)
            self._stop_after = None
        if self.playing is not None:
            try:
                winsound.PlaySound(None, 0)
            except RuntimeError:
                pass
            self.play_buttons[self.playing].configure(text="Play", image=self.app.look.button_image("play"))
            self.playing = None

    def close(self):
        self.stop()
        for clip in self._clips:
            try:
                clip.unlink()
            except OSError:
                pass
        self.destroy()

    # -- saving

    def save(self):
        mapping = {}
        for old, entry in self.entries.items():
            new = " ".join(entry.get().split())
            if not new:
                continue
            problem = name_problem(new)
            if problem:
                messagebox.showerror(APP_NAME, problem, parent=self)
                entry.focus_set()
                return
            mapping[old] = new
        if not mapping:
            self.close()
            return
        files = {kind: path for kind, path in self.outputs.items() if kind in ("txt", "srt", "vtt")}
        try:
            changed = rename_outputs(files, mapping)
        except OSError as exc:
            messagebox.showerror(APP_NAME, f"Couldn't update the transcript files.\n\n{exc}\n\n"
                                           "If one of them is open in another program, close it and try again.",
                                 parent=self)
            return
        self.close()
        self.app.reload_transcript(changed)


def _enable_dpi_awareness():
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # crisp text on scaled displays
        except (AttributeError, OSError):
            pass


def _set_app_id():
    """Give the process its own taskbar identity (otherwise Windows groups it under pythonw.exe)."""
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except (AttributeError, OSError):
            pass


def main():
    _set_app_id()
    _enable_dpi_awareness()
    root, dnd = None, False
    if TkinterDnD is not None:
        try:
            root, dnd = TkinterDnD.Tk(), True
        except (tk.TclError, RuntimeError):
            root = None
    if root is None:
        root = tk.Tk()
    root.withdraw()  # finish styling (incl. the dark title bar) before the first paint
    icon = env.APP_DIR / "assets" / "app.ico"
    if icon.exists():
        try:
            root.iconbitmap(default=str(icon))  # also used by the dialogs
        except tk.TclError:
            pass
    App(root, dnd)
    root.deiconify()
    root.mainloop()
