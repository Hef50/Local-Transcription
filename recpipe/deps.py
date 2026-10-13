"""First-run checks: Python packages and ffmpeg."""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

from . import env

PACKAGES = {  # import name -> pip name
    "whisperx": "whisperx",
    "faster_whisper": "faster-whisper",
    "pyannote.audio": "pyannote-audio",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "nltk": "nltk",
}


def missing_packages() -> list[str]:
    if getattr(sys, "frozen", False):
        return []  # bundled into the .exe
    importlib.invalidate_caches()
    missing = []
    for module, dist in PACKAGES.items():
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(dist)
    return missing


def ffmpeg_missing() -> bool:
    return shutil.which("ffmpeg") is None


def in_virtualenv() -> bool:
    return sys.prefix != sys.base_prefix


def console_python() -> str:
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return str(exe)


def install_command() -> list[str]:
    requirements = str(env.APP_DIR / "requirements.txt")
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", console_python(), "-r", requirements]
    return [console_python(), "-m", "pip", "install", "-r", requirements]


def install_environment() -> dict:
    environ = dict(os.environ)
    environ["UV_CACHE_DIR"] = str(env.APP_DIR / ".cache" / "uv")  # keep caches off C:
    environ["PIP_CACHE_DIR"] = str(env.APP_DIR / ".cache" / "pip")
    return environ
