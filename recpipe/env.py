"""App paths and process environment.

configure() must run before huggingface_hub, transformers, torch or pyannote are
imported: several of them read these variables once, at import time.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _app_dir() -> Path:
    if getattr(sys, "frozen", False):  # PyInstaller build
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


# Portable layout: settings, calibration and models all live in the app folder.
APP_DIR = _app_dir()
CONFIG_PATH = APP_DIR / "config.json"
CALIBRATION_PATH = APP_DIR / "calibration.json"
DEFAULT_MODELS_DIR = APP_DIR / "models"


def configure(models_dir: Path, offline: bool) -> None:
    """Point every model cache at models_dir and switch off all telemetry."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        # pyannote.audio 4.x sends OpenTelemetry usage metrics (incl. audio duration)
        # to otel.pyannote.ai unless this is off. It is ON by default.
        "PYANNOTE_METRICS_ENABLED": "false",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        # Only send a token where we pass one explicitly (the diarization download).
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        # Keep every cache off C: (and inside the app folder).
        "HF_HOME": str(models_dir / "hf-home"),
        "TORCH_HOME": str(models_dir / "torch"),
        "NLTK_DATA": str(models_dir / "nltk_data"),
        "TOKENIZERS_PARALLELISM": "false",
    })
    if offline:
        # Everything is already downloaded: guarantee zero network access.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def set_offline(offline: bool) -> None:
    """Flip network access at runtime.

    The warm worker starts offline, goes online only for a one-time model download,
    then goes dark again. huggingface_hub reads its constant at call time, so updating
    it here is enough; the environment variables cover libraries imported later.
    """
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if offline:
            os.environ[key] = "1"
        else:
            os.environ.pop(key, None)
    try:
        import huggingface_hub.constants as hf_constants
        from huggingface_hub.utils import reset_sessions
    except ImportError:
        return
    hf_constants.HF_HUB_OFFLINE = offline
    reset_sessions()  # the offline adapter is mounted on cached HTTP sessions
