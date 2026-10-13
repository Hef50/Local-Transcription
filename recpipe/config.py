"""Settings in config.json next to the app. Nothing goes to the registry or %APPDATA%."""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import env

DEFAULTS = {
    "model": "medium.en",
    "speakers": "Auto-detect",
    "hf_token": "",
    "token_prompt_done": False,
    "output_mode": "source",  # "source": next to the dropped file; "folder": output_dir
    "output_dir": "",
    "language": "en",  # "en" or "auto"; only matters for the multilingual large-v3
    "threads": 0,  # 0 = automatic
    "quality": "balanced",  # fast | balanced | best (see pipeline.PRESETS)
    "models_dir": "",  # "" = <app folder>/models
}


def load(path: Path | None = None) -> dict:
    cfg = dict(DEFAULTS)
    try:
        data = json.loads(Path(path or env.CONFIG_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return cfg  # missing or unreadable: start from defaults rather than crash
    if isinstance(data, dict):
        cfg.update({k: v for k, v in data.items() if k in DEFAULTS})
    return cfg


def save(cfg: dict, path: Path | None = None) -> None:
    path = Path(path or env.CONFIG_PATH)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def models_dir(cfg: dict) -> Path:
    return Path(cfg.get("models_dir") or env.DEFAULT_MODELS_DIR)
