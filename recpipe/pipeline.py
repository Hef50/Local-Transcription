"""File in -> speaker-labelled transcript out, reporting progress as events.

A Session keeps loaded models between jobs, so the app's warm worker and
multi-file CLI runs skip the ~30 s library import and the model loads after the
first file. torch / whisperx / pyannote are imported inside Session.load_libs(),
after env.configure(), so importing this module stays cheap (the GUI uses its
constants).
"""
from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from . import env, formats
from .errors import DIARIZATION_URL, TOKENS_URL, PipelineError, classify

Emit = Callable[[dict], None]

MODELS = ("small.en", "medium.en", "large-v3")
DEFAULT_MODEL = "medium.en"
SUPPORTED_EXTS = (".wav", ".mp3", ".m4a", ".mp4", ".flac", ".ogg", ".opus", ".aac", ".wma",
                  ".webm", ".mkv", ".mov")
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
SAMPLE_RATE = 16000
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Components that must be on disk before a run can be fully offline.
ALIGN_EN = "align-en"
NLTK_PUNKT = "nltk-punkt_tab"
DIARIZATION = "diarization-community-1"
WHISPER_MB = {"small.en": 484, "medium.en": 1528, "large-v3": 3087}
ALIGN_MB = 360

# Speed/accuracy presets. pyannote scores every second of audio 1/step times with a
# sliding window (default step 0.1 = ten passes); larger steps cut that work directly.
# Measured on a 5-minute two-speaker file: VAD step 0.5 gives identical speech chunks
# 5x faster; diarization step 0.25 is 3x faster with 99.1% of frames labelled as the
# default, step 0.5 is 6x faster at 98.3%. Beam 1 and int8 alignment each save <10%.
QUALITY = ("fast", "balanced", "best")
DEFAULT_QUALITY = "balanced"
PRESETS = {
    "fast": {"beam_size": 1, "vad_step": 0.5, "diarize_step": 0.5, "align_int8": True},
    "balanced": {"beam_size": 5, "vad_step": 0.5, "diarize_step": 0.25, "align_int8": False},
    "best": {"beam_size": 5, "vad_step": 0.5, "diarize_step": 0.1, "align_int8": False},
}

# Starting guesses for ETA on an i7-1355U (int8, 10 threads). Each finished run
# stores measured values in calibration.json, which then take over.
DEFAULT_RTF = {  # processing seconds per second of audio, measured on a 5-minute file under sustained load
    "transcribe:small.en:b5": 0.29, "transcribe:medium.en:b5": 0.55, "transcribe:large-v3:b5": 1.2,
    "align:fp32": 0.16, "align:int8": 0.145,
    "diarize:s0.1": 0.82, "diarize:s0.25": 0.27, "diarize:s0.5": 0.13,
}
DEFAULT_FIXED = {  # seconds, independent of audio length
    "prepare": 20, "load_asr:small.en": 3, "load_asr:medium.en": 8,
    "load_asr:large-v3": 18, "load_align": 2, "load_diarize": 2, "write": 1,
}


def default_rtf(key: str) -> float:
    """Fill in rates we have not measured from the ones we have."""
    if key in DEFAULT_RTF:
        return DEFAULT_RTF[key]
    kind, *rest = key.split(":")
    if kind == "transcribe":
        model, beam = rest
        return DEFAULT_RTF[f"transcribe:{model}:b5"] * (0.7 if beam == "b1" else 1.0)
    if kind == "diarize":
        step = float(rest[0][1:])
        return DEFAULT_RTF["diarize:s0.1"] * (0.1 / step) ** 0.8
    if kind == "align":
        return DEFAULT_RTF["align:fp32"] * (0.6 if rest[0] == "int8" else 1.0)
    raise KeyError(key)


def english_only(model: str) -> bool:
    return model.endswith(".en") or model.startswith("distil")


@dataclass
class Job:
    source: Path
    model: str = DEFAULT_MODEL
    output_dir: Path | None = None  # None: next to the source file
    diarize: bool = True
    hf_token: str = ""
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    language: str | None = "en"  # None: auto-detect (multilingual models only)
    models_dir: Path = env.DEFAULT_MODELS_DIR
    threads: int = 0  # 0: all cores but two
    batch_size: int = 8
    quality: str = DEFAULT_QUALITY
    # Explicit overrides win over the preset.
    beam_size: int | None = None
    vad_step: float | None = None
    diarize_step: float | None = None
    align_int8: bool | None = None

    def settings(self) -> dict:
        preset = PRESETS.get(self.quality, PRESETS[DEFAULT_QUALITY])
        return {key: (getattr(self, key) if getattr(self, key) is not None else value)
                for key, value in preset.items()}

    @property
    def effective_language(self) -> str | None:
        return "en" if english_only(self.model) else self.language


# ---------------------------------------------------------------- model readiness

def required_components(model: str, diarize: bool) -> list[str]:
    components = [f"whisper-{model}", ALIGN_EN, NLTK_PUNKT]
    return components + [DIARIZATION] if diarize else components


def missing_components(models_dir: Path, model: str, diarize: bool) -> list[str]:
    ready = Path(models_dir) / "ready"
    return [c for c in required_components(model, diarize) if not (ready / c).exists()]


def _mark_ready(models_dir: Path, component: str) -> None:
    ready = Path(models_dir) / "ready"
    ready.mkdir(parents=True, exist_ok=True)
    (ready / component).write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")


def clear_ready(models_dir: Path) -> None:
    """Forget what's downloaded, so the next run re-checks online (files already there are reused)."""
    shutil.rmtree(Path(models_dir) / "ready", ignore_errors=True)


def token_missing_error() -> PipelineError:
    return PipelineError(
        "HF_TOKEN_MISSING",
        "Speaker labels need a Hugging Face token for a one-time model download.",
        f"Create a free 'Read' token at {TOKENS_URL}, accept the model conditions at "
        f"{DIARIZATION_URL}, and paste the token in Settings. Or turn speaker labels off.",
    )


# ---------------------------------------------------------------- input checks

def check_source(path: Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise PipelineError("FILE_NOT_FOUND", f"File not found: {path}",
                            "It may have been moved, renamed or deleted.")
    if path.is_dir() or path.suffix.lower() not in SUPPORTED_EXTS:
        raise PipelineError("UNSUPPORTED_FORMAT", f"“{path.name}” isn't an audio or video file this app can read.",
                            "Supported: " + ", ".join(e.lstrip(".") for e in SUPPORTED_EXTS) + ".")
    if path.stat().st_size == 0:
        raise PipelineError("CORRUPT_FILE", f"“{path.name}” is empty (0 bytes).",
                            "The recording may not have finished saving. Try exporting it again.")
    return path


def check_output_dir(folder: Path) -> Path:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / f".write-test-{os.getpid()}"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exc:
        raise PipelineError("OUTPUT_NOT_WRITABLE", f"Can't save transcripts in {folder}",
                            "Choose a different output folder in Settings (for example Documents).",
                            str(exc)) from exc
    return folder


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise PipelineError("FFMPEG_MISSING", "ffmpeg wasn't found, so the audio can't be read.",
                            "Install it with:  winget install Gyan.FFmpeg  then restart the app.")
    return ffmpeg


def probe_audio(path: Path) -> None:
    """Fail fast, with a clear message, on corrupted files or video without sound."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return  # decode_audio will still catch unreadable files
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type",
         "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        creationflags=NO_WINDOW, timeout=120,
    )
    if proc.returncode != 0:
        raise PipelineError("CORRUPT_FILE", f"“{path.name}” couldn't be opened. It may be corrupted or incomplete.",
                            "Check that it plays in a media player, or re-export it as .wav or .mp3.",
                            proc.stderr.strip())
    try:
        streams = json.loads(proc.stdout or "{}").get("streams") or []
    except ValueError:
        streams = [None]  # unexpected output: let decoding decide
    if not streams:
        raise PipelineError("NO_AUDIO", f"“{path.name}” has no audio track.",
                            "Pick a recording that contains sound.")


def decode_audio(path: Path):
    """Decode to 16 kHz mono float32 (same as whisperx.load_audio, with readable errors)."""
    import numpy as np

    ffmpeg = _require_ffmpeg()
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "0", "-i", str(path),
         "-vn", "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True, creationflags=NO_WINDOW,
    )
    if proc.returncode != 0 or not proc.stdout:
        log = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise PipelineError("CORRUPT_FILE", f"“{path.name}” couldn't be decoded. It may be corrupted or incomplete.",
                            "Check that it plays in a media player, or re-export it as .wav or .mp3.",
                            "\n".join(log[-20:]))
    return np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0


# ---------------------------------------------------------------- downloads

def _folder_bytes(folder: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


class FolderWatcher:
    """Download progress from a cache folder's growth (the libraries offer no callback)."""

    def __init__(self, folder: Path, expected_mb: float, emit: Emit):
        self.folder, self.expected, self.emit = folder, expected_mb * 1024 * 1024, emit
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self):
        self._base, self._t0 = _folder_bytes(self.folder), time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        self._thread.join(timeout=2)

    def _loop(self):
        while not self._stop.wait(1.0):
            got = max(_folder_bytes(self.folder) - self._base, 0)
            rate = got / max(time.monotonic() - self._t0, 1e-6)
            eta = (self.expected - got) / rate if rate > 0 and got < self.expected else None
            self.emit({"event": "progress", "stage": "download", "overall_pct": None,
                       "stage_pct": round(min(got / self.expected, 0.99) * 100, 1),
                       "eta_s": None if eta is None else round(eta),
                       "detail": f"{got / 1048576:,.0f} of ~{self.expected / 1048576:,.0f} MB"})


def _size(mb: float) -> str:
    return f"{mb / 1024:.1f} GB" if mb >= 1000 else f"{mb:.0f} MB"


def _download(component: str, job: Job, emit: Emit) -> None:
    models_dir = Path(job.models_dir)
    hf_cache = models_dir / "hf"

    def stage(label):
        emit({"event": "stage", "stage": "download", "label": label})

    if component.startswith("whisper-"):
        model = component.removeprefix("whisper-")
        stage(f"Downloading the {model} speech model ({_size(WHISPER_MB[model])}, one time only)…")
        from faster_whisper.utils import _MODELS, download_model

        repo_folder = "models--" + _MODELS[model].replace("/", "--")
        with FolderWatcher(hf_cache / repo_folder, WHISPER_MB[model], emit):
            download_model(model, cache_dir=str(hf_cache))
    elif component == ALIGN_EN:
        stage(f"Downloading the word-timing model ({_size(ALIGN_MB)}, one time only)…")
        import torchaudio

        torch_dir = models_dir / "torch"
        with FolderWatcher(torch_dir, ALIGN_MB, emit):
            torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model(dl_kwargs={"model_dir": str(torch_dir)})
    elif component == NLTK_PUNKT:
        stage("Downloading sentence-splitting data (one time only)…")
        import nltk

        nltk.download("punkt_tab", download_dir=str(models_dir / "nltk_data"), quiet=True,
                      raise_on_error=True)
    elif component == DIARIZATION:
        if not job.hf_token:
            raise token_missing_error()
        stage("Downloading the speaker-diarization model (one time only)…")
        from huggingface_hub import snapshot_download

        snapshot_download(DIARIZATION_MODEL, cache_dir=str(hf_cache), token=job.hf_token)
    _mark_ready(models_dir, component)


# ---------------------------------------------------------------- progress + ETA

@dataclass
class Stage:
    key: str
    label: str
    estimate: float
    cal_key: str
    per_audio_second: bool
    cached: bool = False  # already done by an earlier job in this process
    track: str = "main"  # "main" (transcribe, align, write) or "diarize" (runs in parallel)


def load_calibration() -> dict:
    try:
        data = json.loads(env.CALIBRATION_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_calibration(measured: dict[str, float]) -> None:
    calibration = load_calibration()
    for key, value in measured.items():
        old = calibration.get(key)
        calibration[key] = round(value if not isinstance(old, (int, float)) else 0.5 * old + 0.5 * value, 4)
    try:
        tmp = env.CALIBRATION_PATH.with_name(env.CALIBRATION_PATH.name + ".tmp")
        tmp.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        os.replace(tmp, env.CALIBRATION_PATH)
    except OSError:
        pass  # a missing ETA calibration is not worth failing a finished job for


def plan_stages(job: Job, duration: float, calibration: dict, cached: set[str] = frozenset()) -> list[Stage]:
    settings, m = job.settings(), job.model

    def fixed(key, stage_key):
        return 0.0 if stage_key in cached else float(calibration.get(key, DEFAULT_FIXED[key]))

    def rtf(key):
        return float(calibration.get(key, default_rtf(key)))

    transcribe_key = f"transcribe:{m}:b{settings['beam_size']}"
    align_key = f"align:{'int8' if settings['align_int8'] else 'fp32'}"
    diarize_key = f"diarize:s{settings['diarize_step']}"
    stages = [
        Stage("prepare", "Loading the speech libraries…", fixed("prepare", "prepare"), "prepare", False, "prepare" in cached),
        Stage("load_asr", f"Loading the {m} model…", fixed(f"load_asr:{m}", "load_asr"), f"load_asr:{m}", False, "load_asr" in cached),
        Stage("transcribe", f"Transcribing speech with {m}…", duration * rtf(transcribe_key), transcribe_key, True),
        Stage("load_align", "Loading the word-timing model…", fixed("load_align", "load_align"), "load_align", False, "load_align" in cached),
        Stage("align", "Aligning word timestamps…", duration * rtf(align_key), align_key, True),
    ]
    if job.diarize:
        stages += [
            Stage("load_diarize", "Loading the speaker model…", fixed("load_diarize", "load_diarize"), "load_diarize", False,
                  "load_diarize" in cached, track="diarize"),
            Stage("diarize", "Working out who is speaking…", duration * rtf(diarize_key), diarize_key, True, track="diarize"),
        ]
    return stages + [Stage("write", "Writing transcript files…", fixed("write", "write"), "write", False)]


class Progress:
    """One overall bar and an ETA across two concurrent tracks (transcribe/align and diarize).

    The bar is time-weighted by stage estimates. The ETA is the slower track's remaining
    time, where a running stage trusts its measured rate more as it progresses and later
    stages are scaled by how fast the heavy stages have been so far on this machine.
    """

    def __init__(self, emit: Emit, ctx: dict, stages: list[Stage], duration: float):
        self.emit, self.ctx, self.duration = emit, ctx, duration
        self.stages = {s.key: s for s in stages}
        self.state = {s.key: {"started": None, "fraction": 0.0, "done": False} for s in stages}
        self.total = sum(s.estimate for s in stages) or 1.0
        self.heavy_estimate = self.heavy_actual = 0.0  # completed per-audio-second stages
        self.run_started = time.monotonic()
        self.last_emit = 0.0
        self.measured: dict[str, float] = {}
        self.lock = threading.Lock()

    def begin(self, key: str) -> None:
        stage = self.stages[key]
        with self.lock:
            self.state[key]["started"] = time.monotonic()
            if stage.track == "main":
                self.ctx["stage"] = key
            if not stage.cached:
                self.emit({"event": "stage", "stage": key, "track": stage.track, "label": stage.label})
                self._report(force=True)

    def update(self, key: str, fraction: float) -> None:
        with self.lock:
            self.state[key]["fraction"] = min(max(fraction, 0.0), 1.0)
            self._report()

    def end(self, key: str) -> None:
        stage = self.stages[key]
        with self.lock:
            state = self.state[key]
            elapsed = time.monotonic() - (state["started"] or time.monotonic())
            state["fraction"], state["done"] = 1.0, True
            if not stage.cached:
                if stage.per_audio_second:
                    self.heavy_estimate += stage.estimate
                    self.heavy_actual += elapsed
                    if self.duration >= 20:  # very short clips give noisy rates
                        self.measured[stage.cal_key] = elapsed / self.duration
                else:
                    self.measured[stage.cal_key] = elapsed
            if stage.track != "main" and all(st["done"] for k, st in self.state.items() if self.stages[k].track == stage.track):
                self.emit({"event": "stage", "stage": key, "track": stage.track, "label": ""})  # track finished
            self._report(force=True)

    def _report(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_emit < 0.5:
            return
        self.last_emit = now
        est, act = self.heavy_estimate, self.heavy_actual
        done_estimate = 0.0
        remaining = {"main": 0.0, "diarize": 0.0}
        for key, stage in self.stages.items():
            state = self.state[key]
            done_estimate += stage.estimate * state["fraction"]
            if state["done"]:
                continue
            left = stage.estimate * (1 - state["fraction"])
            if state["started"] is not None:
                elapsed, fraction = now - state["started"], state["fraction"]
                if fraction > 0.02 and elapsed > 2:
                    weight = min(1.0, fraction * 4)
                    left = weight * elapsed * (1 - fraction) / fraction + (1 - weight) * left
                if stage.per_audio_second and fraction >= 0.1:
                    est, act = est + stage.estimate * fraction, act + elapsed
            remaining[stage.track] = remaining.get(stage.track, 0.0) + left
        speed = min(max(act / est, 0.33), 3.0) if est > 1 else 1.0
        eta = max(remaining.values()) * speed if remaining else 0.0
        self.emit({
            "event": "progress",
            "overall_pct": round(100 * done_estimate / self.total, 1),
            "eta_s": round(eta),
            "elapsed_s": round(now - self.run_started),
        })


# ---------------------------------------------------------------- the session

def _threads(threads: int) -> int:
    return threads if threads > 0 else max(4, (os.cpu_count() or 6) - 2)


class Session:
    """Loaded libraries and models, kept between jobs. One per process."""

    def __init__(self, models_dir: Path = env.DEFAULT_MODELS_DIR, threads: int = 0):
        self.models_dir = Path(models_dir)
        self.hf_cache = self.models_dir / "hf"
        self.threads = _threads(threads)
        self.configured = False
        self.libs: SimpleNamespace | None = None
        self.asr = self.asr_key = None
        self.aligner = self.align_key = None
        self.diarizer = self.diarize_key = None

    # -- environment and libraries

    def configure(self, offline: bool) -> None:
        if self.configured:
            env.set_offline(offline)
        else:
            env.configure(self.models_dir, offline)
            self.configured = True

    def load_libs(self) -> SimpleNamespace:
        if self.libs is not None:
            return self.libs
        # We hand pyannote in-memory audio, so its torchcodec file decoder is never needed.
        warnings.filterwarnings("ignore", message=r"\s*torchcodec is not installed correctly")
        # pyannote's pooling warns on a sub-frame chunk at the very end of a file; harmless.
        warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom")
        import torch

        torch.set_num_threads(self.threads)
        from whisperx.alignment import align, load_align_model
        from whisperx.asr import load_model
        from whisperx.diarize import DiarizationPipeline, assign_word_speakers
        from whisperx.log_utils import setup_logging

        setup_logging("warning")  # whisperx logs INFO chatter by default
        for name in ("lightning", "lightning.pytorch", "lightning.fabric", "pytorch_lightning", "lightning_fabric"):
            logging.getLogger(name).setLevel(logging.WARNING)  # "Lightning automatically upgraded your checkpoint..."
        self.libs = SimpleNamespace(torch=torch, align=align, load_align_model=load_align_model,
                                    load_model=load_model, DiarizationPipeline=DiarizationPipeline,
                                    assign_word_speakers=assign_word_speakers)
        return self.libs

    # -- models (each keyed by the settings that change it)

    @staticmethod
    def asr_key_for(job: Job) -> tuple:
        s = job.settings()
        return job.model, job.effective_language, s["beam_size"], s["vad_step"]

    def load_asr(self, job: Job):
        key = self.asr_key_for(job)
        if self.asr_key == key:
            return self.asr
        self.asr = self.asr_key = None
        gc.collect()
        model, language, beam, vad_step = key
        asr = self.libs.load_model(
            model, device="cpu", compute_type="int8", download_root=str(self.hf_cache), language=language,
            threads=self.threads, vad_method="pyannote",
            asr_options={"beam_size": beam, "best_of": beam},
        )
        vad = asr.vad_model.vad_pipeline._segmentation
        vad.step = vad_step * vad.duration
        self.asr, self.asr_key = asr, key
        return asr

    def load_aligner(self, language: str, int8: bool):
        key = (language, int8)
        if self.align_key == key:
            return self.aligner
        self.aligner = self.align_key = None
        gc.collect()
        model, meta = self.libs.load_align_model(language, "cpu", model_dir=str(self.models_dir / "torch"))
        if int8:
            model = self.libs.torch.ao.quantization.quantize_dynamic(model, {self.libs.torch.nn.Linear},
                                                                     dtype=self.libs.torch.qint8)
        self.aligner, self.align_key = (model, meta), key
        return self.aligner

    def load_diarizer(self, token: str, step: float):
        if self.diarize_key == step and self.diarizer is not None:
            return self.diarizer
        self.diarizer = self.diarize_key = None
        gc.collect()
        diarizer = self.libs.DiarizationPipeline(model_name=DIARIZATION_MODEL, token=token or None,
                                                 device="cpu", cache_dir=str(self.hf_cache))
        segmentation = diarizer.model._segmentation
        segmentation.step = step * segmentation.duration
        self.diarizer, self.diarize_key = diarizer, step
        return diarizer

    def cached_stages(self, job: Job) -> set[str]:
        cached = set()
        if self.libs is not None:
            cached.add("prepare")
        if self.asr_key == self.asr_key_for(job):
            cached.add("load_asr")
        settings = job.settings()
        if self.align_key == (job.effective_language or "en", settings["align_int8"]):
            cached.add("load_align")
        if job.diarize and self.diarize_key == settings["diarize_step"] and self.diarizer is not None:
            cached.add("load_diarize")
        return cached

    # -- entry points

    def run(self, job: Job, emit: Emit) -> dict:
        """Transcribe, align, diarize and write .txt/.srt/.vtt. Raises PipelineError."""
        return self._guard(self._run, job, emit, {"stage": "validate"})

    def prepare(self, job: Job, emit: Emit) -> dict:
        """Download everything a job with these settings needs, so later runs work offline."""
        return self._guard(self._prepare, job, emit, {"stage": "download"})

    def warm(self, job: Job, emit: Emit) -> dict:
        """Load libraries and whatever models are already on disk, so the next job starts at once."""
        return self._guard(self._warm, job, emit, {"stage": "prepare"})

    def _guard(self, fn, job: Job, emit: Emit, ctx: dict):
        try:
            return fn(job, emit, ctx)
        except PipelineError:
            raise
        except Exception as exc:
            err = classify(exc, ctx.get("stage"))
            if err.code == "MODEL_NOT_AVAILABLE":
                clear_ready(job.models_dir)
            raise err from exc

    def _prepare(self, job: Job, emit: Emit, ctx: dict) -> dict:
        missing = missing_components(self.models_dir, job.model, job.diarize)
        self.configure(offline=False)
        if DIARIZATION in missing and not job.hf_token:
            raise token_missing_error()
        for component in missing:
            _download(component, job, emit)
        env.set_offline(True)
        return {"prepared": True, "downloaded": missing, "model": job.model}

    def _warm(self, job: Job, emit: Emit, ctx: dict) -> dict:
        missing = missing_components(self.models_dir, job.model, job.diarize)
        self.configure(offline=not missing)
        settings = job.settings()
        emit({"event": "stage", "stage": "prepare", "label": "Loading the speech libraries…", "warm": True})
        self.load_libs()
        loaded = ["libs"]
        if f"whisper-{job.model}" not in missing:
            emit({"event": "stage", "stage": "load_asr", "label": f"Loading the {job.model} model…", "warm": True})
            self.load_asr(job)
            loaded.append(job.model)
        if ALIGN_EN not in missing and NLTK_PUNKT not in missing:
            ctx["stage"] = "load_align"
            self.load_aligner("en", settings["align_int8"])
            loaded.append("align")
        if job.diarize and DIARIZATION not in missing:
            ctx["stage"] = "load_diarize"
            self.load_diarizer(job.hf_token, settings["diarize_step"])
            loaded.append("diarization")
        return {"warmed": loaded}

    def _run(self, job: Job, emit: Emit, ctx: dict) -> dict:
        run_started = time.monotonic()
        source = check_source(job.source)
        out_dir = check_output_dir(Path(job.output_dir) if job.output_dir else source.parent)
        settings = job.settings()

        missing = missing_components(self.models_dir, job.model, job.diarize)
        self.configure(offline=not missing)
        if DIARIZATION in missing and not job.hf_token:
            raise token_missing_error()

        ctx["stage"] = "decode"
        emit({"event": "stage", "stage": "decode", "label": f"Reading {source.name}…"})
        _require_ffmpeg()
        probe_audio(source)
        audio = decode_audio(source)
        duration = len(audio) / SAMPLE_RATE

        ctx["stage"] = "download"
        for component in missing:
            _download(component, job, emit)
        if missing:
            env.set_offline(True)

        cached = self.cached_stages(job)
        progress = Progress(emit, ctx, plan_stages(job, duration, load_calibration(), cached), duration)
        emit({"event": "info", "duration_s": round(duration, 1), "model": job.model, "quality": job.quality,
              "estimate_s": round(progress.total), "threads": self.threads, "offline": not missing,
              "cached": sorted(cached)})

        progress.begin("prepare")
        libs = self.load_libs()
        progress.end("prepare")

        # Speaker diarization only needs the audio, so it runs alongside transcription.
        # Measured: 1.27x faster overall with the diarization thread on half the torch threads.
        diarization: dict = {}
        diarize_thread = None
        if job.diarize:
            def diarize_track():
                try:
                    libs.torch.set_num_threads(max(2, self.threads // 2))
                    progress.begin("load_diarize")
                    diarizer = self.load_diarizer(job.hf_token, settings["diarize_step"])
                    progress.end("load_diarize")
                    progress.begin("diarize")
                    diarization["turns"] = diarizer(
                        audio, num_speakers=job.num_speakers, min_speakers=job.min_speakers,
                        max_speakers=job.max_speakers,
                        progress_callback=lambda pct: progress.update("diarize", pct / 100))
                    progress.end("diarize")
                except BaseException as exc:  # reported from the main thread
                    diarization["error"] = exc
                finally:
                    libs.torch.set_num_threads(self.threads)

            diarize_thread = threading.Thread(target=diarize_track, name="diarize", daemon=True)
            diarize_thread.start()

        try:
            progress.begin("load_asr")
            language = job.effective_language
            asr = self.load_asr(job)
            progress.end("load_asr")

            progress.begin("transcribe")
            result = asr.transcribe(audio, batch_size=job.batch_size, language=language,
                                    progress_callback=lambda pct: progress.update("transcribe", pct / 100))
            progress.end("transcribe")
            language = result.get("language") or language or "en"
            if not result.get("segments"):
                raise PipelineError("NO_SPEECH", f"No speech was detected in “{source.name}”.",
                                    "Check that the recording has an audible voice and isn't silent or muted.")

            progress.begin("load_align")
            try:
                align_model, align_meta = self.load_aligner(language, settings["align_int8"])
            except Exception:
                if language == "en":
                    raise
                align_model = align_meta = None
                emit({"event": "warning",
                      "message": f"Word-level timing isn't available for language “{language}” right now.",
                      "hint": "Timestamps are per sentence instead. Running once with internet downloads "
                              "the timing model for this language."})
            progress.end("load_align")

            progress.begin("align")
            if align_model is not None:
                result = libs.align(result["segments"], align_model, align_meta, audio, "cpu",
                                    progress_callback=lambda pct: progress.update("align", pct / 100))
            progress.end("align")
        finally:
            if diarize_thread is not None:
                diarize_thread.join()  # never leave a stray thread chewing CPU behind an error

        if "error" in diarization:
            exc = diarization["error"]
            raise exc if isinstance(exc, PipelineError) else classify(exc, "diarize") from exc
        if job.diarize:
            result = libs.assign_word_speakers(diarization["turns"], result, fill_nearest=True)

        progress.begin("write")
        paths, speakers = formats.write_outputs(result, source, out_dir, model=job.model,
                                                duration=duration, label_speakers=job.diarize)
        progress.end("write")
        save_calibration(progress.measured)

        return {
            "txt": str(paths["txt"]), "srt": str(paths["srt"]), "vtt": str(paths["vtt"]),
            "speakers": speakers, "diarized": job.diarize, "language": language, "model": job.model,
            "quality": job.quality, "duration_s": round(duration, 1),
            "elapsed_s": round(time.monotonic() - run_started, 1),
        }


# One-shot conveniences (a throwaway Session).

def run(job: Job, emit: Emit) -> dict:
    return Session(job.models_dir, job.threads).run(job, emit)


def prepare(job: Job, emit: Emit) -> dict:
    return Session(job.models_dir, job.threads).prepare(job, emit)
