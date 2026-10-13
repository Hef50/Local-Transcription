"""Turn library exceptions into short, actionable messages.

The GUI shows `message` and `hint`; `detail` (the traceback) sits behind a Details button.
"""
from __future__ import annotations

import errno
import re
import traceback

TOKENS_URL = "https://huggingface.co/settings/tokens"
DIARIZATION_URL = "https://huggingface.co/pyannote/speaker-diarization-community-1"

STAGE_VERBS = {
    "decode": "reading the audio",
    "download": "downloading models",
    "prepare": "loading the speech libraries",
    "load_asr": "loading the speech model",
    "transcribe": "transcribing",
    "load_align": "loading the word-timing model",
    "align": "aligning words",
    "load_diarize": "loading the speaker model",
    "diarize": "identifying speakers",
    "write": "writing the transcript files",
}
TOKEN_STAGES = {"download", "load_diarize"}
NETWORK_ERRORS = {
    "LocalEntryNotFoundError", "OfflineModeIsEnabled", "ConnectionError", "ConnectTimeout",
    "ReadTimeout", "URLError", "gaierror", "NameResolutionError", "MaxRetryError",
}


class PipelineError(Exception):
    def __init__(self, code: str, message: str, hint: str = "", detail: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.detail = detail

    def to_event(self) -> dict:
        return {"event": "error", "code": self.code, "message": self.message,
                "hint": self.hint, "detail": self.detail}


def _chain(exc: BaseException | None):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def classify(exc: BaseException, stage: str | None = None) -> PipelineError:
    if isinstance(exc, PipelineError):
        return exc
    chain = list(_chain(exc))
    names = {type(e).__name__ for e in chain}
    text = " | ".join(f"{type(e).__name__}: {e}" for e in chain).lower()
    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    def err(code, message, hint):
        return PipelineError(code, message, hint, detail)

    if any(isinstance(e, MemoryError) for e in chain) or any(
            s in text for s in ("bad allocation", "not enough memory", "defaultcpuallocator")):
        return err("OUT_OF_MEMORY", "The computer ran out of memory while processing.",
                   "Close other heavy apps, or choose a smaller model (small.en).")
    if any(isinstance(e, OSError) and e.errno == errno.ENOSPC for e in chain) or any(
            s in text for s in ("no space left", "not enough space on the disk")):
        return err("DISK_FULL", "The disk is full.",
                   "Free up space on the drive with the models folder or the output folder.")
    # A gated repo answers 401 for a bad token and 403 for a good token without accepted terms.
    if re.search(r"\b401\b", text) or "invalid user token" in text or (
            "RepositoryNotFoundError" in names and stage in TOKEN_STAGES):
        return err("HF_TOKEN_INVALID", "Hugging Face rejected your access token.",
                   f"Check that you copied the whole token, or create a new one with 'Read' access at "
                   f"{TOKENS_URL} and paste it in Settings.")
    if "GatedRepoError" in names or (re.search(r"\b403\b", text) and "hugging" in text):
        return err("HF_GATED",
                   "Your Hugging Face account doesn't have access to the speaker model yet.",
                   f"Open {DIARIZATION_URL}, accept the conditions while signed in to the account "
                   "that owns your token, then try again.")
    if names & NETWORK_ERRORS or any(s in text for s in (
            "failed to resolve", "getaddrinfo failed", "max retries exceeded", "offline mode")):
        return err("MODEL_NOT_AVAILABLE",
                   "A model this job needs isn't downloaded yet, and the internet isn't reachable.",
                   "Connect to the internet and run it once more (or use Settings > Download models). "
                   "After that, everything works offline.")
    if any(isinstance(e, PermissionError) for e in chain):
        return err("PERMISSION_DENIED", f"Windows blocked access to a file: {exc}",
                   "If a transcript file is open in another program, close it, "
                   "or choose a different output folder in Settings.")
    first_line = (str(exc).strip().splitlines() or [type(exc).__name__])[0][:300]
    where = f" while {STAGE_VERBS[stage]}" if stage in STAGE_VERBS else ""
    return err("UNEXPECTED", f"Something went wrong{where}: {first_line}",
               "Click Details for the full log.")
