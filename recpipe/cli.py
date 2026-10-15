"""Command line: python transcribe.py meeting.m4a [--model medium.en] [--no-diarize]

With --jsonl (used by the GUI) stdout carries one JSON event per line and
everything else, including library chatter, goes to stderr. With --serve the
process stays alive and takes JSON commands on stdin, keeping models loaded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

from . import APP_NAME, config, env
from .errors import PipelineError, classify


def _jsonl_emitter():
    """Reserve the real stdout for events; send any other output (C libraries too) to stderr."""
    events = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    lock = threading.Lock()

    def emit(event: dict) -> None:
        with lock:
            events.write(json.dumps(event, ensure_ascii=False) + "\n")
            events.flush()

    return emit


class ConsoleReporter:
    """Human-readable progress on stderr."""

    def __init__(self):
        self.tty = sys.stderr.isatty()
        self.lock = threading.Lock()
        self.tracks: dict[str, str] = {}
        self.width = 0

    @property
    def label(self) -> str:
        return "   +   ".join(label for track in ("main", "diarize") if (label := self.tracks.get(track)))

    def __call__(self, event: dict) -> None:
        from .formats import clock, human_duration

        with self.lock:
            kind = event.get("event")
            if kind == "stage":
                self._end_line()
                self.tracks[event.get("track", "main")] = event.get("label", "")
                if not self.tty and event.get("label"):
                    print(f"  {event['label']}", file=sys.stderr)
            elif kind == "progress" and self.tty:
                pct = event.get("overall_pct")
                pct = event.get("stage_pct") if pct is None else pct
                parts = [f"  {self.label:<60}", f"{pct:5.1f}%" if pct is not None else ""]
                if event.get("eta_s") is not None:
                    parts.append(f"ETA {clock(event['eta_s'], event['eta_s'] >= 3600)}")
                line = "  ".join(p for p in parts if p)
                sys.stderr.write("\r" + line.ljust(self.width))
                self.width = len(line)
            elif kind == "info":
                self._end_line()
                print(f"  {human_duration(event['duration_s'])} of audio · model {event['model']} · "
                      f"{event.get('quality', '')} · {event['threads']} threads · "
                      f"estimated {human_duration(event['estimate_s'])}"
                      f"{' · offline' if event.get('offline') else ''}", file=sys.stderr)
            elif kind == "warning":
                self._end_line()
                print(f"  Warning: {event.get('message')} {event.get('hint', '')}", file=sys.stderr)
            sys.stderr.flush()

    def _end_line(self):
        if self.width:
            sys.stderr.write("\n")
            self.width = 0


def build_parser(cfg: dict) -> argparse.ArgumentParser:
    from .pipeline import DEFAULT_MODEL, DEFAULT_QUALITY, MODELS, QUALITY

    p = argparse.ArgumentParser(
        prog="transcribe",
        description=f"{APP_NAME}: local WhisperX transcription with speaker labels. "
                    "Writes <name>.transcript.txt/.srt/.vtt next to each file.")
    p.add_argument("files", nargs="*", help="audio or video files")
    p.add_argument("--model", choices=MODELS, default=cfg["model"] if cfg["model"] in MODELS else DEFAULT_MODEL,
                   help="speech model (default: %(default)s)")
    p.add_argument("--quality", choices=QUALITY,
                   default=cfg.get("quality") if cfg.get("quality") in QUALITY else DEFAULT_QUALITY,
                   help="speed/accuracy preset (default: %(default)s)")
    p.add_argument("--beam-size", type=int, default=None, help="override the preset's beam size")
    p.add_argument("--output-dir", help="write transcripts here instead of next to each file")
    p.add_argument("--no-diarize", action="store_true", help="skip speaker labels")
    p.add_argument("--subtitles", action=argparse.BooleanOptionalAction, default=bool(cfg.get("subtitles", True)),
                   help="also write .srt and .vtt next to the .txt (default: %(default)s)")
    p.add_argument("--speakers", type=int, help="exact number of speakers, if known")
    p.add_argument("--min-speakers", type=int)
    p.add_argument("--max-speakers", type=int)
    p.add_argument("--language", default=None,
                   help="language code for multilingual models, or 'auto' to detect (default: en)")
    p.add_argument("--hf-token", help="Hugging Face token (default: from config.json or HF_TOKEN)")
    p.add_argument("--threads", type=int, default=None, help="CPU threads (default: all cores but two)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--models-dir", help="where models are stored (default: ./models)")
    p.add_argument("--prepare", action="store_true",
                   help="download the models for these settings (for offline use), then exit")
    p.add_argument("--jsonl", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    return p


def _token(cfg: dict, explicit: str | None = None) -> str:
    return (explicit or cfg.get("hf_token") or os.environ.get("HF_TOKEN", "")).strip()


def serve(emit, session) -> int:
    """Warm-worker mode for the GUI: JSON commands on stdin, events (tagged with the command id) on stdout."""
    from . import pipeline

    for raw in sys.stdin.buffer:
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        try:
            command = json.loads(line)
        except ValueError:
            continue
        kind, ident = command.get("cmd"), command.get("id")
        if kind == "quit":
            return 0

        def tagged(event: dict, ident=ident) -> None:
            emit({**event, "id": ident})

        cfg = config.load()  # a token saved in Settings after start-up applies at once
        try:
            language = command.get("language") or cfg.get("language") or "en"
            job = pipeline.Job(
                source=Path(command.get("file") or ""),
                model=command.get("model") or cfg["model"],
                output_dir=Path(command["output_dir"]) if command.get("output_dir") else None,
                diarize=bool(command.get("diarize", True)),
                hf_token=_token(cfg),
                num_speakers=command.get("speakers"),
                language=None if language == "auto" else language,
                models_dir=session.models_dir, threads=session.threads,
                quality=command.get("quality") or cfg.get("quality") or pipeline.DEFAULT_QUALITY,
                subtitles=bool(command.get("subtitles", cfg.get("subtitles", True))),
            )
            if kind == "warm":
                tagged({"event": "warm_done", **session.warm(job, tagged)})
            elif kind == "prepare":
                tagged({"event": "done", **session.prepare(job, tagged)})
            elif kind == "job":
                tagged({"event": "done", **session.run(job, tagged)})
            else:
                tagged({"event": "error", "code": "BAD_COMMAND", "message": f"Unknown command {kind!r}",
                        "hint": "", "detail": ""})
        except Exception as exc:
            tagged(classify(exc).to_event())
    return 0


def main(argv: list[str] | None = None) -> int:
    cfg = config.load()
    parser = build_parser(cfg)
    args = parser.parse_args(argv)
    if not args.files and not (args.prepare or args.serve):
        parser.error("give at least one audio or video file (or --prepare)")

    for stream in (sys.stdout, sys.stderr):
        try:
            # Pipes on Windows default to cp1252, which mangles quotes and transcript text.
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    emit = _jsonl_emitter() if args.jsonl else ConsoleReporter()

    from . import pipeline

    models_dir = Path(args.models_dir) if args.models_dir else config.models_dir(cfg)
    threads = args.threads if args.threads is not None else int(cfg.get("threads") or 0)
    session = pipeline.Session(models_dir, threads)
    if args.serve:
        return serve(emit, session)

    language = args.language or cfg.get("language") or "en"
    base = dict(
        model=args.model, quality=args.quality, beam_size=args.beam_size, subtitles=args.subtitles,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        diarize=not args.no_diarize, hf_token=_token(cfg, args.hf_token),
        num_speakers=args.speakers, min_speakers=args.min_speakers, max_speakers=args.max_speakers,
        language=None if language == "auto" else language,
        models_dir=models_dir, threads=threads, batch_size=args.batch_size,
    )

    def fail(exc: BaseException) -> None:
        err = classify(exc) if not isinstance(exc, PipelineError) else exc
        if args.jsonl:
            emit(err.to_event())
        else:
            print(f"\n  Error: {err.message}", file=sys.stderr)
            if err.hint:
                print(f"  {err.hint}", file=sys.stderr)
            if err.detail and os.environ.get("RECPIPE_DEBUG"):
                print(err.detail, file=sys.stderr)

    if args.prepare:
        try:
            summary = session.prepare(pipeline.Job(source=Path(), **base), emit)
        except Exception as exc:
            fail(exc)
            return 1
        if args.jsonl:
            emit({"event": "done", **summary})
        else:
            got = ", ".join(summary["downloaded"]) or "nothing new (already downloaded)"
            print(f"\n  Ready for offline use. Downloaded: {got}", file=sys.stderr)
        return 0

    failures = 0
    for number, name in enumerate(args.files, 1):
        if not args.jsonl:
            print(f"[{number}/{len(args.files)}] {name}", file=sys.stderr)
        try:
            summary = session.run(pipeline.Job(source=Path(name), **base), emit)
        except KeyboardInterrupt:
            print("\n  Cancelled.", file=sys.stderr)
            return 130
        except Exception as exc:
            fail(exc)
            failures += 1
            continue
        if args.jsonl:
            emit({"event": "done", **summary})
        else:
            emit({"event": "stage", "label": ""})  # close the progress line
            speakers = f"{summary['speakers']} speakers" if summary["diarized"] else "no speaker labels"
            print(f"  Done in {summary['elapsed_s']:.0f} s ({speakers}):", file=sys.stderr)
            for kind in ("txt", "srt", "vtt"):
                if kind in summary:
                    print(f"    {summary[kind]}", file=sys.stderr)
    return 1 if failures else 0
