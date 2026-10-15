"""WhisperX result -> .txt / .srt / .vtt. Pure Python: no ML imports, easy to test."""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

SUBTITLE_MAX_CHARS = 84  # two lines of 42, the usual subtitle limit
SUBTITLE_LINE_WIDTH = 42
SUBTITLE_MAX_SECONDS = 6.0
SUBTITLE_MIN_SECONDS = 0.7
TURN_MAX_SECONDS = 90.0  # a long monologue still gets a fresh timestamp this often
UNLABELLED_TURN_MAX_SECONDS = 30.0  # no speaker changes to break on: keep paragraphs readable
TURN_MAX_GAP = 4.0  # a pause this long starts a new paragraph even for the same speaker
UNKNOWN_SPEAKER = "Unknown speaker"


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None
    words: list[Word] = field(default_factory=list)


@dataclass
class Turn:
    start: float
    end: float
    speaker: str | None
    texts: list[str]


@dataclass
class Cue:
    start: float
    end: float
    speaker: str | None
    text: str


def _num(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def speaker_names(raw_segments: list[dict]) -> dict[str, str]:
    """Map SPEAKER_00-style labels to "Speaker 1", "Speaker 2"... by first appearance."""
    names: dict[str, str] = {}
    for seg in raw_segments:
        for label in [seg.get("speaker")] + [w.get("speaker") for w in seg.get("words") or []]:
            if label and label not in names:
                names[label] = f"Speaker {len(names) + 1}"
    return names


def _words(raw_words: list[dict], seg_start: float, seg_end: float) -> list[Word]:
    """Alignment can leave words without times (e.g. numbers); borrow neighbours' times."""
    starts = [_num(w.get("start")) for w in raw_words]
    ends = [_num(w.get("end")) for w in raw_words]
    words, last_end = [], seg_start
    for i, raw in enumerate(raw_words):
        text = (raw.get("word") or "").strip()
        if not text:
            continue
        start = starts[i] if starts[i] is not None else last_end
        end = ends[i]
        if end is None:
            end = next((s for s in starts[i + 1:] if s is not None), seg_end)
        end = max(end, start)
        words.append(Word(text, start, end))
        last_end = end
    return words


def normalize(result: dict, label_speakers: bool) -> list[Segment]:
    raw_segments = result.get("segments") or []
    names = speaker_names(raw_segments) if label_speakers else {}
    segments: list[Segment] = []
    prev_end, prev_speaker = 0.0, None
    for raw in raw_segments:
        text = (raw.get("text") or "").strip()
        if not text:
            continue
        raw_words = raw.get("words") or []
        word_starts = [s for s in (_num(w.get("start")) for w in raw_words) if s is not None]
        word_ends = [e for e in (_num(w.get("end")) for w in raw_words) if e is not None]
        start = _num(raw.get("start"))
        if start is None:
            start = word_starts[0] if word_starts else prev_end
        end = _num(raw.get("end"))
        if end is None:
            end = word_ends[-1] if word_ends else start
        end = max(end, start)

        speaker = None
        if label_speakers:
            speaker = names.get(raw.get("speaker"))
            if speaker is None:
                votes = Counter(names[w["speaker"]] for w in raw_words if w.get("speaker") in names)
                speaker = votes.most_common(1)[0][0] if votes else (prev_speaker or UNKNOWN_SPEAKER)
            prev_speaker = speaker

        segments.append(Segment(start, end, text, speaker, _words(raw_words, start, end)))
        prev_end = end
    return segments


# ---------------------------------------------------------------- time formats

def clock(seconds: float, hours: bool) -> str:
    total = max(int(seconds), 0)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if hours else f"{m:02d}:{s:02d}"


def _subtitle_time(seconds: float, separator: str) -> str:
    ms = max(round(seconds * 1000), 0)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{separator}{ms:03d}"


def human_duration(seconds: float) -> str:
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds} s"
    minutes, s = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {s:02d} s"
    h, m = divmod(minutes, 60)
    return f"{h} h {m:02d} min"


# ---------------------------------------------------------------- plain text

def build_turns(segments: list[Segment]) -> list[Turn]:
    turns: list[Turn] = []
    for seg in segments:
        last = turns[-1] if turns else None
        limit = TURN_MAX_SECONDS if seg.speaker else UNLABELLED_TURN_MAX_SECONDS
        if (last and seg.speaker == last.speaker and seg.start - last.end <= TURN_MAX_GAP
                and seg.end - last.start <= limit):
            last.end = max(last.end, seg.end)
            last.texts.append(seg.text)
        else:
            turns.append(Turn(seg.start, seg.end, seg.speaker, [seg.text]))
    return turns


def render_txt(segments: list[Segment], header_lines: list[str] = ()) -> str:
    hours = bool(segments) and max(s.end for s in segments) >= 3600
    lines = [*header_lines, ""] if header_lines else []
    for turn in build_turns(segments):
        stamp = f"[{clock(turn.start, hours)} - {clock(turn.end, hours)}]"
        lines.append(f"{turn.speaker} {stamp}" if turn.speaker else stamp)
        lines.append(" ".join(turn.texts))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------- subtitles

def _label(speaker: str | None) -> str:
    return f"[{speaker}] " if speaker else ""


def _cues_from_words(seg: Segment) -> list[Cue]:
    limit = SUBTITLE_MAX_CHARS - len(_label(seg.speaker))
    cues: list[Cue] = []
    buf: list[Word] = []
    for word in seg.words:
        if buf:
            length = sum(len(w.text) + 1 for w in buf) + len(word.text)
            too_long = length > limit or word.end - buf[0].start > SUBTITLE_MAX_SECONDS
            sentence_done = buf[-1].text[-1] in ".?!" and buf[-1].end - buf[0].start >= 2.0
            if too_long or sentence_done:
                cues.append(Cue(buf[0].start, buf[-1].end, seg.speaker, " ".join(w.text for w in buf)))
                buf = []
        buf.append(word)
    if buf:
        cues.append(Cue(buf[0].start, buf[-1].end, seg.speaker, " ".join(w.text for w in buf)))
    return cues


def _cues_from_text(seg: Segment) -> list[Cue]:
    """No word timings: split the text into chunks and share the time out by length."""
    limit = SUBTITLE_MAX_CHARS - len(_label(seg.speaker))
    chunks, current = [], []
    for token in seg.text.split():
        if current and len(" ".join(current + [token])) > limit:
            chunks.append(" ".join(current))
            current = []
        current.append(token)
    if current:
        chunks.append(" ".join(current))
    total_chars = sum(len(c) for c in chunks) or 1
    cues, t = [], seg.start
    for chunk in chunks:
        duration = (seg.end - seg.start) * len(chunk) / total_chars
        cues.append(Cue(t, t + duration, seg.speaker, chunk))
        t += duration
    return cues


def build_cues(segments: list[Segment]) -> list[Cue]:
    cues: list[Cue] = []
    for seg in segments:
        cues.extend(_cues_from_words(seg) if seg.words else _cues_from_text(seg))
    cues.sort(key=lambda c: c.start)
    for cue, following in zip(cues, cues[1:] + [None]):
        cue.end = max(cue.end, cue.start + SUBTITLE_MIN_SECONDS)
        if following is not None and following.start > cue.start:
            cue.end = min(cue.end, following.start)
    return cues


def wrap(text: str, width: int = SUBTITLE_LINE_WIDTH) -> str:
    """Break a cue into two lines at the space closest to the middle."""
    if len(text) <= width:
        return text
    middle = len(text) // 2
    spaces = [i for i in (text.rfind(" ", 0, middle + 1), text.find(" ", middle)) if i > 0]
    if not spaces:
        return text
    cut = min(spaces, key=lambda i: abs(i - middle))
    return text[:cut] + "\n" + text[cut + 1:]


def render_srt(cues: list[Cue]) -> str:
    blocks = [
        f"{n}\n{_subtitle_time(c.start, ',')} --> {_subtitle_time(c.end, ',')}\n{wrap(_label(c.speaker) + c.text)}\n"
        for n, c in enumerate(cues, 1)
    ]
    return "\n".join(blocks)


def render_vtt(cues: list[Cue]) -> str:
    blocks = [
        f"{_subtitle_time(c.start, '.')} --> {_subtitle_time(c.end, '.')}\n{wrap(_label(c.speaker) + c.text)}\n"
        for c in cues
    ]
    return "WEBVTT\n\n" + "\n".join(blocks)


# ---------------------------------------------------------------- files

def output_paths(source: Path, out_dir: Path) -> dict[str, Path]:
    # The ".transcript" infix avoids clobbering an unrelated notes.txt or movie.srt.
    stem = Path(source).stem
    return {kind: Path(out_dir) / f"{stem}.transcript.{kind}" for kind in ("txt", "srt", "vtt")}


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_outputs(result: dict, source: Path, out_dir: Path, *, model: str, duration: float,
                  label_speakers: bool, subtitles: bool = True) -> tuple[dict[str, Path], int]:
    segments = normalize(result, label_speakers)
    speakers = len({s.speaker for s in segments if s.speaker and s.speaker != UNKNOWN_SPEAKER})
    who = (f"{speakers} speaker{'s' if speakers != 1 else ''}" if label_speakers
           else "speaker labels off")
    header = [
        f"Transcript of {Path(source).name}",
        f"{human_duration(duration)} of audio · model {model} · {who} · "
        f"{datetime.now():%Y-%m-%d %H:%M}",
    ]
    contents = {"txt": render_txt(segments, header)}
    if subtitles:
        cues = build_cues(segments)
        contents.update(srt=render_srt(cues), vtt=render_vtt(cues))
    paths = {kind: path for kind, path in output_paths(source, out_dir).items() if kind in contents}
    for kind, path in paths.items():
        _write_atomic(path, contents[kind])
    return paths, speakers


# ---------------------------------------------------------------- naming speakers afterwards

SPEAKER_HEADING_RE = re.compile(
    r"^(?P<name>[^\[\]\n]+?) \[(?P<start>\d{2}:\d{2}(?::\d{2})?) - (?P<end>\d{2}:\d{2}(?::\d{2})?)\]$")
CUE_LABEL_RE = re.compile(r"^\[(?P<name>[^\[\]\n]+)\] ")
SPEAKER_COUNT_RE = re.compile(r"\b\d+ speakers?\b")
MAX_NAME_LENGTH = 60


@dataclass
class SpeakerSummary:
    name: str
    turns: int = 0
    seconds: float = 0.0
    sample: str = ""  # what they said in their longest turn
    sample_start: float = 0.0
    sample_end: float = 0.0


def parse_clock(text: str) -> float:
    seconds = 0
    for part in text.split(":"):
        seconds = seconds * 60 + int(part)
    return float(seconds)


def speakers_in(txt: str) -> list[SpeakerSummary]:
    """Speakers in a .txt transcript, in order of first appearance, with a representative turn each."""
    lines = txt.splitlines()
    found: dict[str, SpeakerSummary] = {}
    for i, line in enumerate(lines):
        match = SPEAKER_HEADING_RE.match(line)
        if not match:
            continue
        start, end = parse_clock(match["start"]), parse_clock(match["end"])
        summary = found.setdefault(match["name"], SpeakerSummary(match["name"]))
        summary.turns += 1
        summary.seconds += end - start
        if not summary.sample or end - start > summary.sample_end - summary.sample_start:
            summary.sample = lines[i + 1].strip() if i + 1 < len(lines) else ""
            summary.sample_start, summary.sample_end = start, end
    return list(found.values())


def name_problem(name: str) -> str | None:
    """Why a typed speaker name can't be used, or None if it's fine."""
    if any(ch in name for ch in "[]\n\r"):
        return "Names can't contain square brackets or line breaks."
    if len(name) > MAX_NAME_LENGTH:
        return f"Names can be at most {MAX_NAME_LENGTH} characters."
    return None


def rename_txt(txt: str, mapping: dict[str, str]) -> str:
    lines = txt.splitlines()
    names = []
    for i, line in enumerate(lines):
        match = SPEAKER_HEADING_RE.match(line)
        if match:
            name = mapping.get(match["name"], match["name"])
            names.append(name)
            lines[i] = f"{name} [{match['start']} - {match['end']}]"
    # Giving two voices the same name merges them, so recount for the header line.
    count = len(set(names))
    for i in range(min(3, len(lines))):
        if SPEAKER_COUNT_RE.search(lines[i]):
            lines[i] = SPEAKER_COUNT_RE.sub(f"{count} speaker{'s' if count != 1 else ''}", lines[i], count=1)
            break
    return "\n".join(lines) + ("\n" if txt.endswith("\n") else "")


def rename_subtitles(text: str, mapping: dict[str, str]) -> str:
    """Rename the [Speaker N] prefix on each .srt/.vtt cue and re-wrap the cue to two lines."""
    blocks = text.rstrip("\n").split("\n\n")
    for b, block in enumerate(blocks):
        lines = block.split("\n")
        timing = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing is None or timing + 1 >= len(lines):
            continue
        body = " ".join(line.strip() for line in lines[timing + 1:] if line.strip())
        match = CUE_LABEL_RE.match(body)
        if match and match["name"] in mapping:
            body = f"[{mapping[match['name']]}] " + body[match.end():]
            blocks[b] = "\n".join(lines[:timing + 1] + wrap(body).split("\n"))
    return "\n\n".join(blocks) + "\n"


def rename_outputs(paths: dict[str, Path], mapping: dict[str, str]) -> list[Path]:
    """Apply new speaker names to every transcript file that exists. Returns the files changed."""
    mapping = {old: new for old, new in mapping.items() if new and new != old}
    changed = []
    for kind, path in paths.items():
        path = Path(path)
        if kind not in ("txt", "srt", "vtt") or not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        updated = rename_txt(text, mapping) if kind == "txt" else rename_subtitles(text, mapping)
        if updated != text:
            _write_atomic(path, updated)
            changed.append(path)
    return changed


def transcript_siblings(txt_path: Path) -> dict[str, Path]:
    """For an existing <name>.transcript.txt: the subtitle files next to it and the original recording."""
    txt_path = Path(txt_path)
    base = txt_path.name[: -len(".transcript.txt")] if txt_path.name.lower().endswith(".transcript.txt") else txt_path.stem
    found = {"txt": txt_path}
    for kind in ("srt", "vtt"):
        candidate = txt_path.with_name(f"{base}.transcript.{kind}")
        if candidate.exists():
            found[kind] = candidate
    try:
        first = txt_path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        first = ""
    if first.startswith("Transcript of "):
        source = txt_path.with_name(first[len("Transcript of "):].strip())
        if source.exists():
            found["source"] = source
    return found
