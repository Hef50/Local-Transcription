# Local Transcriber

Drop an audio or video file in, get a transcript with speaker labels and timestamps
(`.txt`, `.srt`, `.vtt`). Everything runs on your CPU with
[WhisperX](https://github.com/m-bain/whisperX) (faster-whisper, int8 + pyannote
diarization). No cloud services, no paid API keys, no per-use limits.

## Quick start

1. **Prerequisites:** 64-bit Python 3.10–3.13 and ffmpeg (`winget install Gyan.FFmpeg`).
2. **Double-click `run.bat`.** The first launch runs `setup.ps1`, which creates `.venv`
   in this folder and installs the pinned packages (about 1 GB to download, about 5 GB on disk).
   `create_shortcut.ps1` makes a shortcut you can pin to the taskbar. It carries the app's taskbar
   identity, so the running window shows up on the pinned icon rather than as a separate Python button.
3. **Welcome dialog:** paste a Hugging Face token to turn on speaker labels, or skip.
   1. Create a free token with *Read* access: https://huggingface.co/settings/tokens
   2. Accept the model conditions: https://huggingface.co/pyannote/speaker-diarization-community-1
4. **Drop a file.** The first job with each model downloads that model once. After that,
   everything works with no internet connection.

| Download (one time) | Size |
|---|---|
| small.en / medium.en / large-v3 speech model | 0.5 / 1.5 / 3.1 GB |
| Word-timing model (wav2vec2) | 360 MB |
| Speaker model (pyannote community-1) + sentence data | 33 MB |

Every model is stored in `models\` inside this folder. The HF, torch and NLTK caches
are all redirected there, so nothing goes to `C:\Users\...\.cache`.

## Using it on another computer

```bat
git clone https://github.com/Hef50/Local-Transcription.git
cd Local-Transcription
run.bat
```

`run.bat` creates `.venv`, installs the pinned packages and opens the app. Each machine
needs ffmpeg (`winget install Gyan.FFmpeg`) and downloads its own models on first use
(about 2 GB for medium.en plus the timing and speaker models) into its own `models\` folder.
Not in the repo, by design: `config.json` (holds your Hugging Face token), `calibration.json`
(per-machine speed measurements), `models\`, `.venv\` and `.cache\`. Paste the token again
in the welcome dialog on the new machine. If `setup.ps1` is blocked by PowerShell's execution
policy, run it as `powershell -ExecutionPolicy Bypass -File setup.ps1`.

## The window

Windows 11 look via the [Sun Valley ttk theme](https://github.com/rdbende/Sun-Valley-ttk-theme): follows
your Windows light/dark setting (or pick one in Settings → Appearance), Fluent icons, dark title bar.

* **Model / Speakers / Speed** sit at the top with a live *estimated time per hour of audio*, computed
  from the speeds this PC has actually measured.
* **Drop zone:** large when the window is empty, a slim bar while a file is working or a transcript is shown.
* **Progress card:** both parallel stages, percent, elapsed, time left, Cancel.
* **Transcript:** speaker names in colour, timestamps, Open / Subtitles / Show in folder / Copy.
* **Name speakers:** after a transcript is done, give each voice a real name. Each voice is listed with
  a quote, how long they spoke and a **Play** button to hear them. The names replace "Speaker 1",
  "Speaker 2"... in the .txt, .srt, .vtt and the window. Giving two voices the same name merges them
  (useful when one person was split in two).
* **Older transcripts:** drop a `*.transcript.txt` on the window (or pick it with Ctrl+O) to open it
  and name its speakers. Play works when the original recording is still next to it.
* **Status bar:** whether the speech engine is loaded and whether the next file can run fully offline.
* `Ctrl+O` opens the file picker.

## Command line

```bat
.venv\Scripts\python.exe transcribe.py meeting.m4a
.venv\Scripts\python.exe transcribe.py a.mp3 b.mp4 --model small.en --speakers 3
.venv\Scripts\python.exe transcribe.py interview.wav --no-diarize --output-dir D:\Transcripts
.venv\Scripts\python.exe transcribe.py lecture.mp4 --quality fast
.venv\Scripts\python.exe transcribe.py notes.m4a --no-diarize --no-subtitles   &:: .txt only, fastest
.venv\Scripts\python.exe transcribe.py --prepare --model medium.en   &:: download models for offline use
```

The CLI and the app share `config.json` (token, default model, quality, output folder).
Several files in one command share the loaded models, so only the first pays the start-up cost.

## Output

`meeting.mp4` produces `meeting.transcript.txt`, `.srt` and `.vtt` next to the file, or in
the output folder set in Settings:

```text
Transcript of meeting.mp4
42 min 10 s of audio · model medium.en · 2 speakers · 2026-10-13 16:12

Speaker 1 [00:00:00 - 00:00:08]
Good morning. Thanks for joining the planning call. I want to review the budget...

Speaker 2 [00:00:09 - 00:00:17]
Morning. I looked at the numbers last night...
```

Subtitle cues are at most 2 lines × 42 characters and about 6 seconds each, prefixed with `[Speaker N]`.

**Only need the .txt?** Turn off Settings → Transcript files → *Also save subtitles*. On its own that
saves nothing measurable (writing files takes ~0.03 s). But with speaker labels also off, the app
skips the word-timing (alignment) step, which only subtitles and speaker turns need: see the Speed
section. For pasting into Claude or another AI, the .txt is the best format anyway: about half the
tokens of the .srt/.vtt, with each speaker's turn kept as one paragraph.

## Speed

Measured on this laptop (i7-1355U, CPU only, int8, 10 threads) with a 5-minute two-speaker
recording. Numbers are processing time as a multiple of audio length; lower is faster.

| Stage | Library defaults | Now: Balanced | Now: Fastest | What changed |
|---|---|---|---|---|
| Speech detection (VAD) | 0.048× | 0.009× | 0.009× | window step 0.1 → 0.5: 5× faster, identical chunks |
| Transcription (medium.en) | 0.44–0.55× | same | ~0.42× | beam 5 kept; beam 1 only saves ~5% |
| Word alignment | 0.16× | 0.16× | 0.145× | int8 is 9% faster, same timings (median shift 0 ms) |
| Speaker diarization | 0.82× | 0.27× | 0.13× | window step 0.1 → 0.25 (3×, 99.1% same labels) / 0.5 (6×, 98.3%) |
| Word alignment, for a .txt without speaker labels | 0.16× | skipped | skipped | only subtitles and speaker turns need it: 190 s → 145 s on the 5-min file (24% faster), same words |
| Diarization scheduling | after transcription | **in parallel** | in parallel | 1.27× on the pair; the ETA tracks both |
| Per-file start-up | ~30 s imports + ~8 s model loads | **~0 s** | ~0 s | one warm worker process keeps models loaded |
| **5-minute file, with speakers** | **~9.7 min** | **5.2 min** (4.7 min after the first file) | | measured end to end |

Pick the preset from the **Speed** menu on the main window. "Most accurate" keeps pyannote's default
diarization window (0.1) and costs about 3× more diarization time than Balanced.

The app refines its ETA while it runs and stores measured speeds in `calibration.json`
after every job, so estimates match your machine after the first run with each setting.

**Things that were tried and did not help on this CPU:** `distil-large-v3` (0.48× vs
medium.en's 0.44×: its bigger encoder costs more than its 2-layer decoder saves on CPU),
6 threads instead of 10 (0.56× vs 0.42×), batch size 16 (no change).

**Thermal throttling is real on this laptop.** Cool, it bursts to ~3.1 GHz; after a few
minutes of sustained load it settles at ~2.0 GHz (121% of base), so the same job runs up
to 25% slower when the machine is already warm. Windows is already on *Best performance*
with the processor allowed 100%; the limit is the chip's own 15 W budget. Keep the vents
clear or use a cooling pad; nothing in software changes this.

## Privacy

* Audio and transcripts never leave the machine.
* **pyannote.audio 4.x sends usage telemetry by default** (OpenTelemetry to
  `otel.pyannote.ai`, including audio duration). The app turns it off with
  `PYANNOTE_METRICS_ENABLED=false` before pyannote is imported, and disables Hugging Face telemetry too.
* Once the models for a job are downloaded, runs use `HF_HUB_OFFLINE=1` and make no network
  requests (verified with every socket and DNS call blocked). The warm worker goes online
  only for a first-time model download, then turns offline mode back on.
* The Hugging Face token is stored in plain text in `config.json` in this folder. It is sent
  only to huggingface.co, only for the speaker-model download
  (`HF_HUB_DISABLE_IMPLICIT_TOKEN=1` keeps it off every other request). Once that model is
  downloaded, the token is no longer needed.

## How it works

```text
run.bat -> app.py (Tkinter + tkinterdnd2 window)
             |  starts ONE worker at launch:  app.py --worker --jsonl --serve
             |  (imports torch/whisperx, loads the current models; stays alive between jobs)
             |  per dropped file, a JSON command on the worker's stdin:
             |     check file -> ffprobe -> decode (ffmpeg) -> [download missing models]
             |     main thread:   faster-whisper int8 (VAD + batched ASR) -> wav2vec2 alignment
             |     second thread: pyannote diarization                     (both ~ at once)
             |     -> assign speakers -> write .txt/.srt/.vtt
             '-- JSON progress events on stdout -> progress bar, ETA, errors, result view
```

* **Why Tkinter:** WhisperX is Python and Tk ships with Python, so there's no Node or Rust toolchain,
  and it packages cleanly with PyInstaller. PyQt adds ~100 MB and LGPL obligations, and
  Electron/Tauri add a second runtime and IPC for no benefit in a single-window tool.
* **Why a separate process:** the window stays responsive during long CPU work, Cancel just
  kills the worker, and a crash in native code (torch, CTranslate2) can't take the window down.
* **Why keep it alive:** importing the speech libraries took ~30 s on this machine and loading
  medium.en ~8 s. Doing that once per app session instead of once per file makes a dropped
  file start transcribing immediately.

| File | Purpose |
|---|---|
| `app.py` | GUI entry; `--worker` runs the pipeline headless (`--serve` keeps it alive) |
| `transcribe.py` | CLI entry |
| `recpipe/pipeline.py` | WhisperX stages, presets, downloads, parallel diarization, progress/ETA |
| `recpipe/formats.py` | .txt / .srt / .vtt rendering (pure Python, unit-tested) |
| `recpipe/errors.py` | exception -> friendly message + hint |
| `recpipe/gui.py` | window (Sun Valley / Windows 11 theme, light + dark), settings, queue, warm worker |
| `recpipe/env.py` | cache locations, telemetry off, offline mode |
| `recpipe/config.py`, `deps.py` | settings file; first-run dependency checks |
| `tests/` | `.venv\Scripts\python.exe -m unittest discover tests` |

## Troubleshooting

| Message | Fix |
|---|---|
| "Setup needed: missing Python packages" | Run `setup.ps1`, or click **Install** in the app |
| "ffmpeg wasn't found" | `winget install Gyan.FFmpeg`, then restart the app |
| "Speaker labels need a Hugging Face token" | Settings → paste token (see Quick start) |
| "Hugging Face rejected your access token" | Copy the whole token, or make a new *Read* token |
| "…doesn't have access to the speaker model" | Accept the conditions on the model page with the same HF account |
| "…isn't downloaded yet, and the internet isn't reachable" | Connect once, or use Settings → **Download models now** |
| "ran out of memory" | Use small.en and close heavy apps |
| `torchcodec is not installed correctly` in the log | Harmless: audio goes to pyannote in memory, already decoded by ffmpeg |

Set `RECPIPE_DEBUG=1` to print full tracebacks in the CLI. The app has a **Details** button.

## Packaging as a standalone .exe (next step, not yet built)

The code already supports being frozen: when `sys.frozen` is set, the app launches
itself with `--worker` and uses the .exe's folder for `config.json` and `models\`. Use a
PyInstaller **one-folder** build. A one-file build would unpack ~2 GB of torch on every launch.

```bat
uv pip install --python .venv\Scripts\python.exe pyinstaller
.venv\Scripts\pyinstaller.exe --noconfirm --windowed --name LocalTranscriber ^
  --collect-all whisperx --collect-all pyannote --collect-all faster_whisper ^
  --collect-all lightning_fabric --collect-all pytorch_lightning --collect-all asteroid_filterbanks ^
  --collect-data tkinterdnd2 --collect-data nltk --add-data "assets;assets" app.py
```

Expect a few rounds of adding `--hidden-import` / `--collect-*` flags for whatever fails to
import at runtime. Put `ffmpeg.exe` and `ffprobe.exe` next to `LocalTranscriber.exe` (or keep
them on PATH).
