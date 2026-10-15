"""Run with:  .venv\\Scripts\\python.exe -m unittest discover tests"""
import tempfile
import unittest
from pathlib import Path

from recpipe import formats
from recpipe.errors import classify


def word(text, start, end, speaker=None):
    w = {"word": text}
    if start is not None:
        w["start"] = start
    if end is not None:
        w["end"] = end
    if speaker:
        w["speaker"] = speaker
    return w


RESULT = {
    "segments": [
        {"start": 0.5, "end": 2.0, "text": " Hello there.", "speaker": "SPEAKER_01",
         "words": [word("Hello", 0.5, 0.9, "SPEAKER_01"), word("there.", 1.0, 2.0, "SPEAKER_01")]},
        {"start": 2.2, "end": 4.0, "text": " How are you?", "speaker": "SPEAKER_01",
         "words": [word("How", 2.2, 2.5), word("are", None, None), word("you?", 3.0, 4.0)]},
        {"start": 4.5, "end": 6.0, "text": " Fine, thanks.", "speaker": "SPEAKER_00",
         "words": [word("Fine,", 4.5, 5.0), word("thanks.", 5.1, 6.0)]},
    ]
}


class FormatTests(unittest.TestCase):
    def test_speakers_numbered_by_first_appearance(self):
        segments = formats.normalize(RESULT, label_speakers=True)
        self.assertEqual([s.speaker for s in segments], ["Speaker 1", "Speaker 1", "Speaker 2"])

    def test_missing_word_times_are_filled_from_neighbours(self):
        seg = formats.normalize(RESULT, label_speakers=True)[1]
        self.assertEqual((seg.words[1].start, seg.words[1].end), (2.5, 3.0))

    def test_txt_merges_same_speaker_turns(self):
        txt = formats.render_txt(formats.normalize(RESULT, True), ["Header"])
        self.assertEqual(txt, "Header\n\nSpeaker 1 [00:00 - 00:04]\nHello there. How are you?\n\n"
                              "Speaker 2 [00:04 - 00:06]\nFine, thanks.\n")

    def test_txt_without_labels(self):
        txt = formats.render_txt(formats.normalize(RESULT, False))
        self.assertTrue(txt.startswith("[00:00 - 00:06]\nHello there. How are you? Fine, thanks."))
        self.assertNotIn("Speaker", txt)

    def test_long_monologue_gets_new_timestamps(self):
        segs = [{"start": t, "end": t + 9, "text": "words " * 5, "speaker": "S"} for t in range(0, 300, 10)]
        turns = formats.build_turns(formats.normalize({"segments": segs}, True))
        self.assertGreater(len(turns), 3)
        self.assertTrue(all(t.end - t.start <= formats.TURN_MAX_SECONDS for t in turns))

    def test_unlabelled_text_is_split_into_shorter_paragraphs(self):
        segs = [{"start": t, "end": t + 4, "text": "words"} for t in range(0, 120, 5)]
        turns = formats.build_turns(formats.normalize({"segments": segs}, False))
        self.assertGreaterEqual(len(turns), 4)
        self.assertTrue(all(t.end - t.start <= formats.UNLABELLED_TURN_MAX_SECONDS for t in turns))

    def test_hours_in_timestamps_for_long_audio(self):
        segs = formats.normalize({"segments": [{"start": 3725, "end": 3730, "text": "late"}]}, False)
        self.assertIn("[01:02:05 - 01:02:10]", formats.render_txt(segs))

    def test_srt_and_vtt(self):
        cues = formats.build_cues(formats.normalize(RESULT, True))
        srt = formats.render_srt(cues)
        self.assertTrue(srt.startswith("1\n00:00:00,500 --> 00:00:02,000\n[Speaker 1] Hello there.\n"))
        vtt = formats.render_vtt(cues)
        self.assertTrue(vtt.startswith("WEBVTT\n\n00:00:00.500 --> 00:00:02.000\n[Speaker 1] Hello there.\n"))
        for cue, nxt in zip(cues, cues[1:]):
            self.assertLessEqual(cue.end, nxt.start)

    def test_long_segment_is_split_into_short_cues(self):
        words = [word(f"word{i}", i * 0.4, i * 0.4 + 0.3) for i in range(60)]
        seg = {"start": 0, "end": 24, "text": " ".join(w["word"] for w in words), "words": words}
        cues = formats.build_cues(formats.normalize({"segments": [seg]}, False))
        self.assertGreater(len(cues), 4)
        for cue in cues:
            self.assertLessEqual(len(cue.text), formats.SUBTITLE_MAX_CHARS)
            self.assertLessEqual(cue.end - cue.start, formats.SUBTITLE_MAX_SECONDS + 0.5)
            self.assertTrue(all(len(line) <= formats.SUBTITLE_LINE_WIDTH + 12
                                for line in formats.wrap(cue.text).splitlines()))

    def test_segment_without_words_still_produces_cues(self):
        seg = {"start": 10, "end": 20, "text": "alpha " * 40}
        cues = formats.build_cues(formats.normalize({"segments": [seg]}, False))
        self.assertGreater(len(cues), 1)
        self.assertAlmostEqual(cues[0].start, 10)
        self.assertAlmostEqual(cues[-1].end, 20, places=3)

    def test_write_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, speakers = formats.write_outputs(RESULT, Path("C:/x/meeting.mp4"), Path(tmp),
                                                    model="small.en", duration=6.0, label_speakers=True)
            self.assertEqual(speakers, 2)
            self.assertEqual(paths["txt"].name, "meeting.transcript.txt")
            self.assertIn("Speaker 2", paths["txt"].read_text(encoding="utf-8"))
            self.assertFalse(list(Path(tmp).glob("*.tmp")))


class ErrorTests(unittest.TestCase):
    def test_gated_repo(self):
        gated = type("GatedRepoError", (Exception,), {})
        self.assertEqual(classify(gated("403 Client Error"), "download").code, "HF_GATED")

    def test_gated_repo_with_bad_token_is_a_token_problem(self):
        gated = type("GatedRepoError", (Exception,), {})
        self.assertEqual(classify(gated("401 Client Error. Cannot access gated repo"), "download").code,
                         "HF_TOKEN_INVALID")

    def test_invalid_token(self):
        self.assertEqual(classify(RuntimeError("401 Client Error: Unauthorized"), "download").code,
                         "HF_TOKEN_INVALID")

    def test_offline(self):
        missing = type("LocalEntryNotFoundError", (Exception,), {})
        self.assertEqual(classify(missing("not cached"), "load_asr").code, "MODEL_NOT_AVAILABLE")

    def test_memory(self):
        self.assertEqual(classify(RuntimeError("DefaultCPUAllocator: not enough memory")).code,
                         "OUT_OF_MEMORY")

    def test_unknown_mentions_stage(self):
        err = classify(ValueError("boom"), "align")
        self.assertEqual(err.code, "UNEXPECTED")
        self.assertIn("aligning words", err.message)


if __name__ == "__main__":
    unittest.main()


class PresetTests(unittest.TestCase):
    def setUp(self):
        from recpipe import pipeline
        self.pipeline = pipeline

    def test_preset_with_override(self):
        job = self.pipeline.Job(source=Path("x.wav"), quality="fast", diarize_step=0.33)
        settings = job.settings()
        self.assertEqual(settings["beam_size"], 1)
        self.assertEqual(settings["diarize_step"], 0.33)

    def test_unknown_quality_uses_default_preset(self):
        job = self.pipeline.Job(source=Path("x.wav"), quality="nonsense")
        self.assertEqual(job.settings(), self.pipeline.PRESETS[self.pipeline.DEFAULT_QUALITY])

    def test_rate_fallbacks_point_the_right_way(self):
        rtf = self.pipeline.default_rtf
        self.assertLess(rtf("transcribe:medium.en:b1"), rtf("transcribe:medium.en:b5"))
        self.assertLess(rtf("diarize:s0.5"), rtf("diarize:s0.1"))
        self.assertLess(rtf("align:int8"), rtf("align:fp32"))

    def test_cached_stages_cost_nothing_in_the_plan(self):
        job = self.pipeline.Job(source=Path("x.wav"))
        stages = {s.key: s for s in self.pipeline.plan_stages(job, 60, {}, cached={"prepare", "load_asr"})}
        self.assertTrue(stages["prepare"].cached and stages["prepare"].estimate == 0)
        self.assertTrue(stages["load_asr"].cached and stages["load_asr"].estimate == 0)
        self.assertFalse(stages["transcribe"].cached)
        self.assertGreater(stages["transcribe"].estimate, 0)

    def test_english_only_models_force_english(self):
        Job = self.pipeline.Job
        self.assertEqual(Job(source=Path("x"), model="medium.en", language=None).effective_language, "en")
        self.assertIsNone(Job(source=Path("x"), model="large-v3", language=None).effective_language)


class TxtOnlyAndRenameTests(unittest.TestCase):
    def _write(self, tmp, subtitles=True):
        return formats.write_outputs(RESULT, Path("C:/x/meeting.mp4"), Path(tmp), model="small.en", duration=6.0,
                                     label_speakers=True, subtitles=subtitles)

    def test_txt_only_writes_one_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, _ = self._write(tmp, subtitles=False)
            self.assertEqual(set(paths), {"txt"})
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["meeting.transcript.txt"])

    def test_speakers_in_finds_each_voice_with_a_sample(self):
        txt = formats.render_txt(formats.normalize(RESULT, True), ["Header", "6 s of audio · 2 speakers"])
        speakers = formats.speakers_in(txt)
        self.assertEqual([s.name for s in speakers], ["Speaker 1", "Speaker 2"])
        self.assertEqual(speakers[1].sample, "Fine, thanks.")
        self.assertEqual((speakers[1].sample_start, speakers[1].sample_end), (4.0, 6.0))

    def test_rename_everywhere_and_recount_merged_speakers(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, _ = self._write(tmp)
            changed = formats.rename_outputs(paths, {"Speaker 1": "Alex", "Speaker 2": "Alex"})
            self.assertEqual(len(changed), 3)
            txt = paths["txt"].read_text(encoding="utf-8")
            self.assertIn("Alex [00:00 - 00:04]", txt)
            self.assertNotIn("Speaker", txt.split("\n", 2)[2])
            self.assertIn("1 speaker ·", txt.splitlines()[1])  # header recounted after the merge
            for kind in ("srt", "vtt"):
                body = paths[kind].read_text(encoding="utf-8")
                self.assertIn("[Alex] Hello there.", body)
                self.assertNotIn("[Speaker", body)

    def test_rename_rewraps_long_cues(self):
        srt = "1\n00:00:00,000 --> 00:00:03,000\n[Speaker 1] I want to review the budget for\nthe community garden today.\n"
        out = formats.rename_subtitles(srt, {"Speaker 1": "Alexandra Montgomery"})
        body = out.split("\n")[2:]
        self.assertTrue(body[0].startswith("[Alexandra Montgomery] "))
        self.assertTrue(all(len(line) <= 60 for line in body if line))
        self.assertTrue(out.endswith("\n"))

    def test_rename_leaves_spoken_words_alone(self):
        txt = "Speaker 1 [00:00 - 00:04]\nI am Speaker 1 in this test.\n"
        self.assertEqual(formats.rename_txt(txt, {"Speaker 1": "Sam"}), "Sam [00:00 - 00:04]\nI am Speaker 1 in this test.\n")

    def test_name_problems(self):
        self.assertIsNone(formats.name_problem("Dr. Jane O'Neil"))
        self.assertIsNotNone(formats.name_problem("[Host]"))
        self.assertIsNotNone(formats.name_problem("x" * 61))

    def test_transcript_siblings(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "meeting.mp4"
            source.write_bytes(b"fake")
            paths, _ = formats.write_outputs(RESULT, source, Path(tmp), model="small.en", duration=6.0,
                                             label_speakers=True)
            found = formats.transcript_siblings(paths["txt"])
            self.assertEqual(set(found), {"txt", "srt", "vtt", "source"})
            self.assertEqual(found["source"], source)


class AlignmentPlanTests(unittest.TestCase):
    def test_alignment_skipped_only_when_nothing_needs_it(self):
        from recpipe import pipeline
        def stage_keys(**kw):
            return {s.key for s in pipeline.plan_stages(pipeline.Job(source=Path("x"), **kw), 60, {})}
        self.assertNotIn("align", stage_keys(diarize=False, subtitles=False))
        self.assertIn("align", stage_keys(diarize=False, subtitles=True))
        self.assertIn("align", stage_keys(diarize=True, subtitles=False))
