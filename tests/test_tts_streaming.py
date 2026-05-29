"""Unit tests for Week 3 Day 3 — streaming TTS (synthesize_stream).

The unit boundary is ``tts.load_voice``: patched to return a fake voice whose
``synthesize_wav(text, wav_file)`` writes deterministic int16 frames into the
WAV object it's handed — exactly as the real Piper call does. This isolates
the streaming/buffering logic from Piper, ONNX, and any model files.

Mirrors the existing test_tts architecture (mocked voice, fast unit tests).
One opt-in integration test drives the real voice when --run-integration is
passed.
"""

from __future__ import annotations

import wave
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src import tts


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_fake_voice(sample_rate: int = 22050, samples_per_char: int = 10):
    """Build a fake Piper voice.

    Its synthesize_wav writes ``samples_per_char * len(text)`` int16 frames
    (value = text length, so different sentences produce distinguishable PCM)
    into the wav object the caller passes in.
    """
    voice = MagicMock()
    voice.config = SimpleNamespace(sample_rate=sample_rate)

    def fake_synth(text, wav_file):
        n = max(1, len(text)) * samples_per_char
        frames = np.full(n, len(text), dtype=np.int16)
        wav_file.writeframes(frames.tobytes())

    voice.synthesize_wav.side_effect = fake_synth
    return voice


def _drain(gen):
    """Consume a generator fully; return (yielded_items, return_value)."""
    out = []
    try:
        while True:
            out.append(next(gen))
    except StopIteration as stop:
        return out, (stop.value or {})


# --------------------------------------------------------------------------- #
# OutputSampleRate
# --------------------------------------------------------------------------- #


class TestOutputSampleRate:
    def test_returns_voice_config_rate(self) -> None:
        voice = _make_fake_voice(sample_rate=22050)
        with patch.object(tts, "load_voice", return_value=voice):
            assert tts.output_sample_rate() == 22050

    def test_returns_int(self) -> None:
        voice = _make_fake_voice(sample_rate=16000)
        with patch.object(tts, "load_voice", return_value=voice):
            rate = tts.output_sample_rate()
        assert isinstance(rate, int)
        assert rate == 16000


# --------------------------------------------------------------------------- #
# SynthesizeStream — core behaviour
# --------------------------------------------------------------------------- #


class TestSynthesizeStreamCore:
    def test_one_chunk_per_sentence(self) -> None:
        voice = _make_fake_voice()
        with patch.object(tts, "load_voice", return_value=voice):
            out, _ = _drain(tts.synthesize_stream(["One.", "Two.", "Three."]))
        assert len(out) == 3

    def test_chunks_are_int16_ndarrays(self) -> None:
        voice = _make_fake_voice()
        with patch.object(tts, "load_voice", return_value=voice):
            out, _ = _drain(tts.synthesize_stream(["Hello there."]))
        assert len(out) == 1
        assert isinstance(out[0], np.ndarray)
        assert out[0].dtype == np.int16
        assert out[0].ndim == 1

    def test_chunks_are_non_empty(self) -> None:
        voice = _make_fake_voice()
        with patch.object(tts, "load_voice", return_value=voice):
            out, _ = _drain(tts.synthesize_stream(["A.", "Bee.", "Sea."]))
        assert all(chunk.size > 0 for chunk in out)

    def test_pcm_roundtrips_voice_output(self) -> None:
        # Fake voice writes frames = len(text), so we can verify the exact PCM.
        voice = _make_fake_voice(samples_per_char=4)
        with patch.object(tts, "load_voice", return_value=voice):
            out, _ = _drain(tts.synthesize_stream(["Hi."]))  # len("Hi.")==3
        expected = np.full(3 * 4, 3, dtype=np.int16)
        np.testing.assert_array_equal(out[0], expected)

    def test_sets_wave_params_before_synthesis(self) -> None:
        # The "critical param order" contract: nchannels/sampwidth/framerate
        # must be set on the wav object before synthesize_wav writes frames.
        # We verify synthesize_wav received a wav object already configured.
        captured = {}

        def fake_synth(text, wav_file):
            captured["nchannels"] = wav_file.getnchannels()
            captured["sampwidth"] = wav_file.getsampwidth()
            captured["framerate"] = wav_file.getframerate()
            wav_file.writeframes(np.zeros(10, dtype=np.int16).tobytes())

        voice = MagicMock()
        voice.config = SimpleNamespace(sample_rate=22050)
        voice.synthesize_wav.side_effect = fake_synth
        with patch.object(tts, "load_voice", return_value=voice):
            _drain(tts.synthesize_stream(["Test."]))
        assert captured == {"nchannels": 1, "sampwidth": 2, "framerate": 22050}


# --------------------------------------------------------------------------- #
# SynthesizeStream — input handling
# --------------------------------------------------------------------------- #


class TestSynthesizeStreamInput:
    def test_skips_empty_and_whitespace_sentences(self) -> None:
        voice = _make_fake_voice()
        with patch.object(tts, "load_voice", return_value=voice):
            out, stats = _drain(
                tts.synthesize_stream(["Real.", "", "   ", "\t\n", "Also real."])
            )
        assert len(out) == 2
        assert stats["num_sentences"] == 2

    def test_accepts_a_generator(self) -> None:
        voice = _make_fake_voice()

        def sentence_gen():
            yield "First."
            yield "Second."

        with patch.object(tts, "load_voice", return_value=voice):
            out, _ = _drain(tts.synthesize_stream(sentence_gen()))
        assert len(out) == 2

    def test_empty_iterable_yields_nothing(self) -> None:
        voice = _make_fake_voice()
        with patch.object(tts, "load_voice", return_value=voice):
            out, stats = _drain(tts.synthesize_stream([]))
        assert out == []
        assert stats["num_sentences"] == 0
        assert stats["time_to_first_audio_s"] is None

    def test_strips_sentence_before_synthesis(self) -> None:
        voice = _make_fake_voice(samples_per_char=1)
        with patch.object(tts, "load_voice", return_value=voice):
            out, _ = _drain(tts.synthesize_stream(["   Padded.   "]))
        # "Padded." has length 7 after strip → 7 frames each valued 7.
        assert out[0].size == 7
        assert int(out[0][0]) == 7


# --------------------------------------------------------------------------- #
# SynthesizeStream — timing stats
# --------------------------------------------------------------------------- #


class TestSynthesizeStreamStats:
    def test_stats_keys_present(self) -> None:
        voice = _make_fake_voice()
        with patch.object(tts, "load_voice", return_value=voice):
            _, stats = _drain(tts.synthesize_stream(["A.", "B."]))
        assert set(stats) == {
            "time_to_first_audio_s",
            "total_latency_s",
            "num_sentences",
            "sample_rate",
        }

    def test_num_sentences_counts_chunks(self) -> None:
        voice = _make_fake_voice()
        with patch.object(tts, "load_voice", return_value=voice):
            _, stats = _drain(tts.synthesize_stream(["A.", "B.", "C.", "D."]))
        assert stats["num_sentences"] == 4

    def test_time_to_first_audio_is_float_when_audio_produced(self) -> None:
        voice = _make_fake_voice()
        with patch.object(tts, "load_voice", return_value=voice):
            _, stats = _drain(tts.synthesize_stream(["Hello."]))
        assert isinstance(stats["time_to_first_audio_s"], float)
        assert stats["time_to_first_audio_s"] >= 0.0

    def test_total_at_least_first_audio(self) -> None:
        voice = _make_fake_voice()
        with patch.object(tts, "load_voice", return_value=voice):
            _, stats = _drain(tts.synthesize_stream(["A.", "B."]))
        assert stats["total_latency_s"] >= stats["time_to_first_audio_s"]

    def test_sample_rate_defaults_to_voice_rate(self) -> None:
        voice = _make_fake_voice(sample_rate=22050)
        with patch.object(tts, "load_voice", return_value=voice):
            _, stats = _drain(tts.synthesize_stream(["Hi."]))
        assert stats["sample_rate"] == 22050

    def test_sample_rate_override_honored(self) -> None:
        voice = _make_fake_voice(sample_rate=22050)
        with patch.object(tts, "load_voice", return_value=voice):
            _, stats = _drain(
                tts.synthesize_stream(["Hi."], sample_rate=16000)
            )
        assert stats["sample_rate"] == 16000

    def test_rejects_non_positive_sample_rate(self) -> None:
        voice = _make_fake_voice()
        with patch.object(tts, "load_voice", return_value=voice):
            with pytest.raises(ValueError, match="sample_rate must be > 0"):
                _drain(tts.synthesize_stream(["Hi."], sample_rate=0))


# --------------------------------------------------------------------------- #
# SynthesizeStream — integration with llm_client.stream_sentences
# --------------------------------------------------------------------------- #


class TestSynthesizeStreamChaining:
    def test_consumes_stream_sentences_output(self) -> None:
        # Wire the real Day 2 sentence stream (LLM mocked) into TTS (voice
        # mocked). Confirms the two streaming layers chain cleanly.
        from src import llm_client

        def fake_tokens(*_a, **_k):
            for tok in ["Hello there. ", "How are you?"]:
                yield tok

        voice = _make_fake_voice()
        with patch.object(llm_client, "stream_generate", fake_tokens):
            with patch.object(tts, "load_voice", return_value=voice):
                sentences = llm_client.stream_sentences("prompt")
                out, stats = _drain(tts.synthesize_stream(sentences))
        assert stats["num_sentences"] == 2
        assert all(c.dtype == np.int16 and c.size > 0 for c in out)


# --------------------------------------------------------------------------- #
# RealVoiceStream — opt-in integration against the real Piper voice
# --------------------------------------------------------------------------- #


@pytest.mark.integration
class TestRealVoiceStream:
    def test_real_stream_two_sentences(self) -> None:
        out = []
        gen = tts.synthesize_stream(
            ["This is the first sentence.", "And this is the second."]
        )
        stats = {}
        try:
            while True:
                out.append(next(gen))
        except StopIteration as stop:
            stats = stop.value or {}

        assert stats["num_sentences"] == 2
        assert len(out) == 2
        assert all(isinstance(c, np.ndarray) and c.dtype == np.int16 for c in out)
        assert all(c.size > 0 for c in out)
        assert stats["sample_rate"] == tts.output_sample_rate()
        assert stats["time_to_first_audio_s"] is not None

    def test_real_synthesize_unchanged(self, tmp_path) -> None:
        # Batch synthesize() must still work (regression guard).
        out_path = str(tmp_path / "batch.wav")
        result = tts.synthesize("A short test sentence.", out_path)
        assert result["duration_s"] > 0
        with wave.open(out_path, "r") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2