"""Unit tests for Week 3 Day 2 — sentence-buffered LLM streaming.

The unit boundary is ``llm_client.stream_generate``: patching it with a
controlled token iterator isolates the sentence-buffering logic in
``stream_sentences`` from Ollama entirely. No network, no server, no model.

Mirrors the existing fast-unit-test architecture (Section 11): class-grouped,
deterministic, sub-second. One opt-in integration test hits a real Ollama
server when ``--run-integration`` is passed.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from src import llm_client


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _token_stream(tokens, *, delay_before=None, delay=0.0):
    """Build a fake stream_generate that yields ``tokens``.

    Accepts any positional/keyword args (so it stands in for
    ``stream_generate(prompt)`` and ``stream_generate(prompt, model=...)``).

    Args:
        tokens: Iterable of token strings to yield.
        delay_before: Optional index; sleep ``delay`` seconds just before
            yielding the token at this index (used for timing tests).
        delay: Sleep duration in seconds.
    """

    def fake(*_args, **_kwargs):
        for idx, tok in enumerate(tokens):
            if delay_before is not None and idx == delay_before and delay:
                time.sleep(delay)
            yield tok

    return fake


def _drain(gen: Iterator[str]) -> tuple[list[str], dict]:
    """Fully consume a generator, returning (yielded_items, return_value)."""
    out: list[str] = []
    try:
        while True:
            out.append(next(gen))
    except StopIteration as stop:
        return out, (stop.value or {})


# --------------------------------------------------------------------------- #
# SplitSentences — the pure buffer-splitting helper
# --------------------------------------------------------------------------- #


class TestSplitSentences:
    def test_single_complete_sentence(self) -> None:
        sentences, remainder = llm_client._split_sentences("Hello world.")
        assert sentences == ["Hello world."]
        assert remainder == ""

    def test_partial_sentence_left_in_remainder(self) -> None:
        sentences, remainder = llm_client._split_sentences("Done. More tex")
        assert sentences == ["Done."]
        assert remainder == " More tex"

    def test_multiple_sentences_in_one_buffer(self) -> None:
        sentences, remainder = llm_client._split_sentences("One. Two? Three!")
        assert sentences == ["One.", "Two?", "Three!"]
        assert remainder == ""

    def test_consecutive_terminators_single_break(self) -> None:
        sentences, remainder = llm_client._split_sentences("Really?! Yes...")
        assert sentences == ["Really?!", "Yes..."]
        assert remainder == ""

    def test_no_terminator_all_remainder(self) -> None:
        sentences, remainder = llm_client._split_sentences("no end here")
        assert sentences == []
        assert remainder == "no end here"

    def test_sentences_are_stripped(self) -> None:
        sentences, _ = llm_client._split_sentences("  Spaced out.   Next.")
        assert sentences == ["Spaced out.", "Next."]

    def test_question_and_exclamation(self) -> None:
        sentences, remainder = llm_client._split_sentences("Why? Stop!")
        assert sentences == ["Why?", "Stop!"]
        assert remainder == ""


# --------------------------------------------------------------------------- #
# StreamSentences — splitting behaviour
# --------------------------------------------------------------------------- #


class TestStreamSentencesSplitting:
    def test_single_sentence_one_token(self) -> None:
        fake = _token_stream(["Hello there."])
        with patch.object(llm_client, "stream_generate", fake):
            out, _ = _drain(llm_client.stream_sentences("p"))
        assert out == ["Hello there."]

    def test_sentence_spanning_multiple_tokens(self) -> None:
        fake = _token_stream(["Hel", "lo wor", "ld", "."])
        with patch.object(llm_client, "stream_generate", fake):
            out, _ = _drain(llm_client.stream_sentences("p"))
        assert out == ["Hello world."]

    def test_multiple_sentences_across_tokens(self) -> None:
        fake = _token_stream(["First one. ", "Second one? ", "Third!"])
        with patch.object(llm_client, "stream_generate", fake):
            out, _ = _drain(llm_client.stream_sentences("p"))
        assert out == ["First one.", "Second one?", "Third!"]

    def test_multiple_sentences_in_single_token(self) -> None:
        fake = _token_stream(["One. Two. Three."])
        with patch.object(llm_client, "stream_generate", fake):
            out, _ = _drain(llm_client.stream_sentences("p"))
        assert out == ["One.", "Two.", "Three."]

    def test_terminator_split_across_tokens(self) -> None:
        # The '.' arrives in its own token, after the sentence body.
        fake = _token_stream(["The answer is 42", "."])
        with patch.object(llm_client, "stream_generate", fake):
            out, _ = _drain(llm_client.stream_sentences("p"))
        assert out == ["The answer is 42."]

    def test_yielded_sentences_are_stripped(self) -> None:
        fake = _token_stream(["  Leading space.  ", "  Trailing too.  "])
        with patch.object(llm_client, "stream_generate", fake):
            out, _ = _drain(llm_client.stream_sentences("p"))
        assert out == ["Leading space.", "Trailing too."]


# --------------------------------------------------------------------------- #
# StreamSentences — edge cases
# --------------------------------------------------------------------------- #


class TestStreamSentencesEdgeCases:
    def test_no_terminal_punctuation_flushes_buffer(self) -> None:
        fake = _token_stream(["this has ", "no punctuation"])
        with patch.object(llm_client, "stream_generate", fake):
            out, stats = _drain(llm_client.stream_sentences("p"))
        assert out == ["this has no punctuation"]
        assert stats["num_sentences"] == 1

    def test_trailing_partial_after_complete_sentence(self) -> None:
        fake = _token_stream(["Complete. ", "and a trailing bit"])
        with patch.object(llm_client, "stream_generate", fake):
            out, _ = _drain(llm_client.stream_sentences("p"))
        assert out == ["Complete.", "and a trailing bit"]

    def test_empty_stream_yields_nothing(self) -> None:
        fake = _token_stream([])
        with patch.object(llm_client, "stream_generate", fake):
            out, stats = _drain(llm_client.stream_sentences("p"))
        assert out == []
        assert stats["num_sentences"] == 0
        assert stats["first_token_latency_s"] is None
        assert stats["time_to_first_sentence_s"] is None

    def test_whitespace_only_stream_yields_nothing(self) -> None:
        fake = _token_stream(["   ", "  \n ", "\t"])
        with patch.object(llm_client, "stream_generate", fake):
            out, stats = _drain(llm_client.stream_sentences("p"))
        assert out == []
        assert stats["num_sentences"] == 0

    def test_empty_token_chunks_ignored(self) -> None:
        # Ollama can emit empty content chunks; they must not break anything.
        fake = _token_stream(["", "Hello.", "", "World."])
        with patch.object(llm_client, "stream_generate", fake):
            out, _ = _drain(llm_client.stream_sentences("p"))
        assert out == ["Hello.", "World."]


# --------------------------------------------------------------------------- #
# StreamSentences — timing stats
# --------------------------------------------------------------------------- #


class TestStreamSentencesTiming:
    def test_stats_keys_present(self) -> None:
        fake = _token_stream(["Hi.", "Bye."])
        with patch.object(llm_client, "stream_generate", fake):
            _, stats = _drain(llm_client.stream_sentences("p"))
        assert set(stats) == {
            "first_token_latency_s",
            "time_to_first_sentence_s",
            "total_latency_s",
            "num_sentences",
        }

    def test_num_sentences_counted(self) -> None:
        fake = _token_stream(["A. ", "B. ", "C."])
        with patch.object(llm_client, "stream_generate", fake):
            _, stats = _drain(llm_client.stream_sentences("p"))
        assert stats["num_sentences"] == 3

    def test_timings_are_non_negative_floats(self) -> None:
        fake = _token_stream(["Hello.", "World."])
        with patch.object(llm_client, "stream_generate", fake):
            _, stats = _drain(llm_client.stream_sentences("p"))
        assert isinstance(stats["first_token_latency_s"], float)
        assert isinstance(stats["time_to_first_sentence_s"], float)
        assert isinstance(stats["total_latency_s"], float)
        assert stats["first_token_latency_s"] >= 0.0
        assert stats["time_to_first_sentence_s"] >= 0.0
        assert stats["total_latency_s"] >= 0.0

    def test_time_to_first_sentence_brackets_the_assembly_gap(self) -> None:
        # Delay between the first token and the token that completes the
        # first sentence. time_to_first_sentence_s must be at least that gap.
        gap = 0.03
        fake = _token_stream(
            ["Building up ", "the sentence now."],
            delay_before=1,
            delay=gap,
        )
        with patch.object(llm_client, "stream_generate", fake):
            _, stats = _drain(llm_client.stream_sentences("p"))
        assert stats["time_to_first_sentence_s"] >= gap * 0.8  # margin for jitter

    def test_total_latency_at_least_first_sentence_time(self) -> None:
        fake = _token_stream(["One. ", "Two."])
        with patch.object(llm_client, "stream_generate", fake):
            _, stats = _drain(llm_client.stream_sentences("p"))
        assert stats["total_latency_s"] >= stats["time_to_first_sentence_s"]


# --------------------------------------------------------------------------- #
# StreamSentences — model forwarding
# --------------------------------------------------------------------------- #


class TestStreamSentencesModelForwarding:
    def test_default_model_calls_stream_generate_without_model_kwarg(self) -> None:
        mock = MagicMock(side_effect=lambda *a, **k: iter(["Hi."]))
        with patch.object(llm_client, "stream_generate", mock):
            list(llm_client.stream_sentences("my prompt"))
        mock.assert_called_once()
        args, kwargs = mock.call_args
        assert args == ("my prompt",)
        assert "model" not in kwargs

    def test_explicit_model_forwarded(self) -> None:
        mock = MagicMock(side_effect=lambda *a, **k: iter(["Hi."]))
        with patch.object(llm_client, "stream_generate", mock):
            list(llm_client.stream_sentences("my prompt", model="tinyllama:1.1b"))
        mock.assert_called_once()
        args, kwargs = mock.call_args
        assert args == ("my prompt",)
        assert kwargs.get("model") == "tinyllama:1.1b"


# --------------------------------------------------------------------------- #
# measure_first_sentence_latency — benchmark wrapper
# --------------------------------------------------------------------------- #


class TestMeasureFirstSentenceLatency:
    def test_returns_expected_keys(self) -> None:
        fake = _token_stream(["Hello.", "World."])
        with patch.object(llm_client, "stream_generate", fake):
            result = llm_client.measure_first_sentence_latency("p")
        assert set(result) == {
            "model",
            "num_sentences",
            "first_token_latency_s",
            "time_to_first_sentence_s",
            "total_latency_s",
            "sentences",
        }

    def test_collects_all_sentences(self) -> None:
        fake = _token_stream(["First. ", "Second. ", "Third."])
        with patch.object(llm_client, "stream_generate", fake):
            result = llm_client.measure_first_sentence_latency("p")
        assert result["sentences"] == ["First.", "Second.", "Third."]
        assert result["num_sentences"] == 3

    def test_default_model_field(self) -> None:
        fake = _token_stream(["Hi."])
        with patch.object(llm_client, "stream_generate", fake):
            result = llm_client.measure_first_sentence_latency("p")
        assert result["model"] == llm_client.PRIMARY_MODEL

    def test_override_model_field(self) -> None:
        fake = _token_stream(["Hi."])
        with patch.object(llm_client, "stream_generate", fake):
            result = llm_client.measure_first_sentence_latency(
                "p", model=llm_client.FALLBACK_MODEL
            )
        assert result["model"] == llm_client.FALLBACK_MODEL

    def test_empty_stream_safe(self) -> None:
        fake = _token_stream([])
        with patch.object(llm_client, "stream_generate", fake):
            result = llm_client.measure_first_sentence_latency("p")
        assert result["sentences"] == []
        assert result["num_sentences"] == 0
        assert result["first_token_latency_s"] is None
        assert result["time_to_first_sentence_s"] is None


# --------------------------------------------------------------------------- #
# RealStream — opt-in integration against a live Ollama server
# --------------------------------------------------------------------------- #


@pytest.mark.integration
class TestRealStream:
    def test_real_stream_sentences_primary(self) -> None:
        result = llm_client.measure_first_sentence_latency(
            "Reply with exactly two short sentences about the sky."
        )
        # At least one sentence, all non-empty and stripped.
        assert result["num_sentences"] >= 1
        assert all(s == s.strip() and s for s in result["sentences"])
        assert result["first_token_latency_s"] is not None
        assert result["time_to_first_sentence_s"] is not None
        assert result["total_latency_s"] >= result["time_to_first_sentence_s"]

    def test_real_generate_unchanged(self) -> None:
        # generate() must still work end-to-end (regression guard).
        out = llm_client.generate("Say hello in one short sentence.")
        assert isinstance(out["text"], str) and out["text"].strip()
        assert isinstance(out["latency_s"], float) and out["latency_s"] > 0


class TestStreamSentencesFromMessages:
    def test_yields_sentences_from_messages(self):
        fake = _token_stream(["Hello there. ", "How are you?"])
        with patch.object(llm_client, "stream_generate_messages", fake):
            out, _ = _drain(llm_client.stream_sentences_from_messages([{"role": "user", "content": "hi"}]))
        assert out == ["Hello there.", "How are you?"]

    def test_default_model_omits_model_kwarg(self):
        mock = MagicMock(side_effect=lambda *a, **k: iter(["Hi."]))
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        with patch.object(llm_client, "stream_generate_messages", mock):
            list(llm_client.stream_sentences_from_messages(msgs))
        args, kwargs = mock.call_args
        assert args == (msgs,)
        assert "model" not in kwargs

    def test_explicit_model_forwarded(self):
        mock = MagicMock(side_effect=lambda *a, **k: iter(["Hi."]))
        msgs = [{"role": "user", "content": "u"}]
        with patch.object(llm_client, "stream_generate_messages", mock):
            list(llm_client.stream_sentences_from_messages(msgs, model="tinyllama:1.1b"))
        args, kwargs = mock.call_args
        assert args == (msgs,)
        assert kwargs.get("model") == "tinyllama:1.1b"

    def test_stats_returned_via_stopiteration(self):
        fake = _token_stream(["One. ", "Two."])
        with patch.object(llm_client, "stream_generate_messages", fake):
            _, stats = _drain(llm_client.stream_sentences_from_messages([{"role": "user", "content": "x"}]))
        assert stats["num_sentences"] == 2
        assert set(stats) == {"first_token_latency_s", "time_to_first_sentence_s", "total_latency_s", "num_sentences"}

    def test_no_terminal_punctuation_flushed(self):
        fake = _token_stream(["no period here"])
        with patch.object(llm_client, "stream_generate_messages", fake):
            out, _ = _drain(llm_client.stream_sentences_from_messages([{"role": "user", "content": "x"}]))
        assert out == ["no period here"]