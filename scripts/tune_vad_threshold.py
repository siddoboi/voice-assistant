#!/usr/bin/env python3
"""Sweep Silero VAD decision thresholds over a single WAV file.

On Pi Day 1 this is run against a real SIM7600EI call recording to pick the
silence threshold that best separates speech from the GSM noise floor. The
clean-WAV default of 0.5 over-triggers on line hiss, so the production value
is expected to land around 0.6-0.65.

How it works
------------
``vad.test_on_file()`` cannot be reused for a sweep because it bakes in the
module-level ``SILENCE_THRESHOLD`` via ``is_speech()``. Instead this script
makes a single VAD pass with ``vad.get_speech_prob()``, stores the raw
per-chunk probability array, and then evaluates every candidate threshold
against that stored array. One model pass, five thresholds, no module-state
mutation.

The WAV is loaded and resampled to 16 kHz mono using the same conventions as
``vad.test_on_file()`` (int16 -> float32 / 32768, ``np.interp`` resampling).

No new dependencies: uses only ``src.vad``, ``src.audio_io``, and numpy
(already required by both).

Usage
-----
    python scripts/tune_vad_threshold.py recordings/sample1.wav
    python scripts/tune_vad_threshold.py /path/to/gsm_call.wav --output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# --- Make the project importable and fix CWD ------------------------------
# scripts/ sits one level below the project root. src.vad hardcodes a
# *relative* model path ("models/silero/..."), so the working directory must
# be the project root or model loading fails. Resolve both from __file__ so
# the script works regardless of where it is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from src import audio_io, vad  # noqa: E402  (import after sys.path setup)

# Candidate thresholds to sweep. 0.5 is the WSL2 / clean-WAV default; the
# higher values are the GSM-noise-floor candidates.
THRESHOLDS: tuple[float, ...] = (0.5, 0.55, 0.6, 0.65, 0.7)

# Default output location (relative to project root, which is now CWD).
DEFAULT_OUTPUT = Path("recordings") / "vad_threshold_results.json"

# Width of the ASCII bar used to visualise speech_ratio.
_BAR_WIDTH = 40


def _load_mono_16k(wav_path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV as float32 mono at 16 kHz, mirroring vad.test_on_file().

    Args:
        wav_path: Path to the source WAV file.

    Returns:
        Tuple of (audio, original_rate) where audio is a 1-D float32 array
        normalised to [-1.0, 1.0] at ``vad.SAMPLE_RATE`` (16 kHz).
    """
    raw, original_rate = audio_io.load_wav(wav_path)

    # Down-mix to mono if the file is multi-channel (load_wav returns a 2-D
    # array shaped (n_samples, channels) in that case).
    if raw.ndim == 2:
        audio = raw.astype(np.float32).mean(axis=1)
    else:
        audio = raw.astype(np.float32)

    # int16 PCM -> [-1.0, 1.0], matching vad.test_on_file()'s convention.
    audio = audio / 32768.0

    # Resample to 16 kHz via linear interpolation if needed. Same np.interp
    # approach used inside vad.test_on_file(); kept here so any input rate is
    # handled (audio_io.resample_8_to_16k only covers the 8->16k case).
    if original_rate != vad.SAMPLE_RATE:
        ratio = vad.SAMPLE_RATE / original_rate
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    return audio, original_rate


def _collect_speech_probs(audio: np.ndarray) -> np.ndarray:
    """Run a single VAD pass and return one speech probability per chunk.

    Chunks the audio into ``vad.CHUNK_SIZE`` frames (dropping any short final
    frame, exactly as vad.test_on_file() does) and feeds each through
    ``vad.get_speech_prob()``. State is reset once before the pass.

    Args:
        audio: 1-D float32 array at 16 kHz.

    Returns:
        1-D float32 array of per-chunk speech probabilities in [0.0, 1.0].
    """
    vad.load_model()
    vad.reset_state()

    probs: list[float] = []
    chunk = vad.CHUNK_SIZE
    # Match test_on_file()'s loop bound: range(0, len - CHUNK_SIZE, CHUNK_SIZE)
    # so only full chunks are scored.
    for i in range(0, len(audio) - chunk, chunk):
        frame = audio[i : i + chunk]
        if len(frame) == chunk:
            probs.append(vad.get_speech_prob(frame))

    return np.asarray(probs, dtype=np.float32)


def _evaluate(probs: np.ndarray, threshold: float) -> dict:
    """Count speech/silence chunks for one threshold over precomputed probs.

    Uses ``>=`` to match ``vad.is_speech()`` semantics exactly.

    Args:
        probs: Per-chunk speech probabilities.
        threshold: Decision threshold to apply.

    Returns:
        Dict with threshold, total_chunks, speech_chunks, silence_chunks,
        and speech_ratio (rounded to 3 dp).
    """
    total = int(probs.size)
    speech = int(np.count_nonzero(probs >= threshold))
    silence = total - speech
    ratio = round(speech / total, 3) if total else 0.0
    return {
        "threshold": threshold,
        "total_chunks": total,
        "speech_chunks": speech,
        "silence_chunks": silence,
        "speech_ratio": ratio,
    }


def _bar(ratio: float, width: int = _BAR_WIDTH) -> str:
    """Render a speech_ratio (0.0-1.0) as a fixed-width ASCII meter."""
    filled = int(round(ratio * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def sweep(wav_path: Path, output_path: Path) -> dict:
    """Run the full threshold sweep, print a report, and save JSON.

    Args:
        wav_path: WAV file to analyse.
        output_path: Where to write the JSON results.

    Returns:
        The results dict that was written to ``output_path``.
    """
    audio, original_rate = _load_mono_16k(wav_path)
    probs = _collect_speech_probs(audio)

    results = [_evaluate(probs, t) for t in THRESHOLDS]

    report = {
        "file": str(wav_path),
        "original_rate": original_rate,
        "resampled_rate": vad.SAMPLE_RATE,
        "chunk_size": vad.CHUNK_SIZE,
        "total_chunks": int(probs.size),
        "thresholds": results,
    }

    # --- Console report ---
    print(f"\n=== VAD Threshold Sweep ===")
    print(f"File          : {wav_path}")
    print(f"Original rate : {original_rate} Hz -> {vad.SAMPLE_RATE} Hz")
    print(f"Total chunks  : {probs.size} (chunk size {vad.CHUNK_SIZE})\n")
    print(f"{'Thresh':>6}  {'Speech':>6}  {'Silence':>7}  {'Ratio':>6}  Speech ratio")
    print("-" * 78)
    for r in results:
        print(
            f"{r['threshold']:>6.2f}  "
            f"{r['speech_chunks']:>6}  "
            f"{r['silence_chunks']:>7}  "
            f"{r['speech_ratio'] * 100:>5.1f}%  "
            f"{_bar(r['speech_ratio'])}"
        )
    print()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Results saved to {output_path}\n")

    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep Silero VAD silence thresholds (0.5-0.7) over a WAV file "
            "and report speech/silence chunk counts per threshold."
        )
    )
    parser.add_argument(
        "wav_path",
        type=Path,
        help="Path to the WAV file to analyse (e.g. recordings/sample1.wav).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write JSON results (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # wav_path may be given relative to the original CWD; since we chdir'd to
    # the project root, resolve it against the root if it isn't absolute and
    # doesn't already exist as given.
    wav_path = args.wav_path
    if not wav_path.is_absolute() and not wav_path.exists():
        candidate = _PROJECT_ROOT / wav_path
        if candidate.exists():
            wav_path = candidate

    if not wav_path.exists():
        print(f"error: WAV file not found: {args.wav_path}", file=sys.stderr)
        return 2

    sweep(wav_path, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())