"""
tts.py — Text-to-Speech via Piper TTS
Voice: en_US-amy-medium
"""

import time
import wave
import struct
from piper.voice import PiperVoice
import io
from collections.abc import Iterable, Iterator
import numpy as np


MODEL_PATH = "models/piper/en_US-amy-medium.onnx"
CONFIG_PATH = "models/piper/en_US-amy-medium.onnx.json"

_voice = None


def load_voice():
    """Load Piper voice model (cached after first call)."""
    global _voice
    if _voice is None:
        print(f"Loading TTS voice: {MODEL_PATH}")
        _voice = PiperVoice.load(MODEL_PATH, config_path=CONFIG_PATH)
        print("TTS voice loaded.")
    return _voice


def synthesize(text: str, output_path: str) -> dict:
    """
    Synthesize text to a .wav file.
    Args:
        text: input text to synthesize
        output_path: path to save .wav file
    Returns:
        dict with keys: output_path, duration_s, latency_s, rtf
    """
    voice = load_voice()
    start = time.time()

    with wave.open(output_path, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(voice.config.sample_rate)
        voice.synthesize_wav(text, wav_file)

    latency = round(time.time() - start, 3)

    # Get audio duration
    with wave.open(output_path, "r") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = round(frames / float(rate), 3)

    rtf = round(latency / duration, 3) if duration > 0 else 0

    return {
        "output_path": output_path,
        "duration_s": duration,
        "latency_s": latency,
        "rtf": rtf
    }


def output_sample_rate() -> int:
    """Native sample rate (Hz) of the loaded TTS voice."""
    return int(load_voice().config.sample_rate)


def synthesize_stream(sentences, sample_rate=None):
    """Synthesize an iterable of sentences, yielding int16 PCM per sentence.

    Reuses the exact synthesize_wav path as synthesize() (in-memory WAV →
    decode to PCM), so streamed audio is byte-identical to batch. Pairs
    directly with llm_client.stream_sentences().

    Returns (via StopIteration.value):
        {time_to_first_audio_s, total_latency_s, num_sentences, sample_rate}
    """
    voice = load_voice()
    sr = int(sample_rate) if sample_rate is not None else int(voice.config.sample_rate)
    if sr <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sr}")

    t_start = time.time()
    t_first_audio = None
    num_chunks = 0

    for sentence in sentences:
        text = sentence.strip()
        if not text:
            continue
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sr)
            voice.synthesize_wav(text, wav_file)
        buf.seek(0)
        with wave.open(buf, "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
        pcm = np.frombuffer(frames, dtype=np.int16)
        if pcm.size == 0:
            continue
        if t_first_audio is None:
            t_first_audio = time.time()
        num_chunks += 1
        yield pcm

    t_end = time.time()
    return {
        "time_to_first_audio_s": None if t_first_audio is None else round(t_first_audio - t_start, 3),
        "total_latency_s": round(t_end - t_start, 3),
        "num_sentences": num_chunks,
        "sample_rate": sr,
    }

if __name__ == "__main__":
    test_sentences = [
        "Hello, I am your voice assistant. How can I help you today?",
        "The weather today is sunny with a high of 25 degrees.",
        "I'm sorry, I did not understand that. Could you please repeat?",
        "Your appointment has been scheduled for tomorrow at 10 AM.",
        "Thank you for calling. Have a wonderful day!"
    ]

    print("=== TTS Benchmark ===\n")
    for i, sentence in enumerate(test_sentences):
        output_path = f"recordings/tts_test_{i+1}.wav"
        print(f"Sentence {i+1}: {sentence}")
        result = synthesize(sentence, output_path)
        print(f"Audio duration : {result['duration_s']}s")
        print(f"TTS latency    : {result['latency_s']}s")
        print(f"RTF            : {result['rtf']}")
        print(f"Saved to       : {result['output_path']}")
        print()