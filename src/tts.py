"""
tts.py — Text-to-Speech via Piper TTS
Voice: en_US-amy-medium
"""

import time
import wave
import struct
from piper.voice import PiperVoice

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