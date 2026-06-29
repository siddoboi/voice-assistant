# Voice Assistant — Offline Phone Call Responder (Raspberry Pi 5)

A fully offline, on-device conversational AI that autonomously answers real cellular phone calls. When someone calls the SIM card in the SIM7600EI 4G LTE GSM HAT, the Raspberry Pi 5 answers, transcribes the caller's speech, generates a contextual reply with a local LLM, and speaks the response back — with **no cloud services and no internet required at runtime**. All speech recognition, language modelling, and speech synthesis run locally on the Pi.

> A software-based prototype is ready: the full streaming pipeline (VAD → ASR → LLM → TTS), multi-turn conversation management, and GSM call-control signalling are implemented and covered by a passing test suite. Live GSM hardware integration is the remaining step.

---

## Why GSM and not VoIP

Indian Department of Telecommunications (DoT) regulations prohibit bridging VoIP to the public telephone network (PSTN/mobile) without a Unified Telecom License. The SIM7600EI operates as a standard cellular device — no bridging, no license required — which is why this project answers calls over a real GSM module rather than SIP/VoIP.

---

## Architecture

```
Incoming call → SIM7600EI HAT → VAD (Silero) → ASR (faster-whisper)
              → LLM (Llama 3.2 via Ollama) → TTS (Piper) → caller hears reply
```

The pipeline streams: the LLM's token output is buffered into complete sentences as they form, each sentence is synthesized to audio immediately, and a bounded `asyncio.Queue` applies back-pressure so the first words of a reply play before the full response finishes generating. This caps memory on the 4 GB Pi and minimises perceived latency.

**Perceived latency** is measured as ASR time plus time-to-first-audio — the gap between the caller finishing speaking and hearing the first word back. Target: ≤ 3.5 s per turn.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama — Llama 3.2 1B Instruct (Q4_K_M) primary, TinyLlama 1.1B fallback |
| ASR | faster-whisper (tiny.en, int8, CPU) |
| TTS | Piper TTS (en_US-amy-medium) |
| VAD | Silero VAD v4 via ONNX Runtime |
| Noise reduction | noisereduce (lazy-imported, config-toggleable) |
| Telephony | SIM7600EI 4G LTE HAT driven by pyserial AT commands |
| Audio I/O | sounddevice (PortAudio) |
| Concurrency | asyncio (bounded-queue streaming) |
| Persistence | SQLite (conversation history) |
| Config | PyYAML |
| Testing | pytest |
| Dev OS | WSL2 Debian Trixie · **Deploy OS:** Raspberry Pi OS 64-bit |

---

## Hardware Requirements

- Raspberry Pi 5 (4 GB) + official active cooler + 27 W USB-C PD power supply
- 64 GB A2-rated microSD card
- SIM7600EI 4G LTE GSM HAT (Indian LTE bands, VoLTE) + antenna
- Prepaid voice SIM card
- USB audio adapter + TRRS earphones with inline mic (local testing; live call audio routes through the HAT's 3.5 mm jack)

Development is done on WSL2; only live audio, GSM calls, and latency profiling require the Pi.

---

## Setup

```bash
git clone https://github.com/siddoboi/voice-assistant.git
cd voice-assistant
bash setup.sh
source venv/bin/activate
ollama serve &
```

`setup.sh` installs system packages, creates the Python 3.13 virtual environment, installs dependencies, pulls the Ollama models, and downloads the Piper TTS voice and Silero VAD v4 model. It runs identically on WSL2 Debian Trixie and Raspberry Pi OS 64-bit.

On the Pi, activate the Pi configuration before running:

```bash
export VOICE_ASSISTANT_CONFIG=configs/pi_config.yaml
```

---

## Usage

Run the pipeline against a pre-recorded WAV file:

```bash
python -m src.pipeline --input recordings/sample1.wav
```

Flags: `--no-play` (skip playback), `--output <path>` (save reply WAV), `--model <name>` (override LLM), `--session-id <id>` (resume a conversation).

Helper scripts:

```bash
python scripts/tune_vad_threshold.py recordings/sample1.wav   # VAD threshold sweep
python scripts/benchmark_pi.py --input recordings/sample1.wav # full benchmark suite
```

---

## Project Structure

```
voice-assistant/
├── src/
│   ├── audio_io.py        # record/play/resample/WAV I/O + noise reduction
│   ├── asr.py             # faster-whisper transcription
│   ├── llm_client.py      # Ollama client + sentence streaming
│   ├── tts.py             # Piper batch + streaming synthesis
│   ├── vad.py             # Silero VAD v4 voice activity detection
│   ├── conversation.py    # multi-turn history + SQLite persistence
│   ├── pipeline.py        # streaming record→ASR→LLM→TTS→play orchestration
│   ├── main.py            # VAD-driven call loop (needs Pi)
│   └── telephony/
│       └── gsm_adapter.py # SIM7600EI AT-command call control
├── configs/
│   ├── dev_config.yaml    # WSL2 development settings
│   ├── pi_config.yaml     # Raspberry Pi deployment settings
│   └── models.yaml        # model paths and benchmark metrics
├── scripts/               # benchmark + VAD tuning tools
├── tests/                 # pytest unit + opt-in integration suite (302 tests)
└── setup.sh
```

---

## Testing

```bash
pytest tests/                      # unit tests (fast, mocked)
pytest tests/ --run-integration    # full suite with real models
```

A two-layer test suite covers every module — fast mocked unit tests by default, with real-model integration tests available via the opt-in flag.

---

## Roadmap

- **Core software (done):** modules, streaming pipeline, conversation management, GSM signalling, full test suite
- **Tooling (done):** Pi config, benchmark + VAD tuning scripts
- **Next:** Pi hardware setup, GSM audio routing, first live call
- **Then:** WER evaluation, latency profiling, stability testing
- **Then:** systemd packaging, demo, final report
- **Future:** Multilingual support

A parallel implementation that runs on native Ubuntu (with the GSM module over USB) is developed in [voice-assistant-ubuntu](https://github.com/siddoboi/voice-assistant-ubuntu).

---

## License

MIT