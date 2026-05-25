## Day 1 — WSL2 Setup

### Done
- Debian 13 Trixie WSL2 confirmed (x86_64, 3.1 GB RAM)
- Python 3.13.5 installed from Trixie repos
- All system dependencies installed (audio libs, build tools, ffmpeg, sqlite, portaudio)
- Project folder structure created
- Virtual environment created and verified
- Ollama 0.24.0 installed

### Pending (Day 2)
- Pull Llama 3.2 1B Q4_K_M model
- Pull TinyLlama 1.1B fallback
- Write llm_client.py skeleton
- Benchmark LLM latency and RAM usage


## Day 2 — LLM Install & Test

### Done
- Pulled llama3.2:1b-instruct-q4_K_M (807 MB)
- Pulled tinyllama:1.1b (637 MB)
- Both models verified via ollama list
- llm_client.py skeleton written with generate(), stream_generate(), measure_latency()

### Benchmark Results (WSL2 x86_64)
- Llama 3.2 1B — first token: 0.297s, total: 5.084s
- TinyLlama 1.1B — first token: 0.116s, total: 5.981s
- Note: WSL2 latency not representative — Pi ARM benchmarks in Week 3

### Decision
- Llama 3.2 1B confirmed as primary (cleaner responses)
- TinyLlama confirmed as fallback only


### RAM Benchmark (WSL2 x86_64)
- Llama 3.2 1B Q4_K_M RAM delta: ~1020 MB (~1 GB)
- Baseline RAM before load: 454 MB
- RAM after model load: 1474 MB
- Well within 3.5 GB Pi target


## Day 3 — ASR Install & Test

### Done
- faster-whisper 1.2.1 installed
- tiny.en model downloaded and cached
- asr.py skeleton written with load_model(), transcribe()
- Benchmarked on 2 sample .wav files

### Benchmark Results (WSL2 x86_64)
- Sample 1: duration 51s, latency 3.278s, RTF 0.064
- Sample 2: duration 49s, latency 2.889s, RTF 0.059
- Average RTF: ~0.06 (16x faster than real-time)
- Transcription quality: clean, no obvious errors
- Expected Pi latency for 3-5s utterance: ~200-300ms (well under 800ms budget)

### Decision
- tiny.en confirmed as Phase 1 ASR model
- No need to evaluate alternatives


## Day 4 — TTS Install & Test

### Done
- Piper TTS 1.4.2 installed
- en_US-amy-medium voice model downloaded (61MB)
- tts.py skeleton written with load_voice(), synthesize()
- Benchmarked on 5 test sentences

### Benchmark Results (WSL2 x86_64)
- Average RTF     : 0.067 (15x faster than real-time)
- Average latency : 0.261s (under 250ms budget)
- Average duration: 3.885s per sentence
- All 5 .wav files generated correctly

### Decision
- Piper en_US-amy-medium confirmed as Phase 1 TTS voice
- Performance exceeds expectations on WSL2

