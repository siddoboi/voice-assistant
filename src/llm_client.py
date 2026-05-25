"""
llm_client.py — LLM interface via Ollama
Primary model: llama3.2:1b-instruct-q4_K_M
Fallback model: tinyllama:1.1b
"""

import time
import ollama

PRIMARY_MODEL = "llama3.2:1b-instruct-q4_K_M"
FALLBACK_MODEL = "tinyllama:1.1b"


def generate(prompt: str, model: str = PRIMARY_MODEL) -> dict:
    """Generate a full response and return text + timing."""
    start = time.time()
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    elapsed = time.time() - start
    text = response["message"]["content"]
    return {"text": text, "latency_s": round(elapsed, 3)}


def stream_generate(prompt: str, model: str = PRIMARY_MODEL):
    """Stream response tokens, yielding each chunk as it arrives."""
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    for chunk in stream:
        token = chunk["message"]["content"]
        yield token


def measure_latency(model: str, prompt: str = "Reply in one sentence: What is 2 plus 2?") -> dict:
    """Measure first-token latency and total latency for a model."""
    # Total latency
    result = generate(prompt, model=model)

    # First token latency via streaming
    start = time.time()
    first_token_time = None
    full_text = ""
    for token in stream_generate(prompt, model=model):
        if first_token_time is None:
            first_token_time = round(time.time() - start, 3)
        full_text += token

    return {
        "model": model,
        "first_token_latency_s": first_token_time,
        "total_latency_s": result["latency_s"],
        "response": full_text.strip()
    }


if __name__ == "__main__":
    print("=== Benchmarking Primary Model ===")
    r1 = measure_latency(PRIMARY_MODEL)
    print(f"Model         : {r1['model']}")
    print(f"First token   : {r1['first_token_latency_s']}s")
    print(f"Total latency : {r1['total_latency_s']}s")
    print(f"Response      : {r1['response']}")

    print()
    print("=== Benchmarking Fallback Model ===")
    r2 = measure_latency(FALLBACK_MODEL)
    print(f"Model         : {r2['model']}")
    print(f"First token   : {r2['first_token_latency_s']}s")
    print(f"Total latency : {r2['total_latency_s']}s")
    print(f"Response      : {r2['response']}")
