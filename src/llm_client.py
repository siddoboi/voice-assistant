"""
llm_client.py — LLM interface via Ollama
Primary model: llama3.2:1b-instruct-q4_K_M
Fallback model: tinyllama:1.1b
"""

import time
import ollama
from collections.abc import Iterator

PRIMARY_MODEL = "llama3.2:1b-instruct-q4_K_M"
FALLBACK_MODEL = "tinyllama:1.1b"

_SENTENCE_TERMINATORS = ".?!"

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

def _split_sentences(buffer: str) -> tuple[list[str], str]:
    """Split complete sentences off the front of buffer.

    A run of consecutive terminators (e.g. '?!', '...') counts as one break.
    Whitespace-stripped empty fragments are dropped.
    Returns (sentences, remainder) where remainder is the in-progress sentence.

    Note: purely punctuation-based — decimals ('3.5') and abbreviations
    ('Mr.') will break early. Acceptable for short conversational replies.
    """
    sentences: list[str] = []
    start = 0
    i = 0
    n = len(buffer)
    while i < n:
        if buffer[i] in _SENTENCE_TERMINATORS:
            j = i
            while j + 1 < n and buffer[j + 1] in _SENTENCE_TERMINATORS:
                j += 1
            sentence = buffer[start : j + 1].strip()
            if sentence:
                sentences.append(sentence)
            start = j + 1
            i = j + 1
        else:
            i += 1
    return sentences, buffer[start:]


def stream_sentences(prompt: str, model: str | None = None) -> Iterator[str]:
    """Stream LLM output as complete sentences.

    Buffers raw tokens from stream_generate() and yields one complete
    sentence each time a terminator (. ? !) is encountered. At end of
    stream, any non-empty trailing buffer is flushed as a final sentence
    (handles replies with no terminal punctuation).

    Model convention: None uses stream_generate()'s PRIMARY_MODEL default;
    an explicit model string is forwarded.

    The generator return value (StopIteration.value) is a stats dict:
        {
            first_token_latency_s:    float | None,  # request → first token
            time_to_first_sentence_s: float | None,  # first token → first sentence
            total_latency_s:          float,
            num_sentences:            int,
        }
    time_to_first_sentence_s maps to the 'First sentence assembled' line
    in the Pi latency budget (800ms target).
    """
    token_iter = (
        stream_generate(prompt)
        if model is None
        else stream_generate(prompt, model=model)
    )

    buffer = ""
    t_start = time.time()
    t_first_token: float | None = None
    t_first_sentence: float | None = None
    num_sentences = 0

    for token in token_iter:
        if t_first_token is None:
            t_first_token = time.time()
        buffer += token
        sentences, buffer = _split_sentences(buffer)
        for sentence in sentences:
            if t_first_sentence is None:
                t_first_sentence = time.time()
            num_sentences += 1
            yield sentence

    remainder = buffer.strip()
    if remainder:
        if t_first_sentence is None:
            t_first_sentence = time.time()
        num_sentences += 1
        yield remainder

    t_end = time.time()
    return {
        "first_token_latency_s": (
            None if t_first_token is None else round(t_first_token - t_start, 3)
        ),
        "time_to_first_sentence_s": (
            None
            if (t_first_token is None or t_first_sentence is None)
            else round(t_first_sentence - t_first_token, 3)
        ),
        "total_latency_s": round(t_end - t_start, 3),
        "num_sentences": num_sentences,
    }


def measure_first_sentence_latency(
    prompt: str, model: str | None = None
) -> dict:
    """Benchmark sentence-buffered streaming. Mirrors measure_latency().

    Returns:
        {model, num_sentences, first_token_latency_s,
         time_to_first_sentence_s, total_latency_s, sentences}
    """
    sentences: list[str] = []
    gen = stream_sentences(prompt, model=model)
    stats: dict = {}
    try:
        while True:
            sentences.append(next(gen))
    except StopIteration as stop:
        stats = stop.value or {}

    return {
        "model": PRIMARY_MODEL if model is None else model,
        "num_sentences": stats.get("num_sentences", len(sentences)),
        "first_token_latency_s": stats.get("first_token_latency_s"),
        "time_to_first_sentence_s": stats.get("time_to_first_sentence_s"),
        "total_latency_s": stats.get("total_latency_s"),
        "sentences": sentences,
    }


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
