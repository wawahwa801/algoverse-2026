import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import ollama

from core.config.config import (
    KEEP_ALIVE,
    NUM_CTX,
    TOP_LOGPROBS,
    PROBE_WORKERS,
)
from core.evaluation.metrics import get_cut_points, find_commitment_point
from core.clients.clients import get_model_profile, get_client

# Long-lived executor for cut-point probing, shared across every call to
# run_probe_on_item() for the life of the process. Previously a fresh
# ThreadPoolExecutor was created per call, which meant every probe worker
# thread was new and its thread-local OpenRouterModelClient (event loop +
# httpx.AsyncClient) was created once and never closed - fine at a handful
# of examples, but a real leak at thousands of tasks. Reusing this pool lets
# thread-local clients actually be reused as intended.
_PROBE_EXECUTOR = ThreadPoolExecutor(max_workers=PROBE_WORKERS)


def generate_full_chain(
    question_prompt: str,
    model_name: str,
    max_tokens: int = 2048,
    effort: str = "medium",
) -> str:
    backend = get_model_profile(model_name)["backend"]

    if backend == "ollama":
        response = ollama.generate(
            model=model_name,
            prompt=question_prompt,
            raw=True,
            keep_alive=KEEP_ALIVE,
            options={
                "temperature": 0.7,
                "num_predict": max_tokens,
                "num_ctx": NUM_CTX,
            },
        )
        return response["response"]

    client = get_client(model_name)
    response = client.ask(
        question_prompt,
        effort=effort,
        max_tokens=max_tokens,
    )

    return response.thinking or response.content


def probe_cut_point(
    question_prompt: str,
    partial_reasoning: str,
    model_name: str,
    answer_options=("0", "1", "2"),
) -> dict:
    forced_prompt = (
        question_prompt
        + partial_reasoning
        + "\n\nBased on the above, "
        "the answer is number ("
    )

    response = ollama.generate(
        model=model_name,
        prompt=forced_prompt,
        raw=True,
        keep_alive=KEEP_ALIVE,
        options={
            "temperature": 0.0,
            "num_predict": 1,
            "num_ctx": NUM_CTX,
        },
        logprobs=True,
        top_logprobs=TOP_LOGPROBS,
    )

    # Hardened extraction layer
    logprobs = response.get("logprobs") or []

    if not logprobs:
        return None

    first_item = logprobs[0] or {}
    top_logprobs = first_item.get("top_logprobs") or []
    raw_probs = {}

    for entry in top_logprobs:
        if not entry:
            continue
        token_str = (entry.get("token") or "").strip()
        if token_str in answer_options:
            raw_probs[token_str] = math.exp(entry.get("logprob", -99.0))

    total = sum(raw_probs.values()) or 1e-9

    return {
        option: raw_probs.get(option, 0.0) / total
        for option in answer_options
    }


def probe_cut_point_openai_compatible(
    question_prompt: str,
    partial_reasoning: str,
    model_name: str,
    answer_options=("0", "1", "2"),
) -> dict:
    client = get_client(model_name)

    forced_prompt = (
        question_prompt
        + partial_reasoning
        + "\n\nBased on the above, "
        "the answer is number ("
    )

    try:
        logprobs = client.probe_logprobs(
            forced_prompt,
            top_logprobs=TOP_LOGPROBS,
        )
    except Exception as e:
        # A routing/provider failure on one cut point (e.g. no upstream
        # provider satisfies require_parameters for this request) shouldn't
        # take down the whole condition's result - degrade this single cut
        # to None, same as the existing "missing logprobs" case below.
        print(f"Probe request failed, treating this cut point as missing: {e}")
        return None

    content = (logprobs or {}).get("content") or []

    if not content:
        return None

    # Hardened extraction layer
    first_item = content[0] or {}
    top_logprobs = first_item.get("top_logprobs") or []
    raw_probs = {}

    for entry in top_logprobs:
        if not entry:
            continue
        token_str = (entry.get("token") or "").strip()
        if token_str in answer_options:
            raw_probs[token_str] = math.exp(entry.get("logprob", -99.0))

    total = sum(raw_probs.values()) or 1e-9

    return {
        option: raw_probs.get(option, 0.0) / total
        for option in answer_options
    }


def run_probe_on_item(
    question_prompt: str,
    model_name: str,
    num_cuts: int = 4,
    max_tokens: int = 2048,
    full_chain: str = None,
    effort: str = "medium",
):
    backend = get_model_profile(model_name)["backend"]

    if not full_chain:
        full_chain = generate_full_chain(
            question_prompt,
            model_name,
            max_tokens=max_tokens,
            effort=effort,
        )

    cut_points = get_cut_points(
        full_chain,
        num_cuts=num_cuts,
    )

    trajectory = [None] * len(cut_points)

    futures = {}

    for index, (frac, partial_reasoning) in enumerate(cut_points):
        if backend == "ollama":
            future = _PROBE_EXECUTOR.submit(
                probe_cut_point,
                question_prompt,
                partial_reasoning,
                model_name,
            )
        elif backend in ("openrouter", "azure"):
            future = _PROBE_EXECUTOR.submit(
                probe_cut_point_openai_compatible,
                question_prompt,
                partial_reasoning,
                model_name,
            )
        else:
            raise ValueError(
                f"Unsupported backend for probing: {backend}"
            )

        futures[future] = (index, frac)

    for future in as_completed(futures):
        index, frac = futures[future]
        trajectory[index] = (frac, future.result())

    commitment_point = find_commitment_point(trajectory)

    return full_chain, trajectory, commitment_point