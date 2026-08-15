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


def generate_full_chain(
    question_prompt: str,
    model_name: str,
    max_tokens: int = 2048,
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
        effort="medium",
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

    logprobs = response.get("logprobs", [])

    if not logprobs:
        return None

    top_logprobs = logprobs[0].get("top_logprobs", [])
    raw_probs = {}

    for entry in top_logprobs:
        token_str = entry.get("token", "").strip()
        if token_str in answer_options:
            raw_probs[token_str] = math.exp(entry["logprob"])

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

    logprobs = client.probe_logprobs(
        forced_prompt,
        top_logprobs=TOP_LOGPROBS,
    )

    content = (logprobs or {}).get("content") or []

    if not content:
        return None

    top_logprobs = content[0].get("top_logprobs", [])
    raw_probs = {}

    for entry in top_logprobs:
        token_str = (entry.get("token") or "").strip()
        if token_str in answer_options:
            raw_probs[token_str] = math.exp(entry["logprob"])

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
):
    backend = get_model_profile(model_name)["backend"]

    if not full_chain:
        full_chain = generate_full_chain(
            question_prompt,
            model_name,
            max_tokens=max_tokens,
        )

    cut_points = get_cut_points(
        full_chain,
        num_cuts=num_cuts,
    )

    trajectory = [None] * len(cut_points)

    with ThreadPoolExecutor(max_workers=PROBE_WORKERS) as executor:
        futures = {}

        for index, (frac, partial_reasoning) in enumerate(cut_points):
            if backend == "ollama":
                future = executor.submit(
                    probe_cut_point,
                    question_prompt,
                    partial_reasoning,
                    model_name,
                )
            elif backend in ("openrouter", "azure"):
                future = executor.submit(
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
