import time
import os
import sys
import subprocess
import threading

import httpx

from core.config.config import (
    NATIVE_EFFORTS,
    BUDGETS,
    BUDGET_THINK_MODES,
    OPENROUTER_NATIVE_EFFORTS,
    PROBE_CUTS,
    ENABLE_FLIP_RATE_EVAL,
    FLIP_RATE_K,

)

import core.config.config as config
from core.clients.olllama_client import Qwen3Response
from core.clients.clients import get_client
from core.utility.util import get_answer_metadata, format_prompt, parse_answer
from core.evaluation.metrics import (
    get_condition_name,
    keyword_terms,
    find_mentions,

)
from core.evaluation.probes import run_probe_on_item





OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "5"))
OLLAMA_RETRY_BASE_SECONDS = float(os.getenv("OLLAMA_RETRY_BASE_SECONDS", "2"))
OLLAMA_HEALTH_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_HEALTH_TIMEOUT_SECONDS", "5"))
OLLAMA_HEALTH_URL = os.getenv("OLLAMA_HEALTH_URL", "http://127.0.0.1:11434/api/tags")
_OLLAMA_RECOVERY_LOCK = threading.Lock()


def _is_retryable_ollama_error(exc: Exception) -> bool:

    text = str(exc).lower()

    retryable_text = (
        "failed to connect to ollama",
        "server disconnected",
        "connection refused",
        "connection reset",
        "connection aborted",
        "remote protocol error",
        "server disconnected without sending a response",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "502",
        "503",
        "504",
    )
    if any(term in text for term in retryable_text):
        return True

    return isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadError,
            httpx.ReadTimeout,
            httpx.WriteError,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
        ),
    )


def _ollama_healthy() -> bool:
    try:
        response = httpx.get(
            OLLAMA_HEALTH_URL,
            timeout=OLLAMA_HEALTH_TIMEOUT_SECONDS,
        )
        return response.status_code == 200
    except Exception:
        return False


def _restart_ollama() -> None:

    with _OLLAMA_RECOVERY_LOCK:
        if _ollama_healthy():
            return

        print("Ollama health check failed; attempting local recovery...", flush=True)

        try:
            if sys.platform == "darwin":
  
                subprocess.Popen(
                    ["open", "-a", "Ollama"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            else:
                # On Linux, start the Ollama daemon if it is down.
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
        except Exception as recovery_error:
            print(
                f"Ollama restart attempt failed: {type(recovery_error).__name__}: "
                f"{recovery_error}",
                flush=True,
            )


        for _ in range(OLLAMA_MAX_RETRIES):
            time.sleep(2)
            if _ollama_healthy():
                print("Ollama recovered.", flush=True)
                return

        print("Ollama is still unavailable after recovery attempt.", flush=True)


def _ask_with_retries(client, prompt, *, effort, max_tokens, prefix=None):

    last_error = None

    for attempt in range(OLLAMA_MAX_RETRIES + 1):
        try:
            return client.ask(
                prompt,
                effort=effort,
                max_tokens=max_tokens,
                prefix=prefix,
            )
        except Exception as exc:
            last_error = exc
            if not _is_retryable_ollama_error(exc) or attempt >= OLLAMA_MAX_RETRIES:
                raise

            delay = OLLAMA_RETRY_BASE_SECONDS * (2 ** attempt)
            print(
                f"Ollama request failed (attempt {attempt + 1}/{OLLAMA_MAX_RETRIES + 1}) "
                f"with {type(exc).__name__}: {exc}. "
                f"Recovering/retrying in {delay:.1f}s...",
                flush=True,
            )
            _restart_ollama()
            time.sleep(delay)

    raise last_error


def _run_probe_with_retries(**kwargs):

    last_error = None

    for attempt in range(OLLAMA_MAX_RETRIES + 1):
        try:
            return run_probe_on_item(**kwargs)
        except Exception as exc:
            last_error = exc
            if not _is_retryable_ollama_error(exc) or attempt >= OLLAMA_MAX_RETRIES:
                raise

            delay = OLLAMA_RETRY_BASE_SECONDS * (2 ** attempt)
            print(
                f"Ollama probe failed (attempt {attempt + 1}/{OLLAMA_MAX_RETRIES + 1}) "
                f"with {type(exc).__name__}: {exc}. "
                f"Recovering/retrying in {delay:.1f}s...",
                flush=True,
            )
            _restart_ollama()
            time.sleep(delay)

    raise last_error


def get_full_chain_max_tokens(condition: dict) -> int:
    control_type = condition["control_type"]
    if control_type == "native_effort":
        effort = condition.get("effort")
        if effort == "low": return 512
        if effort == "medium": return 1024
        if effort == "high": return 2048
        return 1024

    if control_type == "budget":
        max_tokens = condition.get("max_tokens")
        return 1024 if max_tokens is None else max_tokens


    return 1024


def build_conditions(model_name) -> list:
    conditions = []
    if model_name != "kimi-k2.6":
        for effort in NATIVE_EFFORTS:
            conditions.append({
                "control_type": "native_effort",
                "effort": effort,
                "max_tokens": None,
                "prompt_control": None,
                "think": None,
            })
    else:
        for effort in OPENROUTER_NATIVE_EFFORTS:
            conditions.append({
                "control_type": "toggle",
                "effort": None,
                "max_tokens": None,
                "prompt_control": None,
                "think": effort,
            })

    for max_tokens in BUDGETS:
        for think in BUDGET_THINK_MODES:
            conditions.append({
                "control_type": "budget",
                "effort": "medium" if think else "off",
                "max_tokens": max_tokens,
                "prompt_control": None,
                "think": think,
            })



    return conditions


def evaluate_example(
    client,
    example: dict,
    control_type: str,
    effort: str = None,
    max_tokens: int = None,
    prompt_control: str = None,
    think: bool = None,
    prefix: str = None,
) -> dict:
    prompt = format_prompt(example, prompt_control=prompt_control)
    metadata = get_answer_metadata(example)
    valid_indices = [int(index) for index in example["answers"].keys()]
    
    request_effort = effort
    if request_effort is None:
        request_effort = "medium" if think is True else "off"

    start_time = time.perf_counter()


    response: Qwen3Response = _ask_with_retries(
        client,
        prompt,
        effort=request_effort,
        max_tokens=max_tokens,
        prefix=prefix,
    )

    elapsed = time.perf_counter() - start_time

    model_answer = parse_answer(response.content, valid_indices)
    correct_answer = int(example["correct_answer"])

    is_correct = model_answer is not None and model_answer == correct_answer

    selected_unknown = (
        model_answer is not None
        and model_answer == metadata["unknown_index"]
    )

    selected_stereotype = (
        model_answer is not None
        and model_answer == metadata["stereotype_index"]
    )

    selected_anti_stereotype = (
        model_answer is not None
        and model_answer == metadata["anti_stereotype_index"]
    )


    evidence_alignment = example.get("evidence_allignment", example.get("evidence_alignment"))

    return {
        "uid": example["uid"],
        "category": example["category"],
        "subcategory": example["subcategory"],
        "question_index": example["question_index"],
        "question_polarity": example["question_polarity"],
        "context_condition": example["context_condition"],
        "model": response.model,
        "control_type": control_type,
        "effort": effort,
        "max_tokens": max_tokens,
        "prompt_control": prompt_control,
        "think": think,
        "model_answer": model_answer,
        "correct_answer": correct_answer,
        "is_correct": is_correct,
        "selected_unknown": selected_unknown,
        "selected_stereotype": selected_stereotype,
        "selected_anti_stereotype": selected_anti_stereotype,
        "unknown_index": metadata["unknown_index"],
        "stereotype_index": metadata["stereotype_index"],
        "anti_stereotype_index": metadata["anti_stereotype_index"],
        "content": response.content,
        "thinking": response.thinking,
        "thinking_chars": response.thinking_chars,
        "latency_seconds": elapsed,
        "evidence_alignment": evidence_alignment,
        "is_twin": example.get("is_twin", False),
        "twin_partner_uid": example.get("twin_partner_uid"),
        "twin_side": example.get("twin_side")
    }


def process_example_condition(
    example: dict,
    condition: dict,
    model_name: str,
) -> dict:
    client = get_client(model_name)
    condition_name = get_condition_name({
        "control_type": condition["control_type"],
        "effort": condition["effort"],
        "max_tokens": condition["max_tokens"],
        "prompt_control": condition["prompt_control"],
        "think": condition["think"],
    })

    try:
        result = evaluate_example(
            client=client,
            example=example,
            control_type=condition["control_type"],
            effort=condition["effort"],
            max_tokens=condition["max_tokens"],
            prompt_control=condition["prompt_control"],
            think=condition["think"],
        )

        thinking_text = result.get("thinking") or ""

        stereotype_terms = keyword_terms(example, result["stereotype_index"])
        anti_stereotype_terms = keyword_terms(example, result["anti_stereotype_index"])
        
        stereotype_mentions = find_mentions(thinking_text, stereotype_terms)
        anti_stereotype_mentions = find_mentions(thinking_text, anti_stereotype_terms)

        result["stereotype_mentions"] = stereotype_mentions
        result["anti_stereotype_mentions"] = anti_stereotype_mentions
        result["first_stereotype_mention_pct"] = (
            stereotype_mentions[0]["pct_through_reasoning"] if stereotype_mentions else None
        )
        result["first_anti_stereotype_mention_pct"] = (
            anti_stereotype_mentions[0]["pct_through_reasoning"] if anti_stereotype_mentions else None
        )
        result["stereotype_mention_count"] = len(stereotype_mentions)
        result["anti_stereotype_mention_count"] = len(anti_stereotype_mentions)

        probe_prompt = format_prompt(example, prompt_control=condition["prompt_control"])
        probing_max_tokens = get_full_chain_max_tokens(condition)


        probe_effort = condition["effort"]
        if probe_effort is None:
            probe_effort = "medium" if condition["think"] else "off"

        (
            full_chain,
            trajectory,
            commitment_point,
        ) = _run_probe_with_retries(
            question_prompt=probe_prompt,
            model_name=model_name,
            num_cuts=PROBE_CUTS,
            max_tokens=probing_max_tokens,
            full_chain=result.get("thinking"),
            effort=probe_effort,
        )

        commit_frac, commit_answer = commitment_point
        
        result["probe_final_answer"] = commit_answer
        result["commitment_point_frac"] = commit_frac
        result["commitment_depth_chars"] = (
            int(len(full_chain) * commit_frac)
            if full_chain and commit_frac is not None
            else None
        )
        result["full_chain_generated"] = full_chain
        result["probe_trajectory"] = trajectory

        if config.ENABLE_FORCED_ANSWER:
            result["answer_is_forced"] = result["model_answer"] is None
            result["effective_answer"] = (
                result["model_answer"]
                if result["model_answer"] is not None
                else int(commit_answer) if commit_answer is not None else None
            )
        else:
            result["answer_is_forced"] = False
            result["effective_answer"] = result["model_answer"]


        result["model_answer"] = result["effective_answer"]


        result["is_correct"] = (
            result["model_answer"] is not None
            and result["model_answer"] == result["correct_answer"]
        )


        result["selected_unknown"] = (
            result["model_answer"] is not None
            and result["model_answer"] == result["unknown_index"]
        )

        result["selected_stereotype"] = (
            result["model_answer"] is not None
            and result["model_answer"] == result["stereotype_index"]
        )

        result["selected_anti_stereotype"] = (
            result["model_answer"] is not None
            and result["model_answer"] == result["anti_stereotype_index"]
        )

        result["flip_rate"] = None
        result["flip_flips"] = 0
        result["flip_valid_resamples"] = 0
        result["flip_invalid_resamples"] = 0
        result["flip_k"] = FLIP_RATE_K if ENABLE_FLIP_RATE_EVAL else 0

        if (
            ENABLE_FLIP_RATE_EVAL
            and full_chain
            and commit_frac is not None
            and result.get("effective_answer") is not None
        ):
            prefix_length = int(len(full_chain) * commit_frac)
            reasoning_prefix = full_chain[:prefix_length]

            for _ in range(FLIP_RATE_K):
                resample = evaluate_example(
                    client=client,
                    example=example,
                    control_type=condition["control_type"],
                    effort=condition["effort"],
                    max_tokens=condition["max_tokens"],
                    prompt_control=condition["prompt_control"],
                    think=condition["think"],
                    prefix=reasoning_prefix,
                )

                resampled_answer = resample.get("model_answer")


                if resampled_answer is None:
                    result["flip_invalid_resamples"] += 1
                    continue

                result["flip_valid_resamples"] += 1
                if resampled_answer != result["effective_answer"]:
                    result["flip_flips"] += 1

            if result["flip_valid_resamples"] > 0:
                result["flip_rate"] = (
                    result["flip_flips"] / result["flip_valid_resamples"]
                )

        result["status"] = "success"
        result["error"] = None
        return result

    except Exception as e:
        print(
            f"ERROR processing UID={example['uid']} "
            f"category={example.get('category')} "
            f"context={example.get('context_condition')} "
            f"condition={condition_name} "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        return {
            "uid": example["uid"],
            "category": example.get("category"),
            "subcategory": example.get("subcategory"),
            "question_index": example.get("question_index"),
            "question_polarity": example.get("question_polarity"),
            "context_condition": example.get("context_condition"),
            "model": model_name,
            "control_type": condition["control_type"],
            "effort": condition["effort"],
            "max_tokens": condition["max_tokens"],
            "prompt_control": condition["prompt_control"],
            "think": condition["think"],
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "flip_rate": None,
        }