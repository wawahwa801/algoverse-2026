import time
from core.config.config import NATIVE_EFFORTS, BUDGETS, BUDGET_THINK_MODES, PROBE_CUTS
# Imported as a module, not "from ... import ENABLE_FORCED_ANSWER" - that
# would snapshot the value at import time, so an external script setting
# core.config.config.ENABLE_FORCED_ANSWER = False before a run (see
# models/run_smoke_sample_bigbudget.py for the equivalent pattern) wouldn't
# be seen here. Read as core.config.config.ENABLE_FORCED_ANSWER at the
# point of use instead, so the toggle stays live.
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


def get_full_chain_max_tokens(condition: dict) -> int:
    control_type = condition["control_type"]

    if control_type == "native_effort":
        effort = condition.get("effort")
        if effort == "low":
            return 256
        if effort == "medium":
            return 512
        if effort == "high":
            return 1024
        return 512

    if control_type == "budget":
        max_tokens = condition.get("max_tokens")
        if max_tokens is None:
            return 512
        return max_tokens

    if control_type == "prompt":
        prompt_control = condition.get("prompt_control")
        if prompt_control == "answer_immediately":
            return 256
        if prompt_control == "think_thoroughly":
            return 1024
        return 512

    return 512


def build_conditions() -> list:
    conditions = []

    for effort in NATIVE_EFFORTS:
        conditions.append({
            "control_type": "native_effort",
            "effort": effort,
            "max_tokens": None,
            "prompt_control": None,
            "think": None,
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
) -> dict:
    prompt = format_prompt(example, prompt_control=prompt_control)
    metadata = get_answer_metadata(example)
    valid_indices = [int(index) for index in example["answers"].keys()]

    request_effort = effort
    if request_effort is None:
        request_effort = "medium" if think is True else "off"

    start_time = time.perf_counter()

    response: Qwen3Response = client.ask(
        prompt,
        effort=request_effort,
        max_tokens=max_tokens,
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

        stereotype_terms = keyword_terms(
            example, result["stereotype_index"]
        )
        anti_stereotype_terms = keyword_terms(
            example, result["anti_stereotype_index"]
        )
        stereotype_mentions = find_mentions(
            thinking_text, stereotype_terms
        )
        anti_stereotype_mentions = find_mentions(
            thinking_text, anti_stereotype_terms
        )

        result["stereotype_mentions"] = stereotype_mentions
        result["anti_stereotype_mentions"] = anti_stereotype_mentions
        result["first_stereotype_mention_pct"] = (
            stereotype_mentions[0]["pct_through_reasoning"]
            if stereotype_mentions else None
        )
        result["first_anti_stereotype_mention_pct"] = (
            anti_stereotype_mentions[0]["pct_through_reasoning"]
            if anti_stereotype_mentions else None
        )

        probe_prompt = format_prompt(
            example,
            prompt_control=condition["prompt_control"],
        )

        probing_max_tokens = get_full_chain_max_tokens(condition)

        (
            full_chain,
            trajectory,
            commitment_point,
        ) = run_probe_on_item(
            question_prompt=probe_prompt,
            model_name=model_name,
            num_cuts=PROBE_CUTS,
            max_tokens=probing_max_tokens,
            full_chain=result.get("thinking"),
        )

        commit_frac, commit_answer = commitment_point

        result["probe_final_answer"] = commit_answer
        result["commitment_point_frac"] = commit_frac
        result["full_chain_generated"] = full_chain
        result["probe_trajectory"] = trajectory

        # Budget-capped tasks can run out of tokens mid-reasoning, leaving
        # model_answer=None (no room left to state a digit). The probe's
        # last cut point (100% of whatever reasoning was actually produced,
        # complete or truncated) already forces exactly this answer via a
        # logprob-argmax completion - reuse it instead of a new call.
        # model_answer itself is left untouched so "the model naturally
        # said X" and "we had to force it" stay distinguishable.
        if config.ENABLE_FORCED_ANSWER:
            result["answer_is_forced"] = result["model_answer"] is None

            result["effective_answer"] = (
                result["model_answer"]
                if result["model_answer"] is not None
                # commit_answer is a string ("0"/"1"/"2", a probe dict key) -
                # model_answer is an int (from parse_answer). Cast so
                # effective_answer is consistently typed regardless of
                # source.
                else int(commit_answer)
            )
        else:
            result["answer_is_forced"] = False
            result["effective_answer"] = result["model_answer"]

        return result

    except Exception as e:
        print(
            f"ERROR processing UID={example['uid']} "
            f"with condition {condition_name}: {e}"
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
            "error": str(e),
        }