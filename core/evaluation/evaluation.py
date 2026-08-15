import time

from core.config.config import (
    NATIVE_EFFORTS,
    BUDGETS,
    BUDGET_THINK_MODES,
    PROMPT_CONTROLS,
    PROBE_CUTS,
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

# --- Flip Rate Configurations ---
ENABLE_FLIP_RATE_EVAL = False 
FLIP_RATE_K = 3                


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

    for prompt_control in PROMPT_CONTROLS:
        conditions.append({
            "control_type": "prompt",
            "effort": None,
            "max_tokens": None,
            "prompt_control": prompt_control,
            "think": None,
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

    # Pass the prefix downstream to the client to pre-fill the reasoning sequence
    # (Client wrapper should appropriately inject this for Thought Anchor continuations)
    response: Qwen3Response = client.ask(
        prompt,
        effort=request_effort,
        max_tokens=max_tokens,
        prefix=prefix
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

    # Added extraction for evidence alignment and twin metadata
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
                else int(commit_answer) if commit_answer else None
            )
        else:
            result["answer_is_forced"] = False
            result["effective_answer"] = result["model_answer"]

        # --- Flip Rate (Commitment Robustness) Evaluation ---
        # Resample K continuations from the exact commitment prefix.
        # Invalid/unparseable continuations are excluded from the denominator.
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

                # For robustness, do not count a failed parse as a flip.
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

        return result

    except Exception as e:
        print(f"ERROR processing UID={example['uid']} with condition {condition_name}: {e}")
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
            "flip_rate": None,
        }