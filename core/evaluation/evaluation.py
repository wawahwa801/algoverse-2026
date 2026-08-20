import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# Long-lived, bounded pool for flip-rate resampling, shared across the whole
# run for the same reason probes.py's _PROBE_EXECUTOR is: a per-call
# executor would leak a fresh thread-local client per resample. The K
# resamples for one item are independent of each other, so running them
# concurrently (instead of the previous plain `for` loop) turns K sequential
# full-length generations into ~1, which matters a lot once K=FLIP_RATE_K
# full generations were the dominant per-task cost.
_RESAMPLE_EXECUTOR = (
    ThreadPoolExecutor(max_workers=max(FLIP_RATE_K, 1))
    if ENABLE_FLIP_RATE_EVAL
    else None
)




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

def get_max_tokens(condition: dict) -> int:
    control_type = condition["control_type"]
    if control_type == "native_effort":
        effort = condition.get("effort")
        if effort == "low": return 3000
        if effort == "medium": return 8000
        if effort == "high": return 15000
        return 10000
        
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

    request_max_tokens = max_tokens
    if request_max_tokens is None:
        # native_effort conditions (and Kimi's think on/off toggle) don't
        # set an explicit token budget, which previously meant the main
        # generation call ran fully uncapped. A single long high-effort
        # completion can tie up a parallel serving slot for an
        # unpredictable amount of time - hurting both GPU utilization and
        # run-time variance. Reuse the same effort->token mapping already
        # used for probe fallback generation so the main call gets a sane
        # cap too. This does NOT change the recorded `max_tokens` condition
        # field below (still None) - that field means "no explicit budget
        # by design", not "what was actually sent to the client".
        request_max_tokens = get_max_tokens({
            "control_type": control_type,
            "effort": effort,
        })

        
    start_time = time.perf_counter()

    # Pass the prefix downstream to the client to pre-fill the reasoning sequence
    # (Client wrapper should appropriately inject this for Thought Anchor continuations)
    response: Qwen3Response = client.ask(
        prompt,
        effort=request_effort,
        max_tokens=request_max_tokens,
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

    # Probing, commitment-point analysis, canonical-answer promotion, and
    # flip-rate resampling are all "enrichment" on top of the primary
    # answer above. A failure anywhere in here (e.g. a probe request
    # hitting an OpenRouter routing error) should degrade those fields to
    # None rather than discarding the already-successful primary result -
    # otherwise a single flaky probe call silently throws away a good
    # model_answer/is_correct and replaces it with a bare error record.
    try:
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

        # Same effort resolution evaluate_example() uses, so a fallback
        # full-chain generation (when there's no thinking to reuse) requests
        # the same reasoning level the condition actually asked for, instead
        # of silently forcing reasoning back on.
        probe_effort = condition["effort"]
        if probe_effort is None:
            probe_effort = "medium" if condition["think"] else "off"

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

        # The effective/probe answer is the canonical answer used for
        # evaluation - promote it into model_answer and recompute every
        # field derived from it, so evaluation.py -> CSV -> metrics.py all
        # agree on a single canonical answer.
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

            def _resample():
                # Each worker thread gets its own thread-local client
                # rather than reusing the calling thread's `client`, same
                # pattern as get_client() everywhere else - avoids sharing
                # one client instance across concurrent threads.
                resample_client = get_client(model_name)
                return evaluate_example(
                    client=resample_client,
                    example=example,
                    control_type=condition["control_type"],
                    effort=condition["effort"],
                    max_tokens=condition["max_tokens"],
                    prompt_control=condition["prompt_control"],
                    think=condition["think"],
                    prefix=reasoning_prefix,
                )

            resample_futures = [
                _RESAMPLE_EXECUTOR.submit(_resample) for _ in range(FLIP_RATE_K)
            ]

            for future in as_completed(resample_futures):
                try:
                    resample = future.result()
                except Exception as e:
                    # One flaky resample shouldn't drop the rest of the
                    # flip-rate loop - count it as invalid and move on.
                    print(
                        f"Flip-rate resample failed for UID={example['uid']} "
                        f"condition {condition_name}: {e}"
                    )
                    result["flip_invalid_resamples"] += 1
                    continue

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

    except Exception as e:
        print(
            f"Probe/commitment step failed for UID={example['uid']} "
            f"condition {condition_name}: {e}; keeping primary answer, "
            "degrading probe-derived fields to None."
        )
        result.setdefault("probe_final_answer", None)
        result.setdefault("commitment_point_frac", None)
        result.setdefault("commitment_depth_chars", None)
        result.setdefault("full_chain_generated", None)
        result.setdefault("probe_trajectory", None)
        result.setdefault("answer_is_forced", False)
        result.setdefault("effective_answer", result.get("model_answer"))
        result.setdefault("flip_rate", None)
        result.setdefault("flip_flips", 0)
        result.setdefault("flip_valid_resamples", 0)
        result.setdefault("flip_invalid_resamples", 0)
        result.setdefault("flip_k", FLIP_RATE_K if ENABLE_FLIP_RATE_EVAL else 0)
        result["probe_error"] = str(e)

    return result