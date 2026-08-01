import time
import math
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import ollama

from client import Qwen3Client, Qwen3Response
from config import (
    MODEL,
    RESULTS_JSON,
    RESULTS_CSV,
    NATIVE_EFFORTS,
    BUDGETS,
    PROMPT_CONTROLS,
    BUDGET_THINK_MODES,
    DATASET_PATH,
    test_effort_conversion,
    test_ollama_conversion,
)
from util import (
    load_bbq,
    get_answer_metadata,
    format_prompt,
    parse_answer,
)
from metrics import (
    get_condition_name,
    print_summary,
    save_json,
    save_csv,
    get_cut_points,
    find_commitment_point,
)



MAX_EXAMPLES = 4

TASK_WORKERS = 4
PROBE_WORKERS = 2
PROBE_CUTS = 4
PROBE_MAX_TOKENS = 512
TOP_LOGPROBS = 5
CHECKPOINT_INTERVAL = 10

_thread_local = threading.local()


def get_client(model_name):
    if not hasattr(_thread_local, "client"):
        _thread_local.client = Qwen3Client(
            model=model_name
        )

    return _thread_local.client


def get_full_chain_max_tokens(condition):
    control_type = condition["control_type"]

    if control_type == "native_effort":
        effort = condition.get("effort")

        if effort == "low":
            return 256

        if effort == "medium":
            return 512

        if effort == "high":
            return 512

        return 512

    if control_type == "budget":
        max_tokens = condition.get("max_tokens")

        if max_tokens is None:
            return 512

        return min(
            max_tokens,
            PROBE_MAX_TOKENS,
        )

    if control_type == "prompt":
        prompt_control = condition.get(
            "prompt_control"
        )

        if prompt_control == "answer_immediately":
            return 256

        if prompt_control == "think_thoroughly":
            return 512

        return 512

    return 512


def evaluate_example(
    client: Qwen3Client,
    example,
    control_type,
    effort=None,
    max_tokens=None,
    prompt_control=None,
    think=None,
):
    prompt = format_prompt(
        example,
        prompt_control=prompt_control,
    )

    metadata = get_answer_metadata(example)

    valid_indices = [
        int(index)
        for index in example["answers"].keys()
    ]

    request_effort = effort

    if request_effort is None:
        if think is True:
            request_effort = "medium"
        else:
            request_effort = "off"

    start_time = time.perf_counter()

    response: Qwen3Response = client.ask(
        prompt,
        effort=request_effort,
        max_tokens=max_tokens,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    model_answer = parse_answer(
        response.content,
        valid_indices,
    )

    correct_answer = int(
        example["correct_answer"]
    )

    is_correct = (
        model_answer is not None
        and model_answer == correct_answer
    )

    selected_unknown = (
        model_answer is not None
        and model_answer
        == metadata["unknown_index"]
    )

    selected_stereotype = (
        model_answer is not None
        and model_answer
        == metadata["stereotype_index"]
    )

    selected_anti_stereotype = (
        model_answer is not None
        and model_answer
        == metadata["anti_stereotype_index"]
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


def build_conditions():
    conditions = []

    for effort in NATIVE_EFFORTS:
        conditions.append(
            {
                "control_type": "native_effort",
                "effort": effort,
                "max_tokens": None,
                "prompt_control": None,
                "think": None,
            }
        )

    for max_tokens in BUDGETS:
        for think in BUDGET_THINK_MODES:
            conditions.append(
                {
                    "control_type": "budget",
                    "effort": (
                        "medium"
                        if think
                        else "off"
                    ),
                    "max_tokens": max_tokens,
                    "prompt_control": None,
                    "think": think,
                }
            )

    for prompt_control in PROMPT_CONTROLS:
        conditions.append(
            {
                "control_type": "prompt",
                "effort": "on",
                "max_tokens": None,
                "prompt_control": prompt_control,
                "think": True,
            }
        )

    return conditions


def generate_full_chain(
    question_prompt,
    model_name,
    max_tokens=512,
):
    response = ollama.generate(
        model=model_name,
        prompt=question_prompt,
        raw=True,
        options={
            "temperature": 0.7,
            "num_predict": max_tokens,
        },
    )

    return response["response"]


def probe_cut_point(
    question_prompt,
    partial_reasoning,
    model_name,
    answer_options=("1", "2", "3"),
):
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
        options={
            "temperature": 0.0,
            "num_predict": 1,
        },
        logprobs=True,
        top_logprobs=TOP_LOGPROBS,
    )

    logprobs = response.get(
        "logprobs",
        []
    )

    if not logprobs:
        return {
            option: 0.0
            for option in answer_options
        }

    top_logprobs = logprobs[0].get(
        "top_logprobs",
        []
    )

    raw_probs = {}

    for entry in top_logprobs:
        token_str = entry["token"].strip()

        if token_str in answer_options:
            raw_probs[token_str] = math.exp(
                entry["logprob"]
            )

    total = sum(raw_probs.values())

    if total <= 0:
        return {
            option: 0.0
            for option in answer_options
        }

    return {
        option: raw_probs.get(
            option,
            0.0,
        ) / total
        for option in answer_options
    }


def run_probe_on_item(
    question_prompt,
    model_name,
    num_cuts=4,
    max_tokens=512,
):
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

    with ThreadPoolExecutor(
        max_workers=PROBE_WORKERS
    ) as executor:

        futures = {}

        for index, (
            frac,
            partial,
        ) in enumerate(cut_points):

            future = executor.submit(
                probe_cut_point,
                question_prompt,
                partial,
                model_name,
            )

            futures[future] = (
                index,
                frac,
            )

        for future in as_completed(futures):
            index, frac = futures[
                future
            ]

            probs = future.result()

            trajectory[index] = (
                frac,
                probs,
            )

    commitment_point = find_commitment_point(
        trajectory
    )

    return (
        full_chain,
        trajectory,
        commitment_point,
    )


def process_example_condition(
    example,
    condition,
    model_name,
):
    client = get_client(
        model_name
    )

    condition_name = get_condition_name(
        {
            "control_type": condition[
                "control_type"
            ],
            "effort": condition[
                "effort"
            ],
            "max_tokens": condition[
                "max_tokens"
            ],
            "prompt_control": condition[
                "prompt_control"
            ],
            "think": condition[
                "think"
            ],
        }
    )

    try:
        result = evaluate_example(
            client=client,
            example=example,
            control_type=condition[
                "control_type"
            ],
            effort=condition[
                "effort"
            ],
            max_tokens=condition[
                "max_tokens"
            ],
            prompt_control=condition[
                "prompt_control"
            ],
            think=condition[
                "think"
            ],
        )

        probe_prompt = format_prompt(
            example,
            prompt_control=condition[
                "prompt_control"
            ],
        )

        probing_max_tokens = (
            get_full_chain_max_tokens(
                condition
            )
        )

        (
            full_chain,
            trajectory,
            commitment,
        ) = run_probe_on_item(
            question_prompt=probe_prompt,
            model_name=model_name,
            num_cuts=PROBE_CUTS,
            max_tokens=probing_max_tokens,
        )

        commit_frac, commit_answer = (
            commitment
        )

        result[
            "probe_final_answer"
        ] = commit_answer

        result[
            "commitment_point_frac"
        ] = commit_frac

        result[
            "full_chain_generated"
        ] = full_chain

        result[
            "probe_trajectory"
        ] = trajectory

        return result

    except Exception as e:
        print(
            f"ERROR processing "
            f"UID={example['uid']} "
            f"condition={condition_name}: "
            f"{e}"
        )

        return {
            "uid": example["uid"],
            "control_type": condition[
                "control_type"
            ],
            "effort": condition[
                "effort"
            ],
            "max_tokens": condition[
                "max_tokens"
            ],
            "prompt_control": condition[
                "prompt_control"
            ],
            "think": condition[
                "think"
            ],
            "error": str(e),
        }


def save_checkpoint(results):
    try:
        save_json(
            results,
            RESULTS_JSON,
        )
    except Exception as e:
        print(
            f"Checkpoint save failed: {e}"
        )


def main():
    test_effort_conversion()
    test_ollama_conversion()

    print("good conversion")
    print()
    print("Loading BBQ dataset...")

    dataset = load_bbq(
        DATASET_PATH
    )

    original_count = len(dataset)

    if MAX_EXAMPLES is not None:
        dataset = dataset[
            :MAX_EXAMPLES
        ]

    print(
        f"Loaded {original_count} examples."
    )

    print(
        f"Using {len(dataset)} examples."
    )

    if len(dataset) % 2 != 0:
        print(
            "WARNING: dataset size is odd; "
            "the final pair may be incomplete."
        )

    ambiguous_count = sum(
        1
        for example in dataset
        if example[
            "context_condition"
        ] == "ambig"
    )

    disambiguated_count = sum(
        1
        for example in dataset
        if example[
            "context_condition"
        ] == "disambig"
    )

    print(
        f"Ambiguous examples: "
        f"{ambiguous_count}"
    )

    print(
        f"Disambiguated examples: "
        f"{disambiguated_count}"
    )

    conditions = build_conditions()

    total_tasks = (
        len(dataset)
        * len(conditions)
    )

    calls_per_task = (
        2 + PROBE_CUTS
    )

    estimated_calls = (
        total_tasks
        * calls_per_task
    )

    print()
    print(
        f"Conditions: "
        f"{len(conditions)}"
    )

    print(
        f"Task workers: "
        f"{TASK_WORKERS}"
    )

    print(
        f"Probe workers: "
        f"{PROBE_WORKERS}"
    )

    print(
        f"Probe cuts: "
        f"{PROBE_CUTS}"
    )

    print(
        f"Probe max tokens: "
        f"{PROBE_MAX_TOKENS}"
    )

    print(
        f"Top logprobs: "
        f"{TOP_LOGPROBS}"
    )

    print(
        f"Total tasks: "
        f"{total_tasks}"
    )

    print(
        f"Estimated model calls: "
        f"{estimated_calls}"
    )

    print()

    all_results = []

    tasks = [
        (
            example,
            condition,
        )
        for example in dataset
        for condition in conditions
    ]

    tasks_completed = 0

    start_time = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=TASK_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                process_example_condition,
                example,
                condition,
                MODEL,
            ): (
                example,
                condition,
            )
            for example, condition in tasks
        }

        for future in as_completed(
            futures
        ):
            tasks_completed += 1

            example, condition = (
                futures[future]
            )

            try:
                result = future.result()

                all_results.append(
                    result
                )

            except Exception as e:
                print(
                    f"Worker error for "
                    f"{example['uid']}: "
                    f"{e}"
                )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            average_time = (
                elapsed
                / tasks_completed
            )

            remaining = (
                total_tasks
                - tasks_completed
            )

            eta = (
                average_time
                * remaining
            )

            progress = (
                tasks_completed
                / total_tasks
                * 100
            )

            print(
                f"Progress: "
                f"{tasks_completed}/"
                f"{total_tasks} "
                f"({progress:.1f}%) "
                f"| elapsed "
                f"{elapsed / 60:.1f}m "
                f"| ETA "
                f"{eta / 60:.1f}m"
            )

            if (
                tasks_completed
                % CHECKPOINT_INTERVAL
                == 0
            ):
                save_checkpoint(
                    all_results
                )

    total_elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print(
        f"Finished processing "
        f"{len(all_results)} results."
    )

    print(
        f"Total runtime: "
        f"{total_elapsed / 60:.2f} minutes"
    )

    RESULTS_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        all_results,
        RESULTS_JSON,
    )

    save_csv(
        all_results,
        RESULTS_CSV,
    )

    print(
        f"JSON saved to: "
        f"{RESULTS_JSON}"
    )

    print(
        f"CSV saved to: "
        f"{RESULTS_CSV}"
    )

    print()

    print_summary(
        all_results
    )


if __name__ == "__main__":
    main()