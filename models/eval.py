import time
from client import Qwen3Client
from config import (
    MODEL, DATASET_PATH, RESULTS_JSON, RESULTS_CSV, NATIVE_EFFORTS, BUDGETS, 
    PROMPT_CONTROLS, BUDGET_THINK_MODES, test_effort_conversion, test_ollama_conversion
)
from util import load_bbq, get_answer_metadata, format_prompt, parse_answer
from metrics import get_condition_name, print_summary, save_json, save_csv
import math
import ollama

def evaluate_example(client, example, control_type, effort=None, max_tokens=None, prompt_control=None, think=None):
    prompt = format_prompt(example, prompt_control=prompt_control)
    metadata = get_answer_metadata(example)
    valid_indices = [int(index) for index in example["answers"].keys()]

    request_effort = effort
    if request_effort is None:
        if think is True:
            request_effort = "medium"
        else:
            request_effort = "off"

    start_time = time.perf_counter()
    response = client.ask(prompt, effort=request_effort, max_tokens=max_tokens)
    elapsed = time.perf_counter() - start_time

    model_answer = parse_answer(response.content, valid_indices)
    correct_answer = int(example["correct_answer"])

    is_correct = (model_answer is not None and model_answer == correct_answer)
    selected_unknown = (model_answer is not None and model_answer == metadata["unknown_index"])
    selected_stereotype = (model_answer is not None and model_answer == metadata["stereotype_index"])
    selected_anti_stereotype = (model_answer is not None and model_answer == metadata["anti_stereotype_index"])

    result = {
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
    return result

def build_conditions():
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
            "effort": "on",
            "max_tokens": None,
            "prompt_control": prompt_control,
            "think": True,
        })

    return conditions


def generate_full_chain(question_prompt, model_name):
    response = ollama.generate(
        model=model_name,
        prompt=question_prompt,
        raw=True,
        options={
            "temperature": 0.7,
            "num_predict": 2048,
        },
    )
    return response["response"]

def probe_cut_point(question_prompt, partial_reasoning, model_name, answer_options=["1", "2", "3"]):
    forced_prompt = (
        question_prompt
        + partial_reasoning
        + "\n\nBased on the above, the answer is number ("
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
        top_logprobs=20,
    )

    top_logprobs = response["logprobs"][0]["top_logprobs"]
    raw_probs = {}
    
    for entry in top_logprobs:
        token_str = entry["token"].strip()
        if token_str in answer_options:
            raw_probs[token_str] = math.exp(entry["logprob"])

    total = sum(raw_probs.values()) or 1e-9
    normalized = {opt: raw_probs.get(opt, 0.0) / total for opt in answer_options}
    return normalized

def run_probe_on_item(question_prompt, model_name, num_cuts=15):
    from metrics import get_cut_points, find_commitment_point
    
    full_chain = generate_full_chain(question_prompt, model_name)
    cut_points = get_cut_points(full_chain, num_cuts=num_cuts)

    trajectory = []
    for frac, partial in cut_points:
        probs = probe_cut_point(question_prompt, partial, model_name)
        trajectory.append((frac, probs))

    commitment_point = find_commitment_point(trajectory)
    return full_chain, trajectory, commitment_point

def check_stability(question_prompt, model_name, num_runs=5, num_cuts=15):
    from metrics import summarize_stability
    
    results = []
    for run_num in range(num_runs):
        _, _, (commit_frac, commit_answer) = run_probe_on_item(
            question_prompt, model_name, num_cuts=num_cuts
        )
        results.append((commit_frac, commit_answer))

    return summarize_stability(results)

def main():
    test_effort_conversion()
    test_ollama_conversion()
    print("good conversion")
    
    print("Loading BBQ dataset...")
    dataset = load_bbq(DATASET_PATH)
    print(f"Loaded {len(dataset)} examples.")

    ambiguous_count = sum(1 for example in dataset if example["context_condition"] == "ambig")
    disambiguated_count = sum(1 for example in dataset if example["context_condition"] == "disambig")

    print(f"Ambiguous examples: {ambiguous_count}")
    print(f"Disambiguated examples: {disambiguated_count}")

    client = Qwen3Client(model=MODEL)
    conditions = build_conditions()
    total_runs = len(dataset) * len(conditions)

    print(f"\nTotal model calls: {total_runs}")
    
    results = []
    current_run = 0

    for example in dataset:
        for condition in conditions:
            current_run += 1
            condition_name = get_condition_name({
                "control_type": condition["control_type"],
                "effort": condition["effort"],
                "max_tokens": condition["max_tokens"],
                "prompt_control": condition["prompt_control"],
                "think": condition["think"],
            })

            print(f"\n[{current_run}/{total_runs}] UID={example['uid']} Condition={example['context_condition']} Control={condition_name}")

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
                results.append(result)

                probe_prompt = format_prompt(example, prompt_control=condition["prompt_control"])
                
                print("Finding commitment point")
                full_chain, trajectory, (commit_frac, commit_answer) = run_probe_on_item(
                    question_prompt=probe_prompt, 
                    model_name=MODEL, 
                    num_cuts=3
                )

                print(f"Model answer: {result['model_answer']}")
                print(f"Correct answer: {result['correct_answer']}")
                print(f"Probe final answer: {commit_answer}")
                print(f"Commitment point: {commit_frac*100:.1f}% through the reasoning chain")
                print(f"Thinking chars: {result['thinking_chars']}")
                print(f"Latency: {result['latency_seconds']:.2f}s")
                print("-" * 60)

            except Exception as e:
                print(f"ERROR evaluating {example['uid']} with condition {condition_name}:")
                print(e)

    save_json(results, RESULTS_JSON)
    save_csv(results, RESULTS_CSV)

    print(f"\nJSON saved to: {RESULTS_JSON}")
    print(f"CSV saved to: {RESULTS_CSV}")
    print_summary(results)

if __name__ == "__main__":
    main()