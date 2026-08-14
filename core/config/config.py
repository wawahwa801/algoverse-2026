from core.config.effort import ReasoningEffort
from pathlib import Path
import os

MODEL = "kimi-k2.6"

# Per-model backend routing (models/eval.py::get_client) and result paths,
# so switching MODEL and rerunning never overwrites another model's output.
# Models absent from this registry default to the Ollama backend using the
# model name as-is (preserves the original single-model behavior).
# NOTE: api_key values are intentionally blank - a real key was found
# committed in the algoverse-2026 clone this was merged from (rotate that
# key). Set via environment / local-only config, never commit a real one.
MODEL_PROFILES = {
    "qwen3:4b": {"backend": "ollama", "model_id": "qwen3:4b"},
    "qwen3.5:9b": {"backend": "ollama", "model_id": "qwen3.5:9b"},
    "gpt-oss:20b": {"backend": "ollama", "model_id": "gpt-oss:20b"},
    "kimi-k2.6": {
        "backend": "azure",
        "model_id": "kimi-k2.6",
<<<<<<< Updated upstream
        "endpoint_url":  "https://algoverseproject.services.ai.azure.com/openai/v1",
=======
        "endpoint_url": "",
>>>>>>> Stashed changes
        "api_key": "",
    },
    "deepseek-v4-pro": {
        "backend": "azure",
        "model_id": "deepseek-v4-pro",
        "endpoint_url": "",
        "api_key": "",
    },
    # Kept available even though not in the active roster - OpenRouterModelClient
    # still exists and works, costs nothing to leave wired in.
    "grok-4.5": {"backend": "openrouter", "model_id": "xai/grok-4.5", "api_key": ""},
}


def _model_slug(model_name):
    return model_name.replace(":", "-").replace("/", "-")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "core" / "results"
RESULTS_JSON = RESULTS_DIR / f"bbq_results_{_model_slug(MODEL)}.json"
RESULTS_CSV = RESULTS_DIR / f"bbq_results_{_model_slug(MODEL)}.csv"

NATIVE_EFFORTS = ["low", "medium", "high"]
BUDGETS = [2048, 4096, 8192]
# Empty by default so the main sweep stays budget/native-effort only; set to
# e.g. ["answer_immediately", "think_thoroughly"] to re-enable prompt
# conditions in build_conditions().
PROMPT_CONTROL_OPTIONS = []
PROMPT_CONTROLS = []
BUDGET_THINK_MODES = [True]


DATA_ROOT = PROJECT_ROOT / "data"
# The frozen, opposite-alignment-filtered subset with matched ambiguous
# siblings (data/build_clean_pairs_subset.py) - same canonical dataset
# models/eval.py uses, so core/ and models/ never diverge on what's "the"
# eval set. Replaces the old unfiltered bbq_pairs_subset.jsonl and the
# tiny hand-built core/bbq_subset.jsonl (2 rows, no alignment filter, no
# ambig siblings).
PAIRS_DATASET_PATH = DATA_ROOT / "bbq_pairs_subset_700_clean.jsonl"
CLEAN_DATASET_PATH = DATA_ROOT / "bbq_clean.jsonl"
# Legacy hand-sampled subset path (models/config.py::DATASET_PATH) - not
# used by the main twin-pair flow, kept for scripts/tools that still
# reference a flat single-file dataset.
DATASET_PATH = PROJECT_ROOT / "core" / "bbq_subset.jsonl"

MAX_EXAMPLES = 100
TASK_WORKERS = 2
PROBE_WORKERS = 2
PROBE_CUTS = 4
TOP_LOGPROBS = 4
KEEP_ALIVE = "24h"
CHECKPOINT_INTERVAL = 100
# When True (default), budget-condition tasks that run out of tokens before
# stating an answer get effective_answer filled in from the probe's forced
# completion. Set False for runs that need to measure true natural
# completion rate itself (e.g. a large-budget test) - forcing an answer
# there would mask exactly the "does the model finish on its own" signal.
ENABLE_FORCED_ANSWER = True
# Ollama defaults num_ctx to 4096; probe_cut_point concatenates the question
# prompt with the full partial reasoning trace, which can exceed that on
# longer chains and get silently truncated from the front (keep=4), dropping
# the actual question. Match client.py's raised default here too.
NUM_CTX = 16384

def test_effort_conversion():
    assert ReasoningEffort.from_value("low") == ReasoningEffort.LOW
    assert ReasoningEffort.from_value("medium") == ReasoningEffort.MEDIUM
    assert ReasoningEffort.from_value("high") == ReasoningEffort.HIGH
    assert ReasoningEffort.from_value(True) == ReasoningEffort.ON
    assert ReasoningEffort.from_value(False) == ReasoningEffort.OFF
    assert ReasoningEffort.from_value(None) == ReasoningEffort.MEDIUM

def test_ollama_conversion():
    assert ReasoningEffort.OFF.to_ollama_think() is False
    assert ReasoningEffort.ON.to_ollama_think() is True
    assert ReasoningEffort.LOW.to_ollama_think() == "low"
    assert ReasoningEffort.MEDIUM.to_ollama_think() == "medium"
    assert ReasoningEffort.HIGH.to_ollama_think() == "high"
    assert ReasoningEffort.MAX.to_ollama_think() == "max"