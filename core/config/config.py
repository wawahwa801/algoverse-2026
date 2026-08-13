from core.config.effort import ReasoningEffort
from pathlib import Path
import os

MODEL = "grok-4.3"

# Per-model backend routing (models/eval.py::get_client) and result paths,
# so switching MODEL and rerunning never overwrites another model's output.
# Models absent from this registry default to the Ollama backend using the
# model name as-is (preserves the original single-model behavior).
MODEL_PROFILES = {
    "qwen3.5:9b": {"backend": "ollama", "model_id": "qwen3.5:9b"},
    "gpt-oss:20b": {"backend": "ollama", "model_id": "gpt-oss:20b"},
    "glm-5.2": {
        "backend": "azure",
        "model_id": "glm-5.2",
        "endpoint_url":  "",
        "api_key":  "",
    },
    "kimi-k3": {
        "backend": "azure",
        "model_id": "kimi-k3",
        "endpoint_url":  "",
        "api_key": "",
    },
    "grok-4.3": {
            "backend": "azure",
            "model_id": "grok-4.3",
            "endpoint_url":  "https://algoverseproject.services.ai.azure.com/openai/v1",
            "api_key": "28rgXIjl3sLPm7F4I0zSenqFKDk27dEf7T4Bv5oYqCcY2AkoVMg3JQQJ99CHACYeBjFXJ3w3AAAAACOGZvQc",
        },
    "grok-4.5": {"backend": "openrouter", "model_id": "xai/grok-4.5", "api_key": ""},
}


def _model_slug(model_name):
    return model_name.replace(":", "-").replace("/", "-")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "core" / "results"
RESULTS_JSON = RESULTS_DIR / f"bbq_results_{_model_slug(MODEL)}.json"
RESULTS_CSV = RESULTS_DIR / f"bbq_results_{_model_slug(MODEL)}.csv"

NATIVE_EFFORTS = ["low", "medium", "high"]
BUDGETS = [128, 512, 1024]
PROMPT_CONTROLS = []
BUDGET_THINK_MODES = [True]


DATA_ROOT = PROJECT_ROOT / "data"
PAIRS_DATASET_PATH = DATA_ROOT / "bbq_pairs_subset.jsonl"
CLEAN_DATASET_PATH = DATA_ROOT / "bbq_clean.jsonl"
SUBSET_DATASET_PATH = PROJECT_ROOT / "core" / "bbq_subset.jsonl"

MAX_EXAMPLES = 100
TASK_WORKERS = 1
PROBE_WORKERS = 2
PROBE_CUTS = 4
TOP_LOGPROBS = 4
KEEP_ALIVE = "24h"
CHECKPOINT_INTERVAL = 50
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