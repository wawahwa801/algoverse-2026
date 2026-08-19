from core.config.effort import ReasoningEffort
from pathlib import Path
import os
 
MODEL = "kimi-k2.6"
 
MODEL_PROFILES = {
    "qwen3:4b": {"backend": "ollama", "model_id": "qwen3:4b"},
    "qwen3.5:9b": {"backend": "ollama", "model_id": "qwen3.5:9b"},
    "gpt-oss:20b": {"backend": "ollama", "model_id": "gpt-oss:20b"},
    "kimi-k2.6": {
        "backend": "openrouter",
        "model_id": "kimi-k2.6",
        "endpoint_url": "https://algoverseproject.services.ai.azure.com/openai/v1",
        "api_key": "",
    },
    "deepseek-v4-pro": {
        "backend": "azure",
        "model_id": "deepseek-v4-pro",
        "endpoint_url": "",
        "api_key": "",
    },
 
    "grok-4.5": {"backend": "openrouter", "model_id": "xai/grok-4.5", "api_key": ""},
}
 
 
def _model_slug(model_name):
    return model_name.replace(":", "-").replace("/", "-")
 
 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "core" / "results"
RESULTS_JSON = RESULTS_DIR / f"bbq_results_{_model_slug(MODEL)}.json"
RESULTS_CSV = RESULTS_DIR / f"bbq_results_{_model_slug(MODEL)}.csv"
 
NATIVE_EFFORTS = ["off", "low", "medium", "high"]
OPENROUTER_NATIVE_EFFORTS = [True, False]
BUDGETS = [512, 2048, 8192]

PROMPT_CONTROL_OPTIONS = []
PROMPT_CONTROLS = []
BUDGET_THINK_MODES = [True]
 
 
DATA_ROOT = PROJECT_ROOT / "data"
 
PAIRS_DATASET_PATH = DATA_ROOT / "bbq_twins_fixedjsonl"
CLEAN_DATASET_PATH = DATA_ROOT / "bbq_clean.jsonl"
# Legacy hand-sampled subset path (models/config.py::DATASET_PATH) - not
# used by the main twin-pair flow, kept for scripts/tools that still
# reference a flat single-file dataset.
DATASET_PATH = PROJECT_ROOT / "core" / "bbq_subset.jsonl"
 
MAX_EXAMPLES = None
TASK_WORKERS = 2
PROBE_WORKERS = 2
PROBE_CUTS = 5
TOP_LOGPROBS = 5
KEEP_ALIVE = "24h"
CHECKPOINT_INTERVAL = 100
ENABLE_FLIP_RATE_EVAL = False
FLIP_RATE_K = 3
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