from effort import ReasoningEffort
from pathlib import Path

MODEL = "qwen3.5:9b"

# Per-model backend routing (models/eval.py::get_client) and result paths,
# so switching MODEL and rerunning never overwrites another model's output.
# Models absent from this registry default to the Ollama backend using the
# model name as-is (preserves the original single-model behavior).
MODEL_PROFILES = {
    "qwen3.5:9b": {"backend": "ollama", "model_id": "qwen3.5:9b"},
    "gpt-oss:20b": {"backend": "ollama", "model_id": "gpt-oss:20b"},
    "glm-5.2": {"backend": "openrouter", "model_id": "z-ai/glm-5.2"},
    "kimi-k3": {"backend": "openrouter", "model_id": "moonshotai/kimi-k3"},
}


def _model_slug(model_name):
    return model_name.replace(":", "-").replace("/", "-")


DATASET_PATH = Path("bbq_subset.jsonl")
RESULTS_JSON = Path(f"results/bbq_results_{_model_slug(MODEL)}.json")
RESULTS_CSV = Path(f"results/bbq_results_{_model_slug(MODEL)}.csv")

NATIVE_EFFORTS = ["low", "medium", "high"]
BUDGETS = [128, 512, 1024]
PROMPT_CONTROLS = ["answer_immediately", "think_thoroughly"]
BUDGET_THINK_MODES = [True]

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