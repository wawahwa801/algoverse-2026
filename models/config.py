import os
from pathlib import Path

from effort import ReasoningEffort


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file()

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama")
MODEL = os.getenv("MODEL", "qwen3.5:9b")

API_KEY = os.getenv("API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "")

DATASET_PATH = Path("bbq_subset.jsonl")
RESULTS_JSON = Path("results/bbq_results.json")
RESULTS_CSV = Path("results/bbq_results.csv")

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