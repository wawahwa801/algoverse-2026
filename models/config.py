from effort import ReasoningEffort
from pathlib import Path

MODEL = "qwen3.5:9b"
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