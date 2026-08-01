from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from eval import get_full_chain_max_tokens


def test_full_chain_budget_tracks_reasoning_control():
    assert get_full_chain_max_tokens({"control_type": "native_effort", "effort": "low"}) == 256
    assert get_full_chain_max_tokens({"control_type": "native_effort", "effort": "medium"}) == 512
    assert get_full_chain_max_tokens({"control_type": "native_effort", "effort": "high"}) == 1024
    assert get_full_chain_max_tokens({"control_type": "budget", "max_tokens": 128}) == 128
    assert get_full_chain_max_tokens({"control_type": "prompt", "prompt_control": "answer_immediately"}) == 256
    assert get_full_chain_max_tokens({"control_type": "prompt", "prompt_control": "think_thoroughly"}) == 1024
