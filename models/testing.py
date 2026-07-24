from effort import ReasoningEffort
from client import Qwen3Client


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

if __name__ == "__main__":
    test_effort_conversion()
    test_ollama_conversion()
    print("good conversion")


client = Qwen3Client(
    model="qwen3:4b"
)

prompt = input("prompt \n")
while True:
    response = client.ask(
        prompt,
        effort=input("effort: low/medium/high \n")
    )


    print("MODEL:")
    print(response.model)

    print("\nEFFORT:")
    print(response.effort)

    print("\nTHINKING:")
    print(response.thinking)

    print("\nANSWER:")
    print(response.content)

    print("\nTHINKING CHARACTERS:")
    print(response.thinking_chars)


