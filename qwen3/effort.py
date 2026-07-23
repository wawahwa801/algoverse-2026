from __future__ import annotations

from enum import Enum
from typing import Union

OllamaThink = Union[bool, str]


class ReasoningEffort(str, Enum):

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"
    ON = "on"

    def to_ollama_think(self) -> OllamaThink:
        if self is ReasoningEffort.OFF:
            return False
        if self is ReasoningEffort.ON:
            return True
        return self.value

    @classmethod
    def from_value(cls, value: Union["ReasoningEffort", str, bool, None]) -> "ReasoningEffort":
        if value is None:
            return cls.MEDIUM
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls.ON if value else cls.OFF

        normalized = str(value).strip().lower()
        aliases = {
            "false": cls.OFF,
            "true": cls.ON,
            "none": cls.OFF,
            "0": cls.OFF,
            "1": cls.ON,
        }
        if normalized in aliases:
            return aliases[normalized]
        return cls(normalized)
