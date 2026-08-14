"""Core BBQ evaluation pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.clients.olllama_client import Qwen3Client, Qwen3Response, Qwen3StreamChunk
    from core.config.effort import ReasoningEffort

__all__ = [
    "Qwen3Client",
    "Qwen3Response",
    "Qwen3StreamChunk",
    "ReasoningEffort",
]


def __getattr__(name: str):
    if name in {"Qwen3Client", "Qwen3Response", "Qwen3StreamChunk"}:
        from core.clients.olllama_client import Qwen3Client, Qwen3Response, Qwen3StreamChunk

        return {
            "Qwen3Client": Qwen3Client,
            "Qwen3Response": Qwen3Response,
            "Qwen3StreamChunk": Qwen3StreamChunk,
        }[name]
    if name == "ReasoningEffort":
        from core.config.effort import ReasoningEffort

        return ReasoningEffort
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
