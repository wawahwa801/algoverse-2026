from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import httpx

from core.clients.olllama_client import Qwen3Response
from core.config.effort import ReasoningEffort

BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class OpenRouterResponse:
    content: str
    reasoning: str | None
    model: str
    raw: dict[str, Any]
    logprobs: dict[str, Any] | None = None

    @property
    def reasoning_chars(self) -> int:
        return len(self.reasoning or "")


class OpenRouterClient:
    """Generic async client for OpenRouter's OpenAI-compatible chat API.

    All four target models (GLM 5.2, Qwen 3.6, Kimi K3, DeepSeek V4) expose
    the same unified `reasoning: {"effort": ...}` control on OpenRouter, so
    one client works for all of them - no per-model branching needed.
    """

    def __init__(self, api_key: str | None = None, *, timeout: float = 180.0) -> None:
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )

    async def ask(
        self,
        model: str,
        prompt: str,
        *,
        effort: str | None = "medium",
        max_tokens: int | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
    ) -> OpenRouterResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "include_reasoning": True,
        }
        if effort is not None:
            payload["reasoning"] = {"effort": effort}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if logprobs:
            payload["logprobs"] = True
            if top_logprobs is not None:
                payload["top_logprobs"] = top_logprobs

        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        message = choice["message"]
        return OpenRouterResponse(
            content=message.get("content") or "",
            reasoning=message.get("reasoning"),
            model=model,
            raw=data,
            logprobs=choice.get("logprobs"),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class OpenRouterModelClient:
    """Sync, Qwen3Client-compatible adapter over OpenRouterClient, bound to
    one model id, so evaluate_example() (models/eval.py) can run unmodified
    regardless of whether the backend is Ollama or OpenRouter. Meant to be
    constructed once per worker thread (see eval.py's get_client) - keeps
    one persistent event loop for the client's lifetime rather than
    spinning a fresh loop per call, since OpenRouterClient's httpx.AsyncClient
    isn't safe to reuse across unrelated event loops."""

    def __init__(self, model_id: str, api_key: str | None = None) -> None:
        self.model_id = model_id
        self._loop = asyncio.new_event_loop()
        self._client = OpenRouterClient(api_key=api_key)

    @staticmethod
    def _resolve_effort(effort) -> str | None:
        resolved = ReasoningEffort.from_value(effort)
        think = resolved.to_ollama_think()
        if think is False:
            return None
        if think is True:
            return "medium"
        if think == "max":
            return "high"
        return think

    def ask(self, prompt, *, effort="medium", max_tokens=None, **_ignored):
        or_effort = self._resolve_effort(effort)

        response: OpenRouterResponse = self._loop.run_until_complete(
            self._client.ask(
                self.model_id,
                prompt,
                effort=or_effort,
                max_tokens=max_tokens,
            )
        )

        return Qwen3Response(
            content=response.content,
            thinking=response.reasoning,
            effort=ReasoningEffort.from_value(effort),
            model=response.model,
            raw=response.raw,
        )

    def probe_logprobs(self, prompt, *, top_logprobs=4):
        """Force-continuation probe equivalent to eval.py's probe_cut_point,
        but over the OpenRouter chat API - requests 1 output token with
        logprobs and returns the raw per-token top_logprobs entries (or None
        if the upstream provider didn't return any; not all OpenRouter
        providers support logprobs even when requested)."""
        response: OpenRouterResponse = self._loop.run_until_complete(
            self._client.ask(
                self.model_id,
                prompt,
                effort=None,
                max_tokens=1,
                logprobs=True,
                top_logprobs=top_logprobs,
            )
        )
        return response.logprobs

    def close(self):
        self._loop.run_until_complete(self._client.aclose())
        self._loop.close()
