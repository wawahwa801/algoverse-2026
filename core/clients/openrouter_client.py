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
        prefix: str | None = None,
    ) -> OpenRouterResponse:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

        if prefix:
            messages.append({"role": "assistant", "content": prefix})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "include_reasoning": True,
        }
        if effort is not None:
            payload["reasoning"] = {"effort": effort}
        else:

            payload["reasoning"] = {"enabled": False}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if logprobs:
            payload["logprobs"] = True
            if top_logprobs is not None:
                payload["top_logprobs"] = top_logprobs

            payload["provider"] = {"require_parameters": True}

        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        if "choices" not in data:

            error_info = data.get("error") if isinstance(data, dict) else None
            message = (
                error_info.get("message")
                if isinstance(error_info, dict)
                else None
            )
            raise RuntimeError(
                f"OpenRouter returned no choices for model={model!r} "
                f"(effort={effort!r}, logprobs={logprobs}): "
                f"{message or data}"
            )

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

    def ask(self, prompt, *, effort="medium", max_tokens=None, prefix=None, **_ignored):
        or_effort = self._resolve_effort(effort)

        response: OpenRouterResponse = self._loop.run_until_complete(
            self._client.ask(
                self.model_id,
                prompt,
                effort=or_effort,
                max_tokens=max_tokens,
                prefix=prefix,
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