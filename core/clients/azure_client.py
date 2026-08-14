from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Union

import httpx

from core.config.effort import ReasoningEffort

Message = Mapping[str, Any]
Messages = Sequence[Message]
EffortInput = Union[ReasoningEffort, str, bool, None]


@dataclass(frozen=True)
class AzureResponse:
    content: str
    thinking: str | None
    effort: ReasoningEffort
    model: str
    raw: dict[str, Any]

    @property
    def thinking_chars(self) -> int:
        return len(self.thinking or "")


class AzureOpenAIClient:
    def __init__(
        self,
        endpoint_url: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.endpoint_url = (
            endpoint_url or os.getenv("AZURE_ENDPOINT_URL", "")
        ).rstrip("/")

        self.model = (
            model
            or os.getenv("AZURE_DEPLOYMENT_NAME", "grok-4.3")
        )

        self.api_key = (
            api_key
            or os.getenv("AZURE_API_KEY", "")
        )

        self.timeout = timeout or 180.0

        if not self.endpoint_url:
            raise ValueError(
                "Azure endpoint URL is required. Provide it in config or "
                "via AZURE_ENDPOINT_URL."
            )

        if not self.api_key:
            raise ValueError(
                "Azure API key is required. Provide it in config or "
                "via AZURE_API_KEY."
            )

    def ask(
        self,
        prompt: str,
        *,
        effort: EffortInput = ReasoningEffort.MEDIUM,
        system: str | None = None,
        max_tokens: int | None = None,
        prefix: str | None = None,
        **kwargs: Any,
    ) -> AzureResponse:

        messages: list[Message] = []

        if system:
            messages.append({
                "role": "system",
                "content": system,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        # Azure/OpenAI chat completions don't strictly support prefixing a continuation natively 
        # in the same way open-weight models do, but appending it as an assistant message is 
        # the standard workaround to ensure the model carries the context forward.
        if prefix:
            messages.append({
                "role": "assistant",
                "content": prefix,
            })

        resolved_effort = ReasoningEffort.from_value(effort)

        return self.chat(
            messages,
            effort=resolved_effort,
            max_tokens=max_tokens,
            **kwargs,
        )

    def chat(
        self,
        messages: Messages,
        *,
        effort: EffortInput = ReasoningEffort.MEDIUM,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AzureResponse:

        resolved_effort = ReasoningEffort.from_value(effort)

        payload = self._build_request(
            messages,
            resolved_effort,
            max_tokens=max_tokens,
            **kwargs,
        )

        response_data = self._post(payload)

        return self._to_response(
            response_data,
            resolved_effort,
        )

    def _build_request(
        self,
        messages: Messages,
        effort: ReasoningEffort,
        *,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:

        request: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            **kwargs,
        }

        # For Chat Completions, use max_completion_tokens
        # for reasoning models such as o1/o3.
        is_openai_reasoning_model = any(
            m in self.model.lower()
            for m in ["o1", "o3"]
        )

        # Reasoning models currently do not support logprobs
        if is_openai_reasoning_model:
            request.pop("logprobs", None)
            request.pop("top_logprobs", None)

        if max_tokens is not None:
            if is_openai_reasoning_model:
                request["max_completion_tokens"] = max_tokens
            else:
                request["max_tokens"] = max_tokens

        if is_openai_reasoning_model:
            azure_effort = (
                effort.to_azure_effort()
                if hasattr(effort, "to_azure_effort")
                else None
            )

            if azure_effort:
                request["reasoning_effort"] = azure_effort

        return request

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

        url = f"{self.endpoint_url}/chat/completions"
        max_retries = 5

        # Implemented exponential backoff to handle ThreadPoolExecutor 429 Rate Limits
        for attempt in range(max_retries):
            response = httpx.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            
            if response.status_code in (429, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
                
            response.raise_for_status()
            return response.json()
            
        raise RuntimeError(f"Azure API Request failed after {max_retries} attempts.")

    def _to_response(
        self,
        payload: dict[str, Any],
        effort: ReasoningEffort,
    ) -> AzureResponse:

        choice = payload.get("choices", [{}])[0]
        message = choice.get("message", {})

        return AzureResponse(
            content=message.get("content") or "",
            thinking=(
                message.get("reasoning_content")
                or message.get("reasoning")
            ),
            effort=effort,
            model=self.model,
            raw=payload,
        )

    def probe_logprobs(self, prompt: str, *, top_logprobs: int = 4) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        
        payload = self._build_request(
            messages,
            effort=ReasoningEffort.LOW,
            max_tokens=1,  # We only need the immediate next token logic for probing
            logprobs=True,
            top_logprobs=top_logprobs,
        )
        
        try:
            response_data = self._post(payload)
            choice = response_data.get("choices", [{}])[0]
            logprobs = choice.get("logprobs") or {}
            
            return {"content": logprobs.get("content", [])}
        except Exception as e:
            return {"content": []}