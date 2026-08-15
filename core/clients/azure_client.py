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


class AzureClient:
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

    def _model_family(self) -> str:
        name = self.model.lower().replace("_", "-")

        if "kimi-k2.6" in name or "kimi-k2-6" in name:
            return "kimi_k2.6"

        if any(token in name for token in ("o1", "o3")):
            return "openai_reasoning"

        return "generic"

    @staticmethod
    def _effort_value(effort: EffortInput) -> str:
        if isinstance(effort, ReasoningEffort):
            value = getattr(effort, "value", effort)
        elif effort is None:
            value = "off"
        elif isinstance(effort, bool):
            value = "medium" if effort else "off"
        else:
            value = effort
        return str(value).lower()

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

        if prefix:
            if self._model_family() == "kimi_k2.6":
                # Kimi K2.6 supports preserved thinking through the
                # assistant reasoning_content field. Keep the actual
                # reasoning prefix separate from normal assistant content.
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": prefix,
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": prefix,
                })

        if self._model_family() == "kimi_k2.6":
            resolved_effort = ReasoningEffort.from_value("medium")
        else:
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

        # Kimi K2.6 has a binary native thinking control rather than
        # OpenAI-style low/medium/high reasoning_effort.
        if self._model_family() == "kimi_k2.6":
            resolved_effort = ReasoningEffort.from_value("medium")
        else:
            resolved_effort = ReasoningEffort.from_value(effort)

        payload = self._build_request(
            messages,
            effort,
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
        effort: EffortInput,
        *,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:

        # Clean kwargs to prevent passing internal control flags to Azure
        kwargs.pop("think", None)
        kwargs.pop("prompt_control", None)

        request: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            **kwargs,
        }

        model_family = self._model_family()
        is_openai_reasoning = model_family == "openai_reasoning" or any(
            k in self.model.lower() for k in ("o1", "o3")
        )

        if is_openai_reasoning:
            request.pop("logprobs", None)
            request.pop("top_logprobs", None)
            if max_tokens is not None:
                request["max_completion_tokens"] = max_tokens
            
            azure_effort = (
                effort.to_azure_effort()
                if isinstance(effort, ReasoningEffort) and hasattr(effort, "to_azure_effort")
                else self._effort_value(effort)
            )
            if azure_effort in {"low", "medium", "high"}:
                request["reasoning_effort"] = azure_effort
        else:
            if max_tokens is not None:
                request["max_tokens"] = max_tokens

        return request

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

        url = f"{self.endpoint_url}/chat/completions"
        max_retries = 5

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

        build_kwargs: dict[str, Any] = {
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }

        # Commitment probing needs the answer-token distribution immediately;
        # do not let Kimi spend the one-token probe call generating another
        # reasoning step first.
        if self._model_family() == "kimi_k2.6":
            build_kwargs["thinking"] = {"type": "disabled"}

        payload = self._build_request(
            messages,
            effort=ReasoningEffort.LOW,
            max_tokens=1,
            **build_kwargs,
        )

        try:
            response_data = self._post(payload)
            choice = response_data.get("choices", [{}])[0]
            logprobs = choice.get("logprobs") or {}

            return {"content": logprobs.get("content", [])}
        except Exception:
            return {"content": []}