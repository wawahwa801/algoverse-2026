from __future__ import annotations

import json
import os
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
    """Azure OpenAI API client compatible with Qwen3Client interface.
    
    Supports both GLM-5.2 (Zhipu) and Kimi-K3 (Moonshot) via Azure's
    OpenAI-compatible endpoints.
    """

    DEFAULT_API_VERSION = "2024-08-01-preview"

    def __init__(
        self,
        model: str | None = None,
        *,
        resource_name: str | None = None,
        deployment_name: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model or "gpt-4"
        self.resource_name = resource_name or os.getenv("AZURE_RESOURCE_NAME", "")
        self.deployment_name = deployment_name or os.getenv("AZURE_DEPLOYMENT_NAME", "")
        self.api_key = api_key or os.getenv("AZURE_API_KEY", "")
        self.api_version = api_version or self.DEFAULT_API_VERSION
        self.timeout = timeout or 180.0

        if not self.resource_name:
            raise ValueError(
                "Azure resource name is required. Provide it in config or via "
                "AZURE_RESOURCE_NAME environment variable."
            )
        if not self.deployment_name:
            raise ValueError(
                "Azure deployment name is required. Provide it in config or via "
                "AZURE_DEPLOYMENT_NAME environment variable."
            )
        if not self.api_key:
            raise ValueError(
                "Azure API key is required. Provide it in config or via "
                "AZURE_API_KEY environment variable."
            )

        self.base_url = (
            f"https://{self.resource_name}.openai.azure.com/openai/deployments/"
            f"{self.deployment_name}"
        )

    def ask(
        self,
        prompt: str,
        *,
        effort: EffortInput = ReasoningEffort.MEDIUM,
        system: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AzureResponse:
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

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
        return self._to_response(response_data, resolved_effort)

    def _build_request(
        self,
        messages: Messages,
        effort: ReasoningEffort,
        *,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        request = {
            "messages": list(messages),
            **kwargs,
        }

        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        if effort != ReasoningEffort.OFF:
            effort_value = effort.value if hasattr(effort, "value") else str(effort)
            request["reasoning"] = {
                "type": "enabled",
                "max_tokens": 8000,
                "effort": effort_value if effort_value != "off" else None,
            }
            if request["reasoning"]["effort"] is None:
                request["reasoning"].pop("effort", None)

        return request

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions?api-version={self.api_version}"

        response = httpx.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _to_response(
        self,
        payload: dict[str, Any],
        effort: ReasoningEffort,
    ) -> AzureResponse:
        choice = payload.get("choices", [{}])[0]
        message = choice.get("message", {})

        return AzureResponse(
            content=message.get("content") or "",
            thinking=message.get("reasoning"),
            effort=effort,
            model=self.model,
            raw=payload,
        )

    def probe_logprobs(self, prompt, *, top_logprobs=4):
        """Probe for logprobs - Azure doesn't always support this for all models,
        so return empty dict as fallback."""
        return {"content": []}
