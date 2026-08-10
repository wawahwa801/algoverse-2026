from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence, Union

import httpx
from ollama import ChatResponse, Client

from core.config.effort import OllamaThink, ReasoningEffort
from core.config.config import (
    MODEL as DEFAULT_MODEL_NAME,
    MODEL_PROFILES,
)


Message = Mapping[str, Any]
Messages = Sequence[Message]
EffortInput = Union[ReasoningEffort, str, bool, None]


_SDK_UNSUPPORTED_THINK: frozenset[str] = frozenset({"max"})


@dataclass(frozen=True)
class Qwen3Response:
    content: str
    thinking: str | None
    effort: ReasoningEffort
    model: str
    raw: dict[str, Any]

    @property
    def thinking_chars(self) -> int:
        return len(self.thinking or "")


@dataclass(frozen=True)
class Qwen3StreamChunk:
    content: str | None = None
    thinking: str | None = None
    done: bool = False
    raw: dict[str, Any] | None = None


class Qwen3Client:

    DEFAULT_HOST = "http://127.0.0.1:11434"
    DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_NUM_CTX = 16384

    def __init__(
        self,
        model: str | None = None,
        *,
        host: str | None = None,
        timeout: float | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
    ) -> None:
        selected_model_key = model or DEFAULT_MODEL_NAME

        profile = MODEL_PROFILES.get(selected_model_key, {})
        backend = (provider or profile.get("backend") or "ollama").lower()

        self.backend = backend
        self.model_id = profile.get("model_id", selected_model_key)
        self.model = selected_model_key
        self.timeout = timeout

        if self.backend in {"openrouter", "openai"}:
            self.provider = "openrouter" if self.backend == "openrouter" else "openai"

            self.api_key = (
                api_key
                or profile.get("api_key")
                or os.getenv("OPENROUTER_API_KEY")
                or os.getenv("OPENAI_API_KEY", "")
            )

            default_url = (
                self.DEFAULT_OPENROUTER_BASE_URL
                if self.backend == "openrouter"
                else "https://api.openai.com/v1"
            )
            self.base_url = (
                base_url
                or profile.get("base_url")
                or os.getenv("OPENROUTER_BASE_URL")
                or default_url
            ).rstrip("/")
        else:
            self.provider = "ollama"
            self.host = (host or profile.get("host") or self.DEFAULT_HOST).rstrip("/")
            self.api_key = ""
            self.base_url = ""
            self._client = Client(host=self.host, timeout=timeout)

    def chat(
        self,
        messages: Messages,
        *,
        effort: EffortInput = ReasoningEffort.MEDIUM,
        stream: bool = False,
        **kwargs: Any,
    ) -> Qwen3Response | Iterator[Qwen3StreamChunk]:
        resolved_effort = ReasoningEffort.from_value(effort)

        if self.provider in {"openrouter", "openai"}:
            return self._chat_openrouter(messages, resolved_effort, **kwargs)

        request = self._build_request(messages, resolved_effort, stream=stream, **kwargs)

        if stream:
            return self._stream_chat(request, resolved_effort)

        if self._needs_http(resolved_effort):
            payload = self._post_chat(request)
            return self._to_response(payload, resolved_effort)

        response = self._client.chat(**request)
        return self._to_response(response.model_dump(), resolved_effort)

    def ask(
        self,
        prompt: str,
        *,
        effort: EffortInput = ReasoningEffort.MEDIUM,
        system: str | None = None,
        **kwargs: Any,
    ) -> Qwen3Response:
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = self.chat(messages, effort=effort, stream=False, **kwargs)
        assert isinstance(result, Qwen3Response)
        return result

    def _needs_http(self, effort: ReasoningEffort) -> bool:
        think = effort.to_ollama_think()
        return isinstance(think, str) and think in _SDK_UNSUPPORTED_THINK

    def _build_request(
        self,
        messages: Messages,
        effort: ReasoningEffort,
        *,
        stream: bool,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        think: OllamaThink = effort.to_ollama_think()

        request = {
            "model": self.model_id,
            "messages": list(messages),
            "think": think,
            "stream": stream,
            **kwargs,
        }

        options = request.get("options", {}).copy()
        options.setdefault("num_ctx", self.DEFAULT_NUM_CTX)

        if max_tokens is not None:
            options["num_predict"] = max_tokens

        request["options"] = options

        return request

    def _post_chat(self, request: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(
            f"{self.host}/api/chat",
            json=request,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _build_openrouter_request(
        self,
        messages: Messages,
        effort: ReasoningEffort,
        *,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        request = {
            "model": self.model_id,
            "messages": list(messages),
            **kwargs,
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if effort is not None:
            request.setdefault("reasoning_effort", effort.value)
        return request

    def _chat_openrouter(
        self,
        messages: Messages,
        effort: ReasoningEffort,
        **kwargs: Any,
    ) -> Qwen3Response:
        if not self.api_key:
            raise ValueError(
                f"API Key is required for provider '{self.provider}' (Model: {self.model_id}). "
                "Provide it in MODEL_PROFILES or via environment variables."
            )

        request = self._build_openrouter_request(messages, effort, **kwargs)
        payload = self._post_openrouter(request)
        return self._to_openrouter_response(payload, effort)

    def _post_openrouter(self, request: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=request,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _to_response(self, payload: dict[str, Any] | ChatResponse, effort: ReasoningEffort) -> Qwen3Response:
        if isinstance(payload, ChatResponse):
            payload = payload.model_dump()
        message = payload.get("message", {})
        return Qwen3Response(
            content=message.get("content") or "",
            thinking=message.get("thinking"),
            effort=effort,
            model=self.model,
            raw=payload,
        )

    def _to_openrouter_response(self, payload: dict[str, Any], effort: ReasoningEffort) -> Qwen3Response:
        choice = payload.get("choices", [{}])[0]
        message = choice.get("message", {})
        return Qwen3Response(
            content=message.get("content") or "",
            thinking=message.get("reasoning") or message.get("thinking"),
            effort=effort,
            model=self.model,
            raw=payload,
        )

    def _stream_chat(
        self,
        request: dict[str, Any],
        effort: ReasoningEffort,
    ) -> Iterator[Qwen3StreamChunk]:
        if self._needs_http(effort):
            yield from self._stream_http(request)
            return

        for chunk in self._client.chat(**request):
            payload = chunk.model_dump()
            message = payload.get("message", {})
            yield Qwen3StreamChunk(
                content=message.get("content"),
                thinking=message.get("thinking"),
                done=payload.get("done", False),
                raw=payload,
            )

    def _stream_http(self, request: dict[str, Any]) -> Iterator[Qwen3StreamChunk]:
        with httpx.stream(
            "POST",
            f"{self.host}/api/chat",
            json=request,
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                payload = json.loads(line)
                message = payload.get("message", {})
                yield Qwen3StreamChunk(
                    content=message.get("content"),
                    thinking=message.get("thinking"),
                    done=payload.get("done", False),
                    raw=payload,
                )