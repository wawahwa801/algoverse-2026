from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence, Union

import httpx
from ollama import ChatResponse, Client

from effort import OllamaThink, ReasoningEffort
from config import (
    MODEL as DEFAULT_MODEL_NAME,
    MODEL_PROVIDER as DEFAULT_MODEL_PROVIDER,
    OPENAI_API_KEY as DEFAULT_OPENAI_API_KEY,
    OPENAI_BASE_URL as DEFAULT_OPENAI_BASE_URL,
    OPENAI_MODEL as DEFAULT_OPENAI_MODEL,
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

    DEFAULT_MODEL = "qwen3:4b"
    DEFAULT_HOST = "http://127.0.0.1:11434"
    # Ollama defaults num_ctx to 4096 regardless of what the model supports,
    # which silently truncates long reasoning chains (and, worse, the START
    # of the prompt when a request already exceeds it). Raise it so uncapped
    # native-effort/prompt conditions don't get cut off mid-reasoning.
    DEFAULT_NUM_CTX = 16384

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        host: str | None = None,
        timeout: float | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.provider = self._resolve_provider(provider)
        if self.provider == "openai" and model in {self.DEFAULT_MODEL, DEFAULT_MODEL_NAME}:
            model = DEFAULT_OPENAI_MODEL
        self.model = model
        self.host = (host or self.DEFAULT_HOST).rstrip("/")
        self.timeout = timeout
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", DEFAULT_OPENAI_API_KEY)
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)).rstrip("/")
        self._client = Client(host=self.host, timeout=timeout)

    def _resolve_provider(self, provider: str | None) -> str:
        if provider is None:
            provider = os.getenv("MODEL_PROVIDER", DEFAULT_MODEL_PROVIDER)
        provider = str(provider).strip().lower()
        if provider in {"openai", "openai-compatible", "openai_compatible", "api", "http"}:
            return "openai"
        return "ollama"

    def chat(
        self,
        messages: Messages,
        *,
        effort: EffortInput = ReasoningEffort.MEDIUM,
        stream: bool = False,
        **kwargs: Any,
    ) -> Qwen3Response | Iterator[Qwen3StreamChunk]:
        resolved_effort = ReasoningEffort.from_value(effort)

        if self.provider == "openai":
            return self._chat_openai(messages, resolved_effort, **kwargs)

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
            "model": self.model,
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

    def _build_openai_request(
        self,
        messages: Messages,
        effort: ReasoningEffort,
        *,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        request = {
            "model": self.model,
            "messages": list(messages),
            **kwargs,
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if effort is not None:
            request.setdefault("reasoning_effort", effort.value)
        return request

    def _chat_openai(
        self,
        messages: Messages,
        effort: ReasoningEffort,
        **kwargs: Any,
    ) -> Qwen3Response:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required when MODEL_PROVIDER=openai")

        request = self._build_openai_request(messages, effort, **kwargs)
        payload = self._post_openai(request)
        return self._to_openai_response(payload, effort)

    def _post_openai(self, request: dict[str, Any]) -> dict[str, Any]:
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

    def _to_openai_response(self, payload: dict[str, Any], effort: ReasoningEffort) -> Qwen3Response:
        choice = payload.get("choices", [{}])[0]
        message = choice.get("message", {})
        return Qwen3Response(
            content=message.get("content") or "",
            thinking=message.get("thinking"),
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
