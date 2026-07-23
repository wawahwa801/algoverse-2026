from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence, Union
from ollama import ChatResponse, Client

from qwen3.effort import OllamaThink, ReasoningEffort

import httpx
import json


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

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        host: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model
        self.host = (host or self.DEFAULT_HOST).rstrip("/")
        self.timeout = timeout
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
        **kwargs: Any,
    ) -> dict[str, Any]:
        think: OllamaThink = effort.to_ollama_think()
        return {
            "model": self.model,
            "messages": list(messages),
            "think": think,
            "stream": stream,
            **kwargs,
        }

    def _post_chat(self, request: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(
            f"{self.host}/api/chat",
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
