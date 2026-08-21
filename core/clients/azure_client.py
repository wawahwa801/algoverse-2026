from __future__ import annotations

import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Union

import httpx

from core.config.effort import ReasoningEffort

Message = Mapping[str, Any]
Messages = Sequence[Message]
EffortInput = Union[ReasoningEffort, str, bool, None]


class _RateLimiter:


    def __init__(self, max_calls: int, period: float = 60.0) -> None:
        self.max_calls = max_calls
        self.period = period
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self.period:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return

                sleep_time = self.period - (now - self._timestamps[0])

            time.sleep(max(sleep_time, 0.05))


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
        max_retries: int = 5,
        max_concurrent_requests: int = 4,
        requests_per_minute: int | None = None,
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
        self.max_retries = max_retries

        # Caps concurrent in-flight requests from THIS client instance.
        self._semaphore = threading.BoundedSemaphore(max_concurrent_requests)

        # Caps actual request RATE (e.g. Azure quota RPM), independent of
        # concurrency. A semaphore alone doesn't stop you exceeding RPM if
        # each request is fast - this enforces the real ceiling.
        # Set slightly below your quota (e.g. 45 for a 50 RPM limit) to
        # leave margin for clock skew / in-flight requests.
        self._rate_limiter = (
            _RateLimiter(requests_per_minute) if requests_per_minute else None
        )

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
        think: bool | None = None,
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
            think=think,
            **kwargs,
        )

    def chat(
        self,
        messages: Messages,
        *,
        effort: EffortInput = ReasoningEffort.MEDIUM,
        max_tokens: int | None = None,
        think: bool | None = None,
        **kwargs: Any,
    ) -> AzureResponse:

        if self._model_family() == "kimi_k2.6":
            resolved_effort = ReasoningEffort.from_value("medium")
        else:
            resolved_effort = ReasoningEffort.from_value(effort)

        payload = self._build_request(
            messages,
            effort,
            max_tokens=max_tokens,
            think=think,
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
        think: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:

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

        if think is not None and model_family == "kimi_k2.6":
            request["thinking"] = {"type": "enabled" if think else "disabled"}

        return request

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

        url = f"{self.endpoint_url}/chat/completions"
        last_error: str | None = None

        with self._semaphore:
            for attempt in range(self.max_retries):
                if self._rate_limiter is not None:
                    self._rate_limiter.acquire()

                try:
                    response = httpx.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout,
                    )
                except httpx.TransportError as e:
                    last_error = f"{type(e).__name__}: {e}"
                    time.sleep(self._backoff_seconds(attempt))
                    continue

                if response.status_code in (429, 502, 503, 504):
                    last_error = f"HTTP {response.status_code}: {response.text[:500]}"

                    retry_after = response.headers.get("retry-after")
                    if retry_after is not None:
                        try:
                            sleep_for = float(retry_after)
                        except ValueError:
                            sleep_for = self._backoff_seconds(attempt)
                    else:
                        sleep_for = self._backoff_seconds(attempt)

                    time.sleep(sleep_for)
                    continue

                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    raise RuntimeError(
                        f"Azure API request failed: HTTP {response.status_code}: "
                        f"{response.text[:500]}"
                    ) from e

                return response.json()

        raise RuntimeError(
            f"Azure API Request failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _backoff_seconds(attempt: int) -> float:
        return (2 ** attempt) + random.uniform(0, 1)

    def _to_response(
        self,
        payload: dict[str, Any],
        effort: ReasoningEffort,
    ) -> AzureResponse:

        choices = payload.get("choices") or []
        choice = choices[0] if choices else {}
        if not choice:
            choice = {}

        message = choice.get("message") or {}

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

        payload = self._build_request(
            messages,
            effort=ReasoningEffort.LOW,
            max_tokens=1,
            **build_kwargs,
        )

        try:
            response_data = self._post(payload)

            choices = response_data.get("choices") or []
            choice = choices[0] if choices else {}
            if not choice:
                choice = {}

            logprobs = choice.get("logprobs") or {}

            return {"content": logprobs.get("content") or []}
        except Exception as e:
            print(f"probe_logprobs failed: {e!r}")
            return {"content": []}