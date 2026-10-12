"""Bounded OpenAI-compatible requests with explicit retries and safe errors."""
from __future__ import annotations

import random
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from typing import Iterator
from .extract import extract_judge
from .thinking import add_qwen_thinking_switch


@dataclass(frozen=True)
class APIConfig:
    model: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = 16384
    max_workers: int = 8
    max_retries: int = 4
    timeout: float = 180.0
    retry_wait: float = 1.0
    mode: str = "eval"
    enable_thinking: bool = True

    def __post_init__(self):
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.max_workers < 1 or self.max_tokens < 1 or self.max_retries < 0 or self.timeout <= 0 or self.retry_wait < 0:
            raise ValueError("Invalid API worker/token/retry/timeout configuration")
        if self.mode not in {"eval", "judge"}:
            raise ValueError("mode must be eval or judge")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

    def public_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class APIResult:
    response: str
    attempts: int
    errors: tuple[dict, ...] = ()

    @property
    def succeeded(self):
        return self.response != "API_ERROR"


class APIClient:
    def __init__(self, api_key: str, base_url: str | None, config: APIConfig):
        from openai import OpenAI
        self.config = config
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=config.timeout)

    def close(self):
        self.client.close()

    def request(self, prompt: str) -> APIResult:
        from openai import APIError
        config = self.config
        if config.mode == "eval":
            prompt = add_qwen_thinking_switch(prompt, config.model, config.enable_thinking)
        request = {"model": config.model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": config.max_tokens}
        # Omit unset sampling values: some reasoning models reject these fields.
        if config.temperature is not None:
            request["temperature"] = config.temperature
        if config.top_p is not None:
            request["top_p"] = config.top_p
        errors = []
        for attempt in range(1, config.max_retries + 2):
            try:
                result = self.client.chat.completions.create(**request)
                content = result.choices[0].message.content
                if not content or not content.strip():
                    raise ValueError("empty_response")
                if config.mode == "judge" and extract_judge(content) is None:
                    raise ValueError("invalid_judge")
                return APIResult(content, attempt, tuple(errors))
            except (APIError, ValueError, TypeError, IndexError, AttributeError) as error:
                status = getattr(error, "status_code", None)
                # Do not persist server bodies, headers or exception messages: they may echo credentials.
                errors.append({"attempt": attempt, "type": type(error).__name__,
                               "status_code": status if isinstance(status, int) else None})
                permanent = isinstance(status, int) and 400 <= status < 500 and status not in {408, 409, 429}
                if permanent or attempt > config.max_retries:
                    return APIResult("API_ERROR", attempt, tuple(errors))
                delay = min(30, config.retry_wait * 2 ** (attempt - 1))
                time.sleep(delay + random.uniform(0, config.retry_wait))
        raise AssertionError("unreachable")

    def iter_results(self, prompts: list[str]) -> Iterator[tuple[int, APIResult]]:
        pending = iter(enumerate(prompts))
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}
            def submit():
                item = next(pending, None)
                if item is not None:
                    index, prompt = item
                    futures[executor.submit(self.request, prompt)] = index
            for _ in range(min(len(prompts), self.config.max_workers * 2)):
                submit()
            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    yield futures.pop(future), future.result()
                    submit()
