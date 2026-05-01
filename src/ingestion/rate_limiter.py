"""
Async token bucket rate limiter for Azure OpenAI embedding calls.

Respects per-minute TPM (tokens per minute) and RPM (requests per minute).
Used to keep ingestion under the deployment's quota and avoid 429 storms.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque


class TokenBucket:
    """Sliding-window rate limiter for tokens AND requests per minute.

    - max_tokens_per_minute: e.g., 300_000 (slightly under 350K Azure S0 cap)
    - max_requests_per_minute: e.g., 240 (Azure S0 typical RPM)

    Usage:
        bucket = TokenBucket(300_000, 240)
        await bucket.acquire(estimated_tokens)
        # ... make API call ...
    """

    def __init__(self, max_tokens_per_minute: int, max_requests_per_minute: int):
        self.tpm = max_tokens_per_minute
        self.rpm = max_requests_per_minute
        self._token_log: deque[tuple[float, int]] = deque()
        self._req_log: deque[float] = deque()
        self._lock = asyncio.Lock()

    def _purge(self, now: float) -> None:
        cutoff = now - 60.0
        while self._token_log and self._token_log[0][0] < cutoff:
            self._token_log.popleft()
        while self._req_log and self._req_log[0] < cutoff:
            self._req_log.popleft()

    def _used(self) -> tuple[int, int]:
        return sum(t for _, t in self._token_log), len(self._req_log)

    async def acquire(self, tokens: int) -> None:
        """Block until *tokens* fit within the per-minute window."""
        if tokens > self.tpm:
            raise ValueError(
                f"Single request {tokens} tokens exceeds per-minute limit {self.tpm}"
            )
        while True:
            async with self._lock:
                now = time.monotonic()
                self._purge(now)
                used_tokens, used_reqs = self._used()
                if used_tokens + tokens <= self.tpm and used_reqs + 1 <= self.rpm:
                    self._token_log.append((now, tokens))
                    self._req_log.append(now)
                    return
                # Compute earliest moment we'd have headroom.
                wait = 1.0
                if used_tokens + tokens > self.tpm and self._token_log:
                    excess = (used_tokens + tokens) - self.tpm
                    cum = 0
                    for ts, t in self._token_log:
                        cum += t
                        if cum >= excess:
                            wait = max(wait, 60.0 - (now - ts) + 0.1)
                            break
                if used_reqs + 1 > self.rpm and self._req_log:
                    wait = max(wait, 60.0 - (now - self._req_log[0]) + 0.1)
            await asyncio.sleep(min(wait, 30.0))


def estimate_tokens(texts: list[str], chars_per_token: float = 3.0) -> int:
    """Estimate total tokens. Use 3.0 chars/token for safety with Portuguese."""
    return int(sum(len(t) for t in texts) / chars_per_token) + len(texts) * 5
