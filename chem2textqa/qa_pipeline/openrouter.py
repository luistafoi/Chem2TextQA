"""Async OpenRouter client for the QA generation phases.

Adapted from Robert's pipeline/claim_extractor/openrouter.py — kept simple so
each phase can `client.complete(session, system, user)` without bespoke
plumbing. Retries handle 429 and 5xx.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


class OpenRouterClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.semaphore: Optional[asyncio.Semaphore] = None

    async def complete(
        self,
        session: aiohttp.ClientSession,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.3,
        retries: int = 3,
        timeout: int = 180,
        reasoning: Optional[dict] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Send a chat completion. Returns (response_text, error_string) —
        exactly one will be None.

        Args:
            reasoning: optional OpenRouter reasoning control, e.g.
                {"enabled": False} to disable reasoning on hybrid models
                like Kimi K2.5 that otherwise consume the entire token
                budget on internal reasoning and emit empty content.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/chem2textqa",
        }
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if reasoning is not None:
            payload["reasoning"] = reasoning

        last_error: Optional[str] = None
        for attempt in range(retries):
            try:
                sem = self.semaphore or _NullSemaphore()
                async with sem:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data["choices"][0]["message"]["content"], None
                        elif resp.status == 429:
                            wait = 5 * (attempt + 1)
                            logger.warning("Rate limited on %s, waiting %ds", self.model, wait)
                            await asyncio.sleep(wait)
                            last_error = "Rate limited"
                        elif 500 <= resp.status < 600:
                            wait = 2 * (attempt + 1)
                            text = await resp.text()
                            last_error = f"HTTP {resp.status}: {text[:200]}"
                            await asyncio.sleep(wait)
                        else:
                            text = await resp.text()
                            return None, f"HTTP {resp.status}: {text[:200]}"
            except asyncio.TimeoutError:
                last_error = f"Timeout ({timeout}s)"
                logger.warning("Timeout on attempt %d", attempt + 1)
            except Exception as e:
                last_error = str(e)
                logger.warning("Error on attempt %d: %s", attempt + 1, e)

            await asyncio.sleep(2 * (attempt + 1))

        return None, last_error


class _NullSemaphore:
    """Context manager that allows unbounded concurrency. Used when caller
    forgot to set client.semaphore."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None
