"""Async provider abstraction using httpx for higher throughput."""
from __future__ import annotations
import os
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
import httpx

logger = logging.getLogger(__name__)

@dataclass
class AsyncProviderConfig:
    name: str
    url: str
    model: str
    api_key_env: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = 4000
    timeout: float = 60.0
    concurrency: int = 8  # per-provider semaphore


class ProviderClient:
    def __init__(self, cfg: AsyncProviderConfig):
        self.cfg = cfg
        self._semaphore = asyncio.Semaphore(cfg.concurrency)
        self._client = httpx.AsyncClient(timeout=cfg.timeout)

    async def close(self):
        await self._client.aclose()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key_env:
            key = os.getenv(self.cfg.api_key_env)
            if not key:
                raise ValueError(f"Missing env var {self.cfg.api_key_env}")
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _payload(self, prompt: str) -> Dict[str, Any]:
        data = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.cfg.temperature,
        }
        if self.cfg.max_tokens:
            data["max_tokens"] = self.cfg.max_tokens
        if self.cfg.name == "ollama":
            data["stream"] = False
        return data

    async def complete(self, prompt: str, retries: int = 2) -> str:
        if not prompt.strip():
            raise ValueError("Prompt empty")
        attempt = 0
        backoff = 1.0
        async with self._semaphore:
            while True:
                try:
                    r = await self._client.post(self.cfg.url, headers=self._headers(), json=self._payload(prompt))
                    if r.status_code >= 400:
                        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                            raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                        r.raise_for_status()
                    data = r.json()
                    # Parse
                    if 'choices' in data and data['choices']:
                        return data['choices'][0]['message']['content'].strip()
                    if 'message' in data and isinstance(data['message'], dict):
                        return data['message'].get('content', '').strip()
                    if 'content' in data:
                        return str(data['content']).strip()
                    return str(data)
                except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
                    if attempt >= retries:
                        raise RuntimeError(f"Provider {self.cfg.name} failed after {attempt+1} attempts: {e}") from e
                    logger.warning(f"{self.cfg.name} attempt {attempt+1} failed ({e}); backing off {backoff}s")
                    await asyncio.sleep(backoff)
                    attempt += 1
                    backoff = min(backoff * 2, 8)


PROVIDERS: Dict[str, AsyncProviderConfig] = {
    'openai': AsyncProviderConfig(name='openai', url='https://api.openai.com/v1/chat/completions', model='gpt-4o-mini', api_key_env='OPENAI_API_KEY'),
    'grok': AsyncProviderConfig(name='grok', url='https://api.x.ai/v1/chat/completions', model='grok-beta', api_key_env='XAI_API_KEY'),
    'ollama': AsyncProviderConfig(name='ollama', url='http://localhost:11434/api/chat', model='llama3'),
}

_client_cache: Dict[str, ProviderClient] = {}

def get_async_provider(name: str) -> ProviderClient:
    key = name.lower().strip()
    if key not in PROVIDERS:
        raise ValueError(f"Unknown provider {name}")
    if key not in _client_cache:
        _client_cache[key] = ProviderClient(PROVIDERS[key])
    return _client_cache[key]

async def shutdown_providers():
    await asyncio.gather(*[c.close() for c in _client_cache.values()])