"""Simple smoke tests for AgentForge providers.

Usage:
  python scripts/smoke_test.py --provider openai --prompt "Say hello"
  python scripts/smoke_test.py --async --provider openai --prompt "Say hello"    

Supports OPENAI_BASE_URL / OLLAMA_HOST / OLLAMA_PORT overrides.
"""
from __future__ import annotations
import argparse
import asyncio
import os
import time
from typing import Optional

from llm_providers import get_llm_response
from providers_async import get_async_provider, shutdown_providers


def sync_call(provider: str, prompt: str) -> str:
    start = time.time()
    out = get_llm_response(provider, prompt, max_retries=1)
    dur = time.time() - start
    return f"[sync:{provider}] {dur:.2f}s -> {out[:200]}"  # truncate


async def async_call(provider: str, prompt: str) -> str:
    start = time.time()
    client = get_async_provider(provider)
    out = await client.complete(prompt)
    dur = time.time() - start
    return f"[async:{provider}] {dur:.2f}s -> {out[:200]}"  # truncate


async def run_async(provider: str, prompt: str, concurrent: int):
    tasks = [asyncio.create_task(async_call(provider, f"{prompt} (job {i})")) for i in range(concurrent)]
    for t in asyncio.as_completed(tasks):
        try:
            print(await t)
        except Exception as e:
            print(f"Error: {e}")
    await shutdown_providers()


def main():
    ap = argparse.ArgumentParser(description="AgentForge smoke tests")
    ap.add_argument("--provider", required=True, choices=["openai", "grok", "ollama"], help="Provider name")
    ap.add_argument("--prompt", required=True, help="Prompt text")
    ap.add_argument("--async", dest="use_async", action="store_true", help="Use async provider layer")
    ap.add_argument("--concurrent", type=int, default=1, help="Concurrent async calls")
    args = ap.parse_args()

    if not args.use_async:
        try:
            print(sync_call(args.provider, args.prompt))
        except Exception as e:
            print(f"Sync error: {e}")
    else:
        asyncio.run(run_async(args.provider, args.prompt, args.concurrent))


if __name__ == "__main__":
    main()