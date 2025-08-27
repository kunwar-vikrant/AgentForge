#!/usr/bin/env python3
"""Interactive / CLI multi-agent demo runner.

Usage:
  python scripts/run_multi_agent.py --task "Summarize this website: https://example.com" --provider ollama

Flags:
  --task <text>        (required) user objective
  --provider <name>    one of openai|grok|ollama|groq|anthropic|echo (default echo)
  --max-turns <int>    orchestrator turn cap (default 8)
  --async              use async provider adapter (if available)

The 'echo' provider is a deterministic mock that requires no API keys and runs offline.
"""
from __future__ import annotations
import argparse
import asyncio
import sys
from pathlib import Path

# Ensure src on path when executed from repo root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agents.base import BaseAgent  # noqa: E402
from agents.orchestrator import Orchestrator  # noqa: E402
from agents.adapters import SyncProviderLLM, AsyncProviderLLM, EchoLLM  # noqa: E402
from agents.tools import SleepTool  # noqa: E402


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multi-agent orchestration demo")
    p.add_argument("--task", required=True, help="User objective or problem statement")
    p.add_argument("--provider", default="echo", help="LLM provider (openai/grok/ollama/groq/anthropic/echo)")
    p.add_argument("--max-turns", type=int, default=8, help="Maximum supervisor turns")
    p.add_argument("--async", dest="use_async", action="store_true", help="Use async provider adapter")
    return p.parse_args()


def build_llm(provider: str, use_async: bool):
    if provider == "echo":
        return EchoLLM()
    if use_async:
        return AsyncProviderLLM(provider)
    return SyncProviderLLM(provider)


async def main_async():
    args = build_args()
    llm = build_llm(args.provider, args.use_async)

    supervisor = BaseAgent(
        "supervisor",
        llm,
        "Supervisor: decide NEXT:<agent> or FINISH:<answer>. Keep oversight concise.",
    )
    # Simple worker with a SleepTool to demonstrate tool reflection path
    worker = BaseAgent(
        "worker",
        llm,
        "Worker agent: solve tasks and when complete use RESPOND:<answer>.",
        tools=[SleepTool()],
    )

    orchestrator = Orchestrator({"worker": worker}, supervisor, max_turns=args.max_turns)
    result = await orchestrator.run(args.task)

    print("\n=== Orchestration Result ===")
    print(f"Finished: {result.finished}  Turns: {result.turns}")
    print(f"Final Answer: {result.final_answer}\n")
    print("Transcript:")
    for m in result.transcript:
        print(f"[{m.role}] {m.sender}: {m.content}")


def main():  # pragma: no cover
    asyncio.run(main_async())


if __name__ == "__main__":  # pragma: no cover
    main()
