"""Command-line interface for AgentForge.

Provides subcommands:
  plan      – planning only
  generate  – full single-agent pipeline
  multi     – run multi-agent supervisor loop

Environment variables for providers (set what you have):
  OPENAI_API_KEY, XAI_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY
Use provider 'echo' for an offline deterministic mock (multi only).
"""
from __future__ import annotations
import argparse
import asyncio
import sys

from main import run_generation
from llm_providers import get_llm_response
from config import get_config

# Multi-agent imports (optional path)
try:
    from agents.base import BaseAgent
    from agents.adapters import EchoLLM, SyncProviderLLM
    from agents.orchestrator import Orchestrator
except Exception:  # pragma: no cover
    BaseAgent = None  # type: ignore
    EchoLLM = None  # type: ignore
    SyncProviderLLM = None  # type: ignore
    Orchestrator = None  # type: ignore


def cmd_plan(args: argparse.Namespace):
    cfg = get_config()
    prompt = (
        "You are an expert AI architect. Draft a step-by-step plan for this agent use case:\n" + args.use_case
    )
    text = get_llm_response(args.provider, prompt)
    print("\n=== PLAN ===\n")
    print(text.strip())


def cmd_generate(args: argparse.Namespace):
    run_generation(args.provider, args.use_case)


async def _run_multi(task: str, provider: str, max_turns: int):
    if not (BaseAgent and Orchestrator):
        print("Multi-agent components not available.")
        return 1
    if provider == 'echo':
        llm = EchoLLM()
    else:
        llm = SyncProviderLLM(provider)
    supervisor = BaseAgent("supervisor", llm, "Supervisor: choose NEXT:<agent> or FINISH:<answer>.")
    worker = BaseAgent("worker", llm, "Worker: solve tasks and RESPOND:<answer>.")
    orch = Orchestrator({"worker": worker}, supervisor, max_turns=max_turns)
    result = await orch.run(task)
    print("\nFinal Answer:\n", result.final_answer)
    return 0


def cmd_multi(args: argparse.Namespace):
    asyncio.run(_run_multi(args.task, args.provider, args.max_turns))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentforge", description="AgentForge CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("plan", help="Generate an architecture plan only")
    sp.add_argument("--provider", required=True)
    sp.add_argument("--use-case", required=True, dest="use_case")
    sp.set_defaults(func=cmd_plan)

    sg = sub.add_parser("generate", help="Full single-agent pipeline")
    sg.add_argument("--provider", required=True)
    sg.add_argument("--use-case", required=True, dest="use_case")
    sg.set_defaults(func=cmd_generate)

    sm = sub.add_parser("multi", help="Run multi-agent supervisor loop")
    sm.add_argument("--provider", required=True, help="openai|grok|groq|anthropic|ollama|echo")
    sm.add_argument("--task", required=True)
    sm.add_argument("--max-turns", type=int, default=10)
    sm.set_defaults(func=cmd_multi)
    return p


def main():  # entrypoint
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)  # type: ignore[attr-defined]
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
