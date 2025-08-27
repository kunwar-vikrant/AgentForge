"""Example script demonstrating multi-agent orchestration."""
import asyncio
from providers_async import get_async_provider
from agents.base import BaseAgent
from agents.tools import HttpGetTool, SleepTool
from agents.orchestrator import Orchestrator


class LLMWrapper:
    def __init__(self, provider: str):
        self.client = get_async_provider(provider)
    async def complete(self, prompt: str) -> str:
        return await self.client.complete(prompt)


async def main():
    provider = "ollama"  # change if desired
    researcher = BaseAgent(
        name="researcher",
        llm=LLMWrapper(provider),
        system_prompt="You gather factual information and extract key points.",
        tools=[HttpGetTool()]
    )
    planner = BaseAgent(
        name="planner",
        llm=LLMWrapper(provider),
        system_prompt="You design stepwise implementation approaches.",
        tools=[SleepTool()]
    )
    supervisor = BaseAgent(
        name="supervisor",
        llm=LLMWrapper(provider),
        system_prompt="Coordinate agents. Decide NEXT:<agent> or FINISH:<answer>."
    )

    orch = Orchestrator({"researcher": researcher, "planner": planner}, supervisor, max_turns=8)
    task = "Produce a concise 5-step plan for building a news summarization agent (no code)."
    result = await orch.run(task)
    print("--- Transcript ---")
    for line in result.transcript:
        print(line)
    print("\nFinal Answer:\n", result.final_answer)


if __name__ == "__main__":
    asyncio.run(main())