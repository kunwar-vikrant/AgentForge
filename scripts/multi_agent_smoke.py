"""Manual smoke test for the multi-agent system without external LLM calls.

Usage:
  python scripts/multi_agent_smoke.py

It uses FakeLLM objects to simulate supervisor and worker decisions deterministically.
"""
import asyncio, sys
from pathlib import Path

# Add src to path for direct execution without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agents.base import BaseAgent, LLMInterface
from agents.orchestrator import Orchestrator


class FakeLLM(LLMInterface):
    def __init__(self, responses):
        self._responses = list(responses)

    async def complete(self, prompt: str) -> str:
        if not self._responses:
            return "FINISH:Exhausted"
        return self._responses.pop(0)


async def main():
    sup = BaseAgent("supervisor", FakeLLM(["NEXT:worker", "FINISH:Task solved"]), "Coordinate workers")
    worker = BaseAgent("worker", FakeLLM(["RESPOND:Interim reasoning", "RESPOND:Final output"]), "Execute tasks")
    orch = Orchestrator({"worker": worker}, sup, max_turns=5)
    result = await orch.run("Demonstrate multi-agent orchestration")
    print("Final answer:", result.final_answer)
    print("Turns:", result.turns, "Finished:", result.finished, "Reason:", result.finish_reason)
    print("Transcript:")
    for m in result.transcript:
        print(f"[{m.role}] {m.sender}: {m.content}")


if __name__ == "__main__":
    asyncio.run(main())
