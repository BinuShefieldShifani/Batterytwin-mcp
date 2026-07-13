"""Interactive CLI for the battery-twin agent.

Talk to the digital twin in natural language. The agent decides which MCP tools
to call, runs the physics, and explains the result.

Requires a running Ollama with a tool-capable model pulled (see README).

Run:
    python scripts/chat.py
    python scripts/chat.py "How long does the cell run at 2C and 0C?"
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from battery_twin.agent import config
from battery_twin.agent.agent import ask, build_agent


async def main() -> int:
    print("Booting battery-twin agent...")
    print(f"  model     : {config.OLLAMA_MODEL}")
    print(f"  ollama    : {config.OLLAMA_BASE_URL}")
    agent, tool_names = await build_agent()
    print(f"  mcp tools : {', '.join(tool_names)}\n")

    # One-shot mode if a question was passed on the command line.
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"You: {question}\n")
        print(f"Agent: {await ask(agent, question)}")
        return 0

    print("Type a question (or 'exit').\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit", "q", ""}:
            break
        answer = await ask(agent, question)
        print(f"\nAgent: {answer}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
