"""The agentic layer: a LangGraph ReAct agent driven by a local Ollama model,
using the battery digital twin through MCP.

Flow
----
1. Launch the twin MCP server (stdio) via ``MultiServerMCPClient``.
2. Pull its tools and adapt them into LangChain tools.
3. Build a ReAct agent around a local Ollama chat model.
4. The model decides which twin tools to call, reads the physics results, and
   explains them.

The LLM never does the maths - it orchestrates the twin and narrates the result.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from battery_twin.agent import config


def _connections() -> dict:
    """MCP server connection map (stdio transport).

    We pass through the current environment plus a PYTHONPATH entry for ``src`` so
    the server subprocess can import the package even without a pip install.
    """
    import os
    from pathlib import Path

    src = str(Path(__file__).resolve().parents[2])  # .../src
    env = dict(os.environ)
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")

    return {
        "battery_twin": {
            "command": config.TWIN_SERVER_COMMAND,
            "args": config.TWIN_SERVER_ARGS,
            "transport": "stdio",
            "env": env,
        }
    }


async def build_agent():
    """Create the ReAct agent wired to the twin's MCP tools.

    Returns ``(agent, tool_names)``.
    """
    client = MultiServerMCPClient(_connections())
    tools = await client.get_tools()

    model = ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.OLLAMA_TEMPERATURE,
    )
    agent = create_react_agent(model, tools)
    return agent, [t.name for t in tools]


async def ask(agent, question: str) -> str:
    """Send one question to the agent and return its final text answer."""
    result = await agent.ainvoke(
        {
            "messages": [
                SystemMessage(content=config.SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]
        }
    )
    return result["messages"][-1].content
