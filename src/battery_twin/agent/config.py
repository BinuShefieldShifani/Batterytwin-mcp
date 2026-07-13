"""Configuration for the battery-twin agent.

All settings come from environment variables (see ``.env.example``) so the same
code runs locally and in Docker without edits.
"""

from __future__ import annotations

import os
import sys

# Ollama endpoint. In Docker Compose the agent reaches the Ollama container at
# http://ollama:11434; locally it defaults to localhost.
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# The model used for orchestration. It MUST support tool/function calling, so a
# >=7B instruct model is recommended (e.g. llama3.1:8b, qwen2.5:7b). Tiny models
# such as tinyllama do not reliably emit tool calls and will not drive the twin.
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Sampling temperature - low, because we want deterministic tool orchestration.
OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

# Command used to launch the twin MCP server over stdio.
TWIN_SERVER_COMMAND: str = os.getenv("TWIN_SERVER_COMMAND", sys.executable)
TWIN_SERVER_ARGS: list[str] = os.getenv(
    "TWIN_SERVER_ARGS", "-m battery_twin.servers.twin_server"
).split()

SYSTEM_PROMPT: str = (
    "You are a battery engineering assistant connected to a physics-based digital "
    "twin of a lithium-ion cell (an LG M50 21700) through MCP tools. When a question "
    "depends on the cell's behaviour - runtime, charge time, rate capability, "
    "temperature effects, or ageing - you MUST call the appropriate tool and base "
    "your answer only on the numbers it returns. "
    "To COMPARE discharge across several C-rates, use compare_discharge_rates with all "
    "the rates in one call. To compare charging, use compare_charging_strategies. "
    "Prefer these single-call comparison tools over calling simulate_discharge or "
    "simulate_cccv_charge repeatedly. "
    "Never invent figures, and do not speculate about causes the tools do not report "
    "(for example, do not call a value 'inefficiency', 'over-discharge', or 'wear and "
    "tear' unless a tool reports it). A delivered capacity above the 5.0 Ah nominal "
    "label is normal at gentle rates - nominal is a conservative rating, not a hard "
    "ceiling. After getting results, explain them plainly: the key numbers, what they "
    "mean, and any trade-off such as fast charging versus cycle life. If a question "
    "needs no simulation, answer directly."
)
