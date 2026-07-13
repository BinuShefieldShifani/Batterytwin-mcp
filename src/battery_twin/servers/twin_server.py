"""MCP server exposing the battery digital twin as standard MCP tools.

This is the heart of the "MCP" part of the project. Instead of hard-wiring the
PyBaMM functions into one agent, we expose them as a Model Context Protocol
server. *Any* MCP-capable client - this project's LangGraph agent, Claude
Desktop, an IDE, another team's orchestrator - can discover and call these tools
over a standard protocol. The twin becomes a reusable, vendor-neutral capability
rather than a one-off script.

Run directly over stdio (how the agent launches it):

    python -m battery_twin.servers.twin_server

The tool docstrings below are sent to the LLM as the tool descriptions, so they
are written to be read by a model deciding which tool to call.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from battery_twin.twin import simulator

mcp = FastMCP("battery-twin")


@mcp.tool()
def cell_info() -> dict[str, Any]:
    """Return a static description of the modelled lithium-ion cell.

    Use this first when you need to know the chemistry, format, nominal capacity,
    or voltage window before reasoning about a request.
    """
    return simulator.cell_info()


@mcp.tool()
def simulate_discharge(
    c_rate: float = 1.0,
    temperature_c: float = 25.0,
    initial_soc: float = 1.0,
) -> dict[str, Any]:
    """Simulate a constant-current discharge to the 2.5 V cut-off.

    Args:
        c_rate: Discharge rate as a multiple of capacity (1C is ~5 A).
        temperature_c: Ambient temperature in Celsius.
        initial_soc: Starting state of charge, 0.0-1.0.

    Returns delivered capacity/energy, duration, and a downsampled voltage curve.
    Use this to answer questions about runtime, rate capability, or how
    temperature affects deliverable capacity.
    """
    return simulator.simulate_discharge(c_rate, temperature_c, initial_soc)


@mcp.tool()
def simulate_cccv_charge(
    c_rate: float = 0.5,
    temperature_c: float = 25.0,
    voltage_limit: float = 4.2,
    cutoff_c_rate: float = 0.05,
    initial_soc: float = 0.05,
) -> dict[str, Any]:
    """Simulate a CC-CV (constant-current then constant-voltage) charge.

    Use this to answer questions about charge time or energy taken in at a given
    charge rate. Returns charge time in minutes plus a voltage curve.
    """
    return simulator.simulate_cccv_charge(
        c_rate, temperature_c, voltage_limit, cutoff_c_rate, initial_soc
    )


@mcp.tool()
def compare_charging_strategies(
    strategies: list[dict[str, Any]] | None = None,
    temperature_c: float = 25.0,
) -> dict[str, Any]:
    """Run and rank multiple charging strategies by charge time.

    Args:
        strategies: list of {"name": str, "c_rate": float}. If omitted, compares
            0.3C / 0.5C / 1.0C.
        temperature_c: Ambient temperature in Celsius.

    Use this when the user asks which charging approach is best or wants a
    trade-off between speed and battery stress.
    """
    return simulator.compare_charging_strategies(strategies, temperature_c)


@mcp.tool()
def compare_discharge_rates(
    c_rates: list[float] | None = None,
    temperature_c: float = 25.0,
) -> dict[str, Any]:
    """Discharge at several C-rates and compare runtime and delivered capacity.

    Use this whenever the user wants to compare discharge behaviour across more
    than one C-rate (e.g. "2C versus 0.5C"). It runs every rate in ONE call and
    returns runtime and capacity for each, so you do not need to call
    simulate_discharge multiple times yourself.

    Args:
        c_rates: list of discharge rates to compare, e.g. [2.0, 0.5].
        temperature_c: ambient temperature in Celsius.
    """
    return simulator.compare_discharge_rates(c_rates, temperature_c)


@mcp.tool()
def simulate_degradation(
    n_cycles: int = 10,
    discharge_c_rate: float = 1.0,
    charge_c_rate: float = 0.5,
    temperature_c: float = 25.0,
) -> dict[str, Any]:
    """Simulate capacity fade over N full cycles using SEI-growth physics.

    Args:
        n_cycles: number of cycles to run (capped at 50).
        discharge_c_rate: discharge rate per cycle.
        charge_c_rate: charge rate per cycle.
        temperature_c: ambient temperature in Celsius (higher accelerates fade).

    Returns capacity per cycle, total capacity fade %, and the dominant loss
    mechanism. Use this for cycle-life, ageing, or "how fast will it wear out"
    questions, and to weigh fast-charging against longevity.
    """
    return simulator.simulate_degradation(
        n_cycles, discharge_c_rate, charge_c_rate, temperature_c
    )


def main() -> None:
    """Entry point: serve the twin over stdio (the transport the agent uses)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()