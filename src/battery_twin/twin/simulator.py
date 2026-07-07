"""Battery digital twin core - physics layer.

This module wraps a physics-based lithium-ion battery model (an LG M50 21700
cell parameterised with the Chen2020 data set) using PyBaMM, and exposes a small
set of high-level operations the agent can reason over:

    * simulate_discharge          - a single constant-current discharge
    * simulate_cccv_charge        - a constant-current / constant-voltage charge
    * compare_charging_strategies - run several charge strategies and rank them
    * simulate_degradation        - capacity fade over N cycles (SEI growth)
    * cell_info                   - static description of the modelled cell

Design notes
------------
The agent (an LLM) consumes these results, so every function returns a plain,
JSON-serialisable ``dict`` containing:

    * scalar summary metrics (the part the LLM actually reasons about), and
    * a *downsampled* time series (so the LLM is never handed thousands of raw
      points).

The heavy numerical work happens here in deterministic Python/PyBaMM. The LLM is
deliberately kept out of the maths - it orchestrates and explains. That split is
what makes the system trustworthy: the numbers come from physics, not from a
language model.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pybamm

# Modelled cell. Chen2020 describes a commercial LG M50 21700 (~5 Ah) cell and
# is one of the best-characterised open parameter sets available.
PARAMETER_SET = "Chen2020"
NOMINAL_CAPACITY_AH = 5.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _parameters(temperature_c: float) -> pybamm.ParameterValues:
    """Return the Chen2020 parameter set at a given ambient temperature."""
    pv = pybamm.ParameterValues(PARAMETER_SET)
    pv["Ambient temperature [K]"] = 273.15 + float(temperature_c)
    return pv


def _downsample(t: np.ndarray, y: np.ndarray, n: int = 60) -> list[dict[str, float]]:
    """Reduce a time series to at most ``n`` evenly spaced points for the LLM."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size <= n:
        idx = np.arange(t.size)
    else:
        idx = np.linspace(0, t.size - 1, n).astype(int)
    return [{"t_h": round(float(t[i]), 4), "value": round(float(y[i]), 4)} for i in idx]


def _energy_wh(sol: pybamm.Solution) -> float:
    """Integrate terminal power over time to get delivered/stored energy [Wh]."""
    t_h = sol["Time [h]"].entries
    voltage = sol["Terminal voltage [V]"].entries
    current = sol["Current [A]"].entries
    power = voltage * current  # W
    # np.trapz was renamed to np.trapezoid in NumPy 2.0.
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(abs(trapezoid(power, t_h)))


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #
def cell_info() -> dict[str, Any]:
    """Static description of the modelled cell and the underlying model."""
    return {
        "chemistry": "Lithium-ion (NMC811 / graphite-SiOx)",
        "format": "21700 cylindrical",
        "reference_cell": "LG M50",
        "parameter_set": PARAMETER_SET,
        "nominal_capacity_Ah": NOMINAL_CAPACITY_AH,
        "nominal_voltage_V": 3.63,
        "voltage_window_V": [2.5, 4.2],
        "models": {
            "performance": "SPMe (Single Particle Model with electrolyte)",
            "degradation": "SPM + solvent-diffusion-limited SEI growth",
        },
    }


def simulate_discharge(
    c_rate: float = 1.0,
    temperature_c: float = 25.0,
    initial_soc: float = 1.0,
) -> dict[str, Any]:
    """Constant-current discharge from ``initial_soc`` down to the 2.5 V cut-off.

    Parameters
    ----------
    c_rate:
        Discharge rate as a multiple of nominal capacity (1C ~= 5 A here).
    temperature_c:
        Ambient temperature in degrees Celsius.
    initial_soc:
        Starting state of charge, 0-1.
    """
    t0 = time.time()
    model = pybamm.lithium_ion.SPMe()
    pv = _parameters(temperature_c)
    experiment = pybamm.Experiment([f"Discharge at {c_rate}C until 2.5V"])
    sim = pybamm.Simulation(model, parameter_values=pv, experiment=experiment)
    sol = sim.solve(initial_soc=float(initial_soc))

    t_h = sol["Time [h]"].entries
    voltage = sol["Terminal voltage [V]"].entries
    capacity_ah = float(sol["Discharge capacity [A.h]"].entries[-1])

    return {
        "operation": "discharge",
        "inputs": {
            "c_rate": c_rate,
            "temperature_c": temperature_c,
            "initial_soc": initial_soc,
        },
        "metrics": {
            "delivered_capacity_Ah": round(capacity_ah, 4),
            "delivered_energy_Wh": round(_energy_wh(sol), 4),
            "duration_h": round(float(t_h[-1]), 4),
            "min_voltage_V": round(float(voltage.min()), 4),
            "capacity_utilisation_pct": round(100 * capacity_ah / NOMINAL_CAPACITY_AH, 2),
        },
        "voltage_curve": _downsample(t_h, voltage),
        "solve_time_s": round(time.time() - t0, 3),
    }


def simulate_cccv_charge(
    c_rate: float = 0.5,
    temperature_c: float = 25.0,
    voltage_limit: float = 4.2,
    cutoff_c_rate: float = 0.05,
    initial_soc: float = 0.05,
) -> dict[str, Any]:
    """A constant-current / constant-voltage (CC-CV) charge.

    Charges at ``c_rate`` until ``voltage_limit``, then holds that voltage until
    the current tapers to ``cutoff_c_rate`` (a C-rate). This is how real chargers
    fill a cell.
    """
    t0 = time.time()
    cutoff_a = cutoff_c_rate * NOMINAL_CAPACITY_AH
    model = pybamm.lithium_ion.SPMe()
    pv = _parameters(temperature_c)
    experiment = pybamm.Experiment(
        [
            (
                f"Charge at {c_rate}C until {voltage_limit}V",
                f"Hold at {voltage_limit}V until {cutoff_a}A",
            )
        ]
    )
    sim = pybamm.Simulation(model, parameter_values=pv, experiment=experiment)
    sol = sim.solve(initial_soc=float(initial_soc))

    t_h = sol["Time [h]"].entries
    voltage = sol["Terminal voltage [V]"].entries

    return {
        "operation": "cccv_charge",
        "inputs": {
            "c_rate": c_rate,
            "temperature_c": temperature_c,
            "voltage_limit_V": voltage_limit,
            "cutoff_c_rate": cutoff_c_rate,
            "initial_soc": initial_soc,
        },
        "metrics": {
            "charge_time_min": round(float(t_h[-1]) * 60, 2),
            "energy_in_Wh": round(_energy_wh(sol), 4),
            "max_voltage_V": round(float(voltage.max()), 4),
        },
        "voltage_curve": _downsample(t_h, voltage),
        "solve_time_s": round(time.time() - t0, 3),
    }


def compare_charging_strategies(
    strategies: list[dict[str, Any]] | None = None,
    temperature_c: float = 25.0,
) -> dict[str, Any]:
    """Run several CC-CV charge strategies and rank them by charge time.

    ``strategies`` is a list of dicts, each like ``{"name": ..., "c_rate": ...}``.
    If omitted, a sensible default 0.3C / 0.5C / 1.0C comparison is used.
    """
    if not strategies:
        strategies = [
            {"name": "gentle_0p3C", "c_rate": 0.3},
            {"name": "standard_0p5C", "c_rate": 0.5},
            {"name": "fast_1C", "c_rate": 1.0},
        ]

    results = []
    for s in strategies:
        out = simulate_cccv_charge(
            c_rate=float(s.get("c_rate", 0.5)),
            temperature_c=temperature_c,
        )
        results.append(
            {
                "name": s.get("name", f"{s.get('c_rate')}C"),
                "c_rate": s.get("c_rate", 0.5),
                "charge_time_min": out["metrics"]["charge_time_min"],
                "energy_in_Wh": out["metrics"]["energy_in_Wh"],
            }
        )

    fastest = min(results, key=lambda r: r["charge_time_min"])
    return {
        "operation": "compare_charging_strategies",
        "temperature_c": temperature_c,
        "results": results,
        "fastest": fastest["name"],
        "note": (
            "Faster charging reduces time but, in a degradation-aware study, also "
            "raises stress (heat, plating risk). Use simulate_degradation to weigh "
            "the trade-off over cycle life."
        ),
    }

def compare_discharge_rates(
    c_rates: list[float] | None = None,
    temperature_c: float = 25.0,
) -> dict[str, Any]:
    """Discharge the cell at several C-rates and compare runtime and capacity.

    This runs one discharge per rate in a single call, so an agent can answer
    "compare 2C versus 0.5C" without having to orchestrate multiple tool calls
    itself.

    Args:
        c_rates: list of discharge rates to compare. Defaults to [0.5, 1.0, 2.0].
        temperature_c: ambient temperature in Celsius.
    """
    if not c_rates:
        c_rates = [0.5, 1.0, 2.0]

    results = []
    for cr in c_rates:
        out = simulate_discharge(c_rate=float(cr), temperature_c=temperature_c)
        m = out["metrics"]
        results.append(
            {
                "c_rate": float(cr),
                "delivered_capacity_Ah": m["delivered_capacity_Ah"],
                "delivered_energy_Wh": m["delivered_energy_Wh"],
                "duration_h": m["duration_h"],
                "duration_min": round(m["duration_h"] * 60, 1),
            }
        )

    longest = max(results, key=lambda r: r["duration_h"])
    shortest = min(results, key=lambda r: r["duration_h"])
    return {
        "operation": "compare_discharge_rates",
        "temperature_c": temperature_c,
        "results": results,
        "longest_runtime_c_rate": longest["c_rate"],
        "shortest_runtime_c_rate": shortest["c_rate"],
        "note": (
            "Higher C-rates discharge faster and deliver slightly less usable "
            "capacity (rate capability). Capacity above the 5.0 Ah nominal label "
            "at gentle rates is normal - nominal is a conservative rating."
        ),
    }

def simulate_degradation(
    n_cycles: int = 10,
    discharge_c_rate: float = 1.0,
    charge_c_rate: float = 0.5,
    temperature_c: float = 25.0,
) -> dict[str, Any]:
    """Capacity fade over ``n_cycles`` full cycles via SEI growth.

    Returns capacity per cycle, total fade, and the dominant loss mechanism
    (loss of lithium inventory). ``n_cycles`` is capped at 50 to keep runtime
    sane on a laptop / 12 GB box.
    """
    t0 = time.time()
    n_cycles = int(max(1, min(n_cycles, 50)))

    model = pybamm.lithium_ion.SPM({"SEI": "solvent-diffusion limited"})
    pv = _parameters(temperature_c)
    cycle = (
        f"Discharge at {discharge_c_rate}C until 3.0V",
        f"Charge at {charge_c_rate}C until 4.2V",
        "Hold at 4.2V until 50mA",
    )
    experiment = pybamm.Experiment([cycle] * n_cycles)
    sim = pybamm.Simulation(model, parameter_values=pv, experiment=experiment)
    sol = sim.solve()

    sv = sol.summary_variables
    cap = [round(float(c), 5) for c in sv["Capacity [A.h]"]]
    lli = [round(float(x), 5) for x in sv["Loss of lithium inventory [%]"]]

    cap0, capN = cap[0], cap[-1]
    fade_pct = round(100 * (cap0 - capN) / cap0, 4) if cap0 else 0.0

    return {
        "operation": "degradation",
        "inputs": {
            "n_cycles": n_cycles,
            "discharge_c_rate": discharge_c_rate,
            "charge_c_rate": charge_c_rate,
            "temperature_c": temperature_c,
        },
        "metrics": {
            "initial_capacity_Ah": cap0,
            "final_capacity_Ah": capN,
            "capacity_fade_pct": fade_pct,
            "fade_per_cycle_pct": round(fade_pct / n_cycles, 5),
            "final_LLI_pct": lli[-1],
            "dominant_mechanism": "Loss of lithium inventory (SEI growth)",
        },
        "capacity_per_cycle_Ah": cap,
        "lli_per_cycle_pct": lli,
        "solve_time_s": round(time.time() - t0, 3),
    }


# Map of tool name -> callable, used by the MCP server and the offline test.
TOOLS = {
    "cell_info": cell_info,
    "simulate_discharge": simulate_discharge,
    "simulate_cccv_charge": simulate_cccv_charge,
    "compare_charging_strategies": compare_charging_strategies,
    "simulate_degradation": simulate_degradation,
}
