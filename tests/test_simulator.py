"""Unit tests for the physics layer. No LLM, no GPU - pure PyBaMM.

Run:
    pytest -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from battery_twin.twin import simulator as s


def test_cell_info():
    info = s.cell_info()
    assert info["parameter_set"] == "Chen2020"
    assert info["voltage_window_V"] == [2.5, 4.2]


def test_discharge_basic():
    r = s.simulate_discharge(c_rate=1.0)
    m = r["metrics"]
    # A healthy ~5 Ah cell should deliver most of its capacity at 1C.
    assert 4.0 < m["delivered_capacity_Ah"] < 5.2
    assert m["min_voltage_V"] <= 2.6
    assert len(r["voltage_curve"]) > 1


def test_higher_c_rate_delivers_less():
    low = s.simulate_discharge(c_rate=0.5)["metrics"]["delivered_capacity_Ah"]
    high = s.simulate_discharge(c_rate=2.0)["metrics"]["delivered_capacity_Ah"]
    # Rate capability: faster discharge yields slightly less usable capacity.
    assert high < low


def test_compare_ranks_fastest():
    r = s.compare_charging_strategies()
    times = {x["name"]: x["charge_time_min"] for x in r["results"]}
    # The highest C-rate should be the fastest charge.
    assert r["fastest"] == min(times, key=times.get)


def test_degradation_loses_capacity():
    r = s.simulate_degradation(n_cycles=5)
    m = r["metrics"]
    assert m["final_capacity_Ah"] <= m["initial_capacity_Ah"]
    assert m["capacity_fade_pct"] >= 0.0
    assert len(r["capacity_per_cycle_Ah"]) == 5
