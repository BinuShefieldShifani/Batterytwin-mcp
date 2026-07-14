# Battery Digital Twin x MCP

> An agentic battery-engineering assistant. A physics-based lithium-ion **digital twin**
> (PyBaMM) is exposed over the **Model Context Protocol (MCP)** and driven by a **local
> LLM** (Ollama) through a **LangGraph** agent. Ask questions in plain English; the agent
> runs real electrochemical simulations and explains the results.

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="MCP" src="https://img.shields.io/badge/protocol-MCP-7b3fe4">
  <img alt="PyBaMM" src="https://img.shields.io/badge/physics-PyBaMM-2ea44f">
  <img alt="Ollama" src="https://img.shields.io/badge/LLM-Ollama%20(local)-000000">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## Why this project

Most "AI + battery" demos either (a) let an LLM hallucinate numbers, or (b) bury a
simulator behind a one-off script. This project does neither:

- **The physics is real and deterministic.** Every number comes from PyBaMM, a
  validated electrochemical modelling library - not from the language model.
- **The capability is reusable, not hard-wired.** The twin is an **MCP server**, so
  any MCP client (this agent, Claude Desktop, an IDE, another team's orchestrator) can
  use it over a standard protocol.
- **It runs fully local.** Local LLM (Ollama), local protocol (stdio), local physics.
  No external API calls - air-gap friendly and comfortable on a **12 GB GPU**.

The LLM's only job is orchestration: decide which tool to call, then explain the
result. That division - **physics for the maths, LLM for the language** - is what makes
the answers trustworthy.

## What it can do

Ask things like:

- *"How long does the cell run at 2C versus 0.5C?"*
- *"Which is fastest to charge - 0.3C, 0.5C, or 1C - and what do I trade off?"*
- *"How much capacity is lost over 10 cycles at 40 C?"*
- *"Does a cold 0 C discharge reduce deliverable capacity?"*

The agent calls the right simulation tool(s), reads the physics, and answers with the
actual figures.

## Architecture

```
 Streamlit UI / CLI / any MCP client
              |  natural language
              v
 LangGraph ReAct agent  +  local Ollama model   (chooses tools, explains results)
              |  Model Context Protocol (stdio)
              v
 MCP server (FastMCP)  ->  6 tools
              |  Python calls
              v
 PyBaMM digital twin  (LG M50, Chen2020)  ->  the actual electrochemistry
```

Full details in [`docs/architecture.md`](docs/architecture.md).

### MCP tools exposed

| Tool | What it simulates |
| --- | --- |
| `cell_info` | Static description of the modelled cell |
| `simulate_discharge` | Constant-current discharge (runtime, delivered capacity/energy) |
| `simulate_cccv_charge` | CC-CV charge (charge time, energy in) |
| `compare_charging_strategies` | Runs and ranks multiple charge rates in one call |
| `compare_discharge_rates` | Runs and compares multiple discharge rates in one call |
| `simulate_degradation` | Capacity fade over N cycles via SEI growth |

## Tech stack

| Layer | Technology |
| --- | --- |
| Physics / digital twin | [PyBaMM](https://pybamm.org) (LG M50, Chen2020 parameter set) |
| Tool protocol | [Model Context Protocol](https://modelcontextprotocol.io) (FastMCP) |
| Agent | [LangGraph](https://github.com/langchain-ai/langgraph) ReAct + `langchain-mcp-adapters` |
| LLM (local) | [Ollama](https://ollama.com) - `qwen2.5:7b` recommended |
| UI | Streamlit (optional) |

---

## Quick start

You can run **with Docker** (Ollama in a container) or **locally**. The fastest way to
verify everything works needs **no LLM at all** - see
[Verify without an LLM](#verify-the-twin-without-an-llm).

### Prerequisites

- Python 3.10+ (for local runs) **or** Docker + Docker Compose.
- A GPU is optional. The LLM runs on a ~5 GB quantized model; PyBaMM runs on CPU.

### Option A - Docker

```bash
# 1. Start the Ollama model server (GPU if available, see note below)
docker compose up -d ollama

# 2. Pull a tool-capable model once (~4.7 GB)
docker compose exec ollama ollama pull qwen2.5:7b

# 3. Start the agent UI
docker compose up app
# -> open http://localhost:8501
```

> **No GPU?** Delete the `deploy:` block under the `ollama` service in
> `docker-compose.yml`; it will run on CPU (slower but functional).
> The GPU path needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Option B - Local Python

```bash
# 1. Install (editable)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[ui]"

# 2. Start Ollama and pull a model (https://ollama.com)
ollama pull qwen2.5:7b

# 3. Configure (optional - defaults are fine for local)
cp .env.example .env
export OLLAMA_MODEL=qwen2.5:7b        # Windows: $env:OLLAMA_MODEL="qwen2.5:7b"

# 4a. Chat in the terminal
python scripts/chat.py
python scripts/chat.py "How long does it run at 2C and at 0.5C?"

# 4b. ...or launch the UI
streamlit run src/battery_twin/ui/app.py
```

---

## Verify the twin without an LLM

This drives the **MCP server directly** as a client and calls every tool - proving the
twin + protocol work, with **no Ollama and no GPU** needed:

```bash
pip install -e .
python scripts/test_twin.py
```

Run the unit tests on the physics layer:

```bash
pytest -q
```

---

## Configuration

All via environment variables (see [`.env.example`](.env.example)):

| Variable | Default | Notes |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Use `http://ollama:11434` inside Docker |
| `OLLAMA_MODEL` | `llama3.1:8b` | **Must support tool calling** (>=7B). `qwen2.5:7b` recommended. `tinyllama` will *not* drive the tools. |
| `OLLAMA_TEMPERATURE` | `0.1` | Low = deterministic orchestration |

## Project structure

```
battery-twin-mcp/
├── src/battery_twin/
│   ├── twin/simulator.py        # PyBaMM digital twin (the physics)
│   ├── servers/twin_server.py   # MCP server exposing twin tools (FastMCP)
│   ├── agent/agent.py           # LangGraph ReAct agent (MCP client + Ollama)
│   ├── agent/config.py          # env-driven configuration
│   └── ui/app.py                # Streamlit chat UI
├── scripts/
│   ├── test_twin.py             # offline MCP smoke test (no LLM)
│   └── chat.py                  # CLI chat
├── tests/test_simulator.py      # pytest (physics layer)
├── docs/architecture.md
├── examples/sample_questions.md
├── docker-compose.yml           # Ollama (GPU) + app
├── Dockerfile
└── pyproject.toml
```

---

## Issues encountered & fixes

1. **NumPy 2.0 removed `np.trapz`.** Energy integration raised `AttributeError`.
   *Fix:* use `np.trapezoid` with a `np.trapz` fallback for older NumPy.

2. **MCP server subprocess could not import the package** (`ModuleNotFoundError:
   battery_twin`) when launched over stdio.
   *Fix:* make the project pip-installable (`pip install -e .`) and pass `PYTHONPATH`
   to the server subprocess.

3. **`llama3.1:8b` printed tool calls as JSON text instead of executing them**, so no
   simulation ran.
   *Fix:* switch to `qwen2.5:7b`, which reliably emits structured tool calls.

4. **The agent speculated beyond the data** (e.g. labelling >100% capacity utilisation
   as "inefficiency / over-discharge").
   *Fix:* tighten the system prompt to forbid causal speculation the tools do not
   report, and state that capacity above the nominal label is normal at gentle rates.

5. **Multi-rate discharge comparisons dropped a rate.** Asking "2C vs 0.5C" only ran one
   discharge, because the 7B model would not chain two `simulate_discharge` calls.
   *Fix:* add a single-call `compare_discharge_rates` tool (mirroring
   `compare_charging_strategies`) so the comparison is one tool call the model handles
   reliably.

### Known limitation

Questions that require chaining *different* tools in sequence (e.g. charge comparison +
degradation run + cross-reasoning) may still drop a step on a 7B model. Use
`qwen2.5:14b` or add a planner step for those.

## License

MIT - see [`LICENSE`](LICENSE).