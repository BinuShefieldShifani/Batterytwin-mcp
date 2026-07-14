# Architecture

## Overview

The system has three layers with a deliberate separation of concerns: physics is
deterministic, the protocol is standard, and the LLM only orchestrates.

```
            +---------------------------------------------------+
            |                 Clients (any MCP host)            |
            |   Streamlit UI  |  CLI  |  Claude Desktop  | ...   |
            +-------------------------+-------------------------+
                                      |
                          natural-language question
                                      |
            +-------------------------v-------------------------+
            |        Agentic layer  (LangGraph ReAct)           |
            |        local LLM via Ollama (tool calling)        |
            |   decides WHICH tool to call, reads results,      |
            |   explains them - never does the maths itself     |
            +-------------------------+-------------------------+
                                      |
                       Model Context Protocol (stdio)
                                      |
            +-------------------------v-------------------------+
            |            MCP server  (FastMCP)                  |
            |   cell_info | simulate_discharge |               |
            |   simulate_cccv_charge | compare_charging |       |
            |   simulate_degradation                           |
            +-------------------------+-------------------------+
                                      |
                              Python function calls
                                      |
            +-------------------------v-------------------------+
            |          Physics layer  (PyBaMM digital twin)     |
            |   LG M50 (Chen2020): SPMe performance model +     |
            |   SPM/SEI degradation model                       |
            +---------------------------------------------------+
```

## Why MCP, not just a LangGraph agent

The twin is exposed as a **Model Context Protocol** server, not hard-wired into
the agent. That is the whole point:

- **Reusable.** The same twin server can be driven by this project's agent, by
  Claude Desktop, by an IDE, or by another team's orchestrator - no code changes.
- **Vendor-neutral.** Any MCP-capable client speaks to it over a standard
  protocol.
- **Composable.** Adding a second capability (e.g. a thermal model or a
  standards knowledge base) is just another MCP server the agent can mount.
- **Air-gap friendly.** Everything - model (Ollama), protocol (stdio), and
  physics (PyBaMM) - runs locally with no external calls.

## Why the LLM does not do the maths

All numerical work happens in PyBaMM. The LLM picks tools and explains results.
This is what makes the answers trustworthy: the figures come from a validated
electrochemical model, and a small local model (7-8B) is perfectly capable of
orchestration and narration even though it could not compute battery physics on
its own. It also means the system runs comfortably on a 12 GB GPU.

## The modelled cell

An LG M50 21700 cell described by the open **Chen2020** parameter set:

- ~5 Ah nominal capacity, 2.5-4.2 V window.
- **Performance:** SPMe (Single Particle Model with electrolyte) - fast and
  accurate for discharge/charge behaviour.
- **Degradation:** SPM with solvent-diffusion-limited SEI growth - models
  capacity fade and loss of lithium inventory over cycles.

## Extending the system

- Add a `thermal_server` exposing a lumped thermal model; the agent can then
  reason about temperature rise during fast charging.
- Add a `knowledge_server` (ChromaDB RAG over battery datasheets / standards) so
  the agent can ground recommendations in documentation as well as simulation.
- Swap the cell by changing the parameter set in `simulator.py`.
