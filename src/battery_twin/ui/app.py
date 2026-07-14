"""Streamlit UI for the battery digital twin agent.

A simple chat interface plus a sidebar of one-click example questions. The agent
runs the same MCP + Ollama pipeline as the CLI.

Run:
    streamlit run src/battery_twin/ui/app.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import streamlit as st

from battery_twin.agent import config
from battery_twin.agent.agent import ask, build_agent

st.set_page_config(page_title="Battery Digital Twin (MCP)", page_icon="(battery)", layout="centered")


def _run(coro):
    """Run an async coroutine from Streamlit's sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@st.cache_resource(show_spinner="Connecting to the digital twin via MCP...")
def get_agent():
    return _run(build_agent())


st.title("Battery Digital Twin")
st.caption("Physics-based lithium-ion twin (PyBaMM) exposed over MCP, driven by a local Ollama model.")

with st.sidebar:
    st.subheader("Setup")
    st.write(f"**Model:** `{config.OLLAMA_MODEL}`")
    st.write(f"**Ollama:** `{config.OLLAMA_BASE_URL}`")
    st.divider()
    st.subheader("Try asking")
    examples = [
        "What cell are we modelling?",
        "How long does it run discharging at 2C versus 0.5C?",
        "Which is faster to charge: 0.3C, 0.5C or 1C?",
        "How much capacity is lost over 10 cycles at 40 C?",
        "Does a cold 0 C discharge reduce deliverable capacity?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["pending"] = ex

agent, tool_names = get_agent()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask the twin about discharge, charging, temperature, or ageing...")
if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Running the twin..."):
            answer = _run(ask(agent, prompt))
        st.markdown(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
