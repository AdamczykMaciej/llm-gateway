#!/usr/bin/env python3
"""A ReAct-style, tool-using agent built with LangGraph, running entirely
through llm-gateway — no custom integration code needed on the LangGraph
side. The gateway's /v1/chat/completions is OpenAI-wire-compatible
(including `tools`/`tool_calls`), so LangChain's ChatOpenAI just points its
base_url at the gateway like it would at any OpenAI-compatible endpoint.

Setup:
    pip install -r examples/requirements.txt

Usage:
    LLM_GATEWAY_URL=https://llm-gateway-x6vjkbzkda-uc.a.run.app/v1 \
    LLM_GATEWAY_KEY=... \
    python examples/langgraph_react_agent.py "What's the weather in Warsaw?"

Defaults to http://localhost:8080/v1 (a locally-run `llm-gateway` process)
if LLM_GATEWAY_URL isn't set.
"""

import os
import sys

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    # Deterministic mock so this example needs no external API/network —
    # swap in a real weather API call for actual use.
    fake_forecasts = {
        "warsaw": "22C, sunny",
        "london": "15C, overcast",
        "san francisco": "18C, foggy",
    }
    return fake_forecasts.get(
        city.strip().lower(), f"No data for '{city}', assume mild and cloudy."
    )


def build_agent():
    llm = ChatOpenAI(
        base_url=os.environ.get("LLM_GATEWAY_URL", "http://localhost:8080/v1"),
        api_key=os.environ.get("LLM_GATEWAY_KEY", "not-needed-if-gateway-has-no-keys-configured"),
        model="auto",  # or e.g. "anthropic/claude-sonnet-4-6" to force one
    )
    return create_react_agent(model=llm, tools=[get_weather])


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What's the weather in Warsaw and London?"
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    for message in result["messages"]:
        role = getattr(message, "type", message.get("role") if isinstance(message, dict) else "?")
        content = getattr(
            message, "content", message.get("content") if isinstance(message, dict) else ""
        )
        if content:
            print(f"[{role}] {content}")


if __name__ == "__main__":
    main()
