"""SearchAgent：负责搜索、资料收集和来源整理。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from fishclaw.agents.common import Writer, last_ai_text, tool_event
from fishclaw.agents.prompts import SEARCH_PROMPT
from fishclaw.memory import build_memory, format_memory, merge_sources
from fishclaw.model import create_model
from fishclaw.state import FishState
from fishclaw.tools.harness import build_search_tool, tool_result_json


def run_search_agent(state: FishState, instruction: str, *, writer: Writer | None = None) -> dict[str, Any]:
    """运行 SearchAgent，并返回研究摘要和来源。"""
    writer = writer or (lambda _: None)
    model = create_model().bind_tools([build_search_tool()])
    messages: list[Any] = [
        SystemMessage(content=SEARCH_PROMPT),
        HumanMessage(content=f"用户任务：{state.get('task', '')}\n\nplanner 指令：{instruction}\n\n上下文：\n{format_memory(build_memory(state))}"),
    ]
    produced: list[Any] = []
    sources: list[dict[str, Any]] = []
    answers: list[str] = []
    for _ in range(3):
        response = model.invoke(messages)
        produced.append(response)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for call in tool_calls:
            writer({"type": "tool_call", "node": "searchAgent", "name": call.get("name"), "args": call.get("args", {})})
            tool_message = _execute_search_tool(call)
            produced.append(tool_message)
            messages.append(tool_message)
            event = tool_event("searchAgent", tool_message)
            writer(event)
            parsed = event.get("result") if isinstance(event.get("result"), dict) else {}
            if isinstance(parsed, dict):
                if parsed.get("answer"):
                    answers.append(str(parsed.get("answer")))
                sources.extend([item for item in parsed.get("results", []) or [] if isinstance(item, dict)])
    summary = last_ai_text(produced) or "\n".join(answers)
    return {"ok": True, "summary": summary, "sources": merge_sources(sources), "messages": produced}


def _execute_search_tool(call: dict[str, Any]) -> ToolMessage:
    """执行 SearchAgent 的 WebSearchTool 调用。"""
    tool = build_search_tool()
    name = str(call.get("name", ""))
    args = call.get("args") or {}
    if name != tool.name:
        result = {"ok": False, "error": f"unknown tool: {name}"}
    else:
        try:
            result = tool.invoke(args)
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return ToolMessage(content=tool_result_json(result), name=name, tool_call_id=call.get("id") or f"{name}-call")

