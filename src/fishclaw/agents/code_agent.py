"""CodeAgent：负责 workspace 内代码、文件和命令任务。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from fishclaw.agents.common import Writer, build_agent_summary, last_ai_text, tool_event
from fishclaw.agents.prompts import CODE_PROMPT
from fishclaw.memory import build_agent_memory, format_memory
from fishclaw.model import create_model
from fishclaw.state import FishState
from fishclaw.tools.harness import build_code_tools, tool_result_json


def run_code_agent(
    state: FishState,
    instruction: str,
    *,
    writer: Writer | None = None,
    active_task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """运行 CodeAgent，并返回实现摘要。"""
    writer = writer or (lambda _: None)
    runtime = state["runtime"]
    model = create_model().bind_tools(build_code_tools(runtime))
    messages: list[Any] = [
        SystemMessage(content=CODE_PROMPT),
        HumanMessage(
            content=(
                f"planner 指令：{instruction}\n\n"
                f"当前任务上下文：\n"
                f"{format_memory(build_agent_memory(state, instruction=instruction, active_task=active_task))}"
            )
        ),
    ]
    produced: list[Any] = []
    tool_events: list[dict[str, Any]] = []
    loop_limited = False
    for _ in range(runtime.max_agent_loops):
        response = model.invoke(messages)
        produced.append(response)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for call in tool_calls:
            writer({"type": "tool_call", "node": "codeAgent", "name": call.get("name"), "args": call.get("args", {})})
            tool_message = _execute_code_tool(state, call)
            event = tool_event("codeAgent", tool_message)
            tool_events.append(event)
            writer(event)
            produced.append(tool_message)
            messages.append(tool_message)
    else:
        loop_limited = True
        produced.append(AIMessage(content="CodeAgent 达到工具循环上限，已停止继续调用工具。"))
    summary = build_agent_summary("CodeAgent", last_ai_text(produced), tool_events, loop_limited=loop_limited)
    return {"ok": True, "summary": summary, "messages": produced, "tool_events": tool_events}


def _execute_code_tool(state: FishState, call: dict[str, Any]) -> ToolMessage:
    """执行 CodeAgent 的 harness 工具调用。"""
    tools = {tool.name: tool for tool in build_code_tools(state["runtime"])}
    name = str(call.get("name", ""))
    args = call.get("args") or {}
    tool = tools.get(name)
    if tool is None:
        result = {"ok": False, "error": f"unknown tool: {name}"}
    else:
        try:
            result = tool.invoke(args)
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return ToolMessage(content=tool_result_json(result), name=name, tool_call_id=call.get("id") or f"{name}-call")
