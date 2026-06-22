"""ReviewAgent：负责审查代码修改风险，不直接修改文件。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from fishclaw.agents.common import Writer, build_agent_summary, last_ai_text, tool_event
from fishclaw.agents.prompts import REVIEW_PROMPT
from fishclaw.memory import build_agent_memory, format_memory
from fishclaw.model import create_model
from fishclaw.state import FishRuntime, FishState
from fishclaw.tools.harness import file_read, grep, list_files, tool_result_json


def build_review_tools(runtime: FishRuntime) -> list[StructuredTool]:
    """构建 ReviewAgent 只读工具。"""
    return [
        StructuredTool.from_function(
            name="FileReadTool",
            func=lambda file_path, offset=0, limit=400: file_read(runtime, file_path, offset, limit),
            description="读取 workspace 内文本文件。参数：file_path, offset, limit。",
        ),
        StructuredTool.from_function(
            name="ListFilesTool",
            func=lambda path=".", recursive=False, max_entries=200, include_hidden=False: list_files(
                runtime, path, recursive, max_entries, include_hidden
            ),
            description="列出 workspace 内文件。参数：path, recursive, max_entries, include_hidden。",
        ),
        StructuredTool.from_function(
            name="GrepTool",
            func=lambda pattern, path=".", head_limit=50: grep(runtime, pattern, path, head_limit),
            description="在 workspace 内正则搜索。参数：pattern, path, head_limit。",
        ),
    ]


def run_review_agent(
    state: FishState,
    instruction: str,
    *,
    writer: Writer | None = None,
    active_task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """运行 ReviewAgent，并返回审查摘要。"""
    writer = writer or (lambda _: None)
    runtime = state["runtime"]
    model = create_model().bind_tools(build_review_tools(runtime))
    messages: list[Any] = [
        SystemMessage(content=REVIEW_PROMPT),
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
            writer({"type": "tool_call", "node": "reviewAgent", "name": call.get("name"), "args": call.get("args", {})})
            tool_message = _execute_review_tool(state, call)
            event = tool_event("reviewAgent", tool_message)
            tool_events.append(event)
            writer(event)
            produced.append(tool_message)
            messages.append(tool_message)
    else:
        loop_limited = True
        produced.append(AIMessage(content="ReviewAgent 达到工具循环上限，已停止继续调用工具。"))

    summary = build_agent_summary("ReviewAgent", last_ai_text(produced), tool_events, loop_limited=loop_limited)
    return {
        "ok": True,
        "summary": summary,
        "passed": _looks_review_passed(summary),
        "messages": produced,
        "tool_events": tool_events,
    }


def _execute_review_tool(state: FishState, call: dict[str, Any]) -> ToolMessage:
    """执行 ReviewAgent 的只读工具。"""
    tools = {tool.name: tool for tool in build_review_tools(state["runtime"])}
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


def _looks_review_passed(summary: str) -> bool:
    """粗略判断审查是否通过；后续可以改成结构化输出。"""
    lowered = summary.lower()
    negative_markers = ["严重问题", "阻塞", "必须修复", "failed", "fail", "bug", "风险较高"]
    return not any(marker in lowered for marker in negative_markers)
