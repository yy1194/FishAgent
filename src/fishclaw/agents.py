"""Fishclaw 的 MultiAgent：SearchAgent、CodeAgent 和 planner 包装工具。"""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from fishclaw.harness import build_code_tools, build_search_tool, tool_result_json
from fishclaw.memory import build_memory, format_memory, merge_sources, short_text
from fishclaw.model import create_model
from fishclaw.state import FishState


Writer = Callable[[dict[str, Any]], None]


PLANNER_PROMPT = """你是 Fishclaw 的 planner 节点。

你的入口是用户任务。你只能通过 tool call 委派外部工作：
- SearchAgentTool：搜索、资料收集、来源整理。
- CodeAgentTool：读写文件、运行命令、检查结果。

工作规则：
- 需要资料或外部事实时调用 SearchAgentTool。
- 需要创建/修改/检查 workspace 文件时调用 CodeAgentTool。
- 每次只委派清楚、可执行的一段工作。
- 当你判断任务完成时，不要再调用工具，直接用简洁中文给出最终总结。
"""

SEARCH_PROMPT = """你是 Fishclaw 的 SearchAgent。
你只负责研究和来源整理，并对之前搜索得到的信息进行摘要总结。
必要时调用 WebSearchTool。
当你觉得研究和来源资料已经足够时，就停止调用工具。
最终回复必须包含简洁研究摘要和可用来源 URL。"""

CODE_PROMPT = """你是 Fishclaw 的 CodeAgent。
你只负责 workspace 内实现工作。使用 FileReadTool/FileWriteTool/GrepTool/BashTool。
要求：
- 编辑已有文件前必须先读取。
- 命令必须非交互、可验证。
- 完成后总结改了什么、检查了什么、还有什么风险。"""


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
            event = _tool_event("searchAgent", tool_message)
            writer(event)
            parsed = event.get("result") if isinstance(event.get("result"), dict) else {}
            if isinstance(parsed, dict):
                if parsed.get("answer"):
                    answers.append(str(parsed.get("answer")))
                sources.extend([item for item in parsed.get("results", []) or [] if isinstance(item, dict)])
    summary = _last_ai_text(produced) or "\n".join(answers)
    return {"ok": True, "summary": summary, "sources": merge_sources(sources), "messages": produced}


def run_code_agent(state: FishState, instruction: str, *, writer: Writer | None = None) -> dict[str, Any]:
    """运行 CodeAgent，并返回实现摘要。"""
    writer = writer or (lambda _: None)
    runtime = state["runtime"]
    model = create_model().bind_tools(build_code_tools(runtime))
    messages: list[Any] = [
        SystemMessage(content=CODE_PROMPT),
        HumanMessage(content=f"用户任务：{state.get('task', '')}\n\nplanner 指令：{instruction}\n\n上下文：\n{format_memory(build_memory(state))}"),
    ]
    produced: list[Any] = []
    tool_events: list[dict[str, Any]] = []
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
            event = _tool_event("codeAgent", tool_message)
            tool_events.append(event)
            writer(event)
            produced.append(tool_message)
            messages.append(tool_message)
    else:
        produced.append(AIMessage(content="CodeAgent 达到工具循环上限，已停止继续调用工具。"))
    return {"ok": True, "summary": _last_ai_text(produced), "messages": produced, "tool_events": tool_events}


def build_planner_tools(state: FishState, writer: Writer) -> list[StructuredTool]:
    """构建 planner 唯一可见的两个 agent 包装工具。"""
    return [
        StructuredTool.from_function(
            name="SearchAgentTool",
            func=lambda instruction: _call_search_agent(state, writer, instruction),
            description="委派搜索研究任务。参数：instruction。",
        ),
        StructuredTool.from_function(
            name="CodeAgentTool",
            func=lambda instruction: _call_code_agent(state, writer, instruction),
            description="委派 workspace 代码/文件/命令任务。参数：instruction。",
        ),
    ]


def _call_search_agent(state: FishState, writer: Writer, instruction: str) -> dict[str, Any]:
    """执行 SearchAgentTool，并把结果合并到 planner 工作状态。"""
    writer({"type": "handoff", "from": "planner", "to": "searchAgent", "instruction": instruction})
    result = run_search_agent(state, instruction, writer=writer)
    state["search_notes"] = "\n\n".join(part for part in [state.get("search_notes", ""), result.get("summary", "")] if part)
    state["sources"] = merge_sources(state.get("sources", []), result.get("sources", []))
    state["handoffs"] = state.get("handoffs", []) + [
        {"to": "searchAgent", "instruction": instruction, "summary": short_text(result.get("summary", ""), 600)}
    ]
    writer({"type": "handoff_result", "from": "searchAgent", "to": "planner", "summary": result.get("summary", "")})
    return {"ok": True, "summary": result.get("summary", ""), "sources": state.get("sources", [])}


def _call_code_agent(state: FishState, writer: Writer, instruction: str) -> dict[str, Any]:
    """执行 CodeAgentTool，并把实现摘要合并到 planner 工作状态。"""
    writer({"type": "handoff", "from": "planner", "to": "codeAgent", "instruction": instruction})
    result = run_code_agent(state, instruction, writer=writer)
    state["code_summary"] = "\n\n".join(part for part in [state.get("code_summary", ""), result.get("summary", "")] if part)
    state["handoffs"] = state.get("handoffs", []) + [
        {"to": "codeAgent", "instruction": instruction, "summary": short_text(result.get("summary", ""), 600)}
    ]
    writer({"type": "handoff_result", "from": "codeAgent", "to": "planner", "summary": result.get("summary", "")})
    return {"ok": True, "summary": result.get("summary", "")}


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


def _tool_event(node: str, tool_message: ToolMessage) -> dict[str, Any]:
    """把 ToolMessage 转换成统一事件。"""
    try:
        result = json.loads(str(tool_message.content))
    except json.JSONDecodeError:
        result = tool_message.content
    return {"type": "tool_result", "node": node, "name": tool_message.name, "result": result}


def _last_ai_text(messages: list[Any]) -> str:
    """取最后一条非 ToolMessage 的模型文本。"""
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            continue
        content = getattr(message, "content", "")
        if content:
            return str(content)
    return ""
