"""TestAgent：负责运行非交互式验证命令并分析结果。"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from fishclaw.agents.common import Writer, compact_text, last_ai_text, summarize_tool_events, tool_event
from fishclaw.agents.prompts import TEST_PROMPT
from fishclaw.memory import build_agent_memory, format_memory
from fishclaw.model import create_model
from fishclaw.state import FishRuntime, FishState
from fishclaw.tools.harness import file_read, grep, list_files, run_bash, tool_result_json


def build_test_tools(runtime: FishRuntime) -> list[StructuredTool]:
    """构建 TestAgent 可用的只读和命令验证工具。"""
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
        StructuredTool.from_function(
            name="BashTool",
            func=lambda command, timeout_seconds=None: run_bash(runtime, command, timeout_seconds),
            description="在 workspace 内运行非交互验证命令。参数：command, timeout_seconds。",
        ),
    ]


def run_test_agent(
    state: FishState,
    instruction: str,
    *,
    writer: Writer | None = None,
    active_task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """运行 TestAgent，并返回测试摘要。"""
    writer = writer or (lambda _: None)
    runtime = state["runtime"]
    model = create_model().bind_tools(build_test_tools(runtime))
    messages: list[Any] = [
        SystemMessage(content=TEST_PROMPT),
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
    commands: list[str] = []
    saw_failed_command = False
    saw_successful_command = False
    loop_limited = False

    for _ in range(runtime.max_agent_loops):
        response = model.invoke(messages)
        produced.append(response)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for call in tool_calls:
            writer({"type": "tool_call", "node": "testAgent", "name": call.get("name"), "args": call.get("args", {})})
            tool_message = _execute_test_tool(state, call)
            event = tool_event("testAgent", tool_message)
            tool_events.append(event)
            writer(event)
            produced.append(tool_message)
            messages.append(tool_message)

            result = event.get("result")
            if call.get("name") == "BashTool" and isinstance(result, dict):
                command = str(result.get("command", ""))
                if command:
                    commands.append(command)
                if result.get("ok") is True:
                    saw_successful_command = True
                else:
                    saw_failed_command = True
    else:
        loop_limited = True
        produced.append(AIMessage(content="TestAgent 达到工具循环上限，已停止继续调用工具。"))

    passed = saw_successful_command and not saw_failed_command
    summary = _test_result_summary(
        last_ai_text(produced),
        passed,
        commands,
        saw_successful_command,
        saw_failed_command,
        tool_events,
        loop_limited,
    )
    return {
        "ok": True,
        "summary": summary,
        "passed": passed,
        "commands": commands,
        "messages": produced,
        "tool_events": tool_events,
    }


def _execute_test_tool(state: FishState, call: dict[str, Any]) -> ToolMessage:
    """执行 TestAgent 的验证工具。"""
    tools = {tool.name: tool for tool in build_test_tools(state["runtime"])}
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


def _test_result_summary(
    model_summary: str,
    passed: bool,
    commands: list[str],
    saw_successful_command: bool,
    saw_failed_command: bool,
    tool_events: list[dict[str, Any]],
    loop_limited: bool,
) -> str:
    """Build a deterministic verification summary."""
    lines: list[str] = []
    if commands:
        if passed:
            status = "验证通过"
        elif saw_failed_command:
            status = "验证未通过"
        elif saw_successful_command:
            status = "验证结果不确定"
        else:
            status = "未获得有效验证结果"
        command_text = "；".join(compact_text(command, 180) for command in commands[:6])
        if len(commands) > 6:
            command_text += f"；另有 {len(commands) - 6} 条命令"
        lines.append(f"{status}。运行命令：{command_text}")
    else:
        lines.append("未运行 BashTool 验证命令；无法确认代码是否通过测试。")

    model_summary = model_summary.strip()
    if model_summary:
        lines.append(f"模型总结：{compact_text(model_summary, 800)}")
    if tool_events and (not model_summary or saw_failed_command or loop_limited):
        lines.append(summarize_tool_events(tool_events))
    if loop_limited and "循环上限" not in model_summary and "loop limit" not in model_summary.lower():
        lines.append("TestAgent 已达到工具循环上限并停止。")
    return "\n".join(line for line in lines if line).strip()
