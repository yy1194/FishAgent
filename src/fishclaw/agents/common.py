"""Agents 之间共享的小工具。"""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.messages import ToolMessage


Writer = Callable[[dict[str, Any]], None]


def tool_event(node: str, tool_message: ToolMessage) -> dict[str, Any]:
    """把 ToolMessage 转换成统一事件。"""
    try:
        result = json.loads(str(tool_message.content))
    except json.JSONDecodeError:
        result = tool_message.content
    return {"type": "tool_result", "node": node, "name": tool_message.name, "result": result}


def last_ai_text(messages: list[Any]) -> str:
    """取最后一条非 ToolMessage 的模型文本。"""
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            continue
        content = getattr(message, "content", "")
        if content:
            return str(content)
    return ""


def compact_text(value: Any, limit: int = 300) -> str:
    """Return a single compact text fragment for summaries."""
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def summarize_tool_events(tool_events: list[dict[str, Any]], *, limit: int = 8) -> str:
    """Summarize tool execution records without depending on model prose."""
    if not tool_events:
        return "未记录工具调用。"

    ok_count = 0
    failed_count = 0
    details: list[str] = []
    for event in tool_events[:limit]:
        name = str(event.get("name") or "unknown")
        result = event.get("result")
        status = "unknown"
        detail = ""
        if isinstance(result, dict):
            ok = result.get("ok")
            if ok is True:
                ok_count += 1
                status = "ok"
            elif ok is False:
                failed_count += 1
                status = "failed"
            if result.get("command"):
                detail = f"command={compact_text(result.get('command'), 120)}"
            elif result.get("path"):
                detail = f"path={compact_text(result.get('path'), 120)}"
            elif result.get("file_path"):
                detail = f"file={compact_text(result.get('file_path'), 120)}"
            elif result.get("query"):
                detail = f"query={compact_text(result.get('query'), 120)}"
            elif result.get("error"):
                detail = f"error={compact_text(result.get('error'), 160)}"
        suffix = f": {detail}" if detail else ""
        details.append(f"{name}({status}{suffix})")
    if len(tool_events) > limit:
        details.append(f"... 另有 {len(tool_events) - limit} 次")
    return f"工具调用 {len(tool_events)} 次，成功 {ok_count} 次，失败 {failed_count} 次：" + "；".join(details)


def build_agent_summary(
    agent_name: str,
    model_summary: str,
    tool_events: list[dict[str, Any]],
    *,
    loop_limited: bool = False,
) -> str:
    """Build a stable agent summary with tool-event fallback."""
    lines: list[str] = []
    model_summary = model_summary.strip()
    if model_summary:
        lines.append(model_summary)
    if tool_events and (not model_summary or loop_limited):
        lines.append(summarize_tool_events(tool_events))
    if loop_limited and "循环上限" not in model_summary and "loop limit" not in model_summary.lower():
        lines.append(f"{agent_name} 已达到工具循环上限并停止。")
    if not lines:
        lines.append(f"{agent_name} 已结束，但模型未返回明确总结。")
    return "\n".join(lines).strip()
