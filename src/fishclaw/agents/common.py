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

