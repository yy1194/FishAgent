"""LangGraph state 合并、恢复和消息文本化辅助函数。"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import messages_from_dict
from langgraph.graph import add_messages

from fishclaw.memory import FishStore
from fishclaw.state import FishRuntime, FishState


def _merge_update(state: FishState, event: Any) -> None:
    """把 LangGraph update 合并到本地状态，供持久化使用。"""
    if not isinstance(event, dict):
        return
    for update in event.values():
        if not isinstance(update, dict):
            continue
        for key, value in update.items():
            if key == "messages":
                state["messages"] = list(add_messages(state.get("messages", []), value))
            else:
                state[key] = value


def _restore_saved_state(payload: dict[str, Any], runtime: FishRuntime) -> FishState:
    """把 state.json 反序列化为可继续运行的 FishState。"""
    state: FishState = dict(payload)
    state.pop("status", None)
    state.pop("updated_at", None)
    raw_messages = state.get("messages", [])
    if raw_messages and isinstance(raw_messages[0], dict):
        state["messages"] = messages_from_dict(raw_messages)
    state["runtime"] = runtime
    state.setdefault("messages", [])
    state.setdefault("sources", [])
    state.setdefault("handoffs", [])
    state.setdefault("context_summary", "")
    state.setdefault("history_summary", FishStore(runtime).read_history())
    return state


def _message_text(message: Any) -> str:
    """把消息对象转换成文本。"""
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)

