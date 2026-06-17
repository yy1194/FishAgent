"""Planner 可调用的子 agent 包装工具。"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from fishclaw.agents.code_agent import run_code_agent
from fishclaw.agents.common import Writer
from fishclaw.agents.search_agent import run_search_agent
from fishclaw.memory import short_text
from fishclaw.state import FishState, apply_state_update


def build_planner_tools(state: FishState, writer: Writer, state_updates: FishState | None = None) -> list[StructuredTool]:
    """构建 planner 唯一可见的两个 agent 包装工具。"""
    return [
        StructuredTool.from_function(
            name="SearchAgentTool",
            func=lambda instruction: _call_search_agent(state, writer, instruction, state_updates),
            description="委派搜索研究任务。参数：instruction。",
        ),
        StructuredTool.from_function(
            name="CodeAgentTool",
            func=lambda instruction: _call_code_agent(state, writer, instruction, state_updates),
            description="委派 workspace 代码/文件/命令任务。参数：instruction。",
        ),
    ]


def _call_search_agent(
    state: FishState,
    writer: Writer,
    instruction: str,
    state_updates: FishState | None = None,
) -> dict[str, Any]:
    """执行 SearchAgentTool，并把结果合并到 planner 工作状态。"""
    writer({"type": "handoff", "from": "planner", "to": "searchAgent", "instruction": instruction})
    result = run_search_agent(state, instruction, writer=writer)
    delta: FishState = {
        "search_notes": result.get("summary", ""),
        "sources": result.get("sources", []) or [],
        "handoffs": [
            {"to": "searchAgent", "instruction": instruction, "summary": short_text(result.get("summary", ""), 600)}
        ],
    }
    _record_state_delta(state, state_updates, delta)
    writer({"type": "handoff_result", "from": "searchAgent", "to": "planner", "summary": result.get("summary", "")})
    return {"ok": True, "summary": result.get("summary", ""), "sources": state.get("sources", [])}


def _call_code_agent(
    state: FishState,
    writer: Writer,
    instruction: str,
    state_updates: FishState | None = None,
) -> dict[str, Any]:
    """执行 CodeAgentTool，并把实现摘要合并到 planner 工作状态。"""
    writer({"type": "handoff", "from": "planner", "to": "codeAgent", "instruction": instruction})
    result = run_code_agent(state, instruction, writer=writer)
    delta: FishState = {
        "code_summary": result.get("summary", ""),
        "handoffs": [
            {"to": "codeAgent", "instruction": instruction, "summary": short_text(result.get("summary", ""), 600)}
        ],
    }
    _record_state_delta(state, state_updates, delta)
    writer({"type": "handoff_result", "from": "codeAgent", "to": "planner", "summary": result.get("summary", "")})
    return {"ok": True, "summary": result.get("summary", "")}


def _record_state_delta(state: FishState, state_updates: FishState | None, delta: FishState) -> None:
    """Apply a child-agent delta to the working state and optional return update."""
    apply_state_update(state, delta)
    if state_updates is not None:
        apply_state_update(state_updates, delta)
