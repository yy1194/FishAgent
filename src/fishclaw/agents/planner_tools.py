"""Planner 可调用的子 agent 包装工具。"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from fishclaw.agents.code_agent import run_code_agent
from fishclaw.agents.common import Writer
from fishclaw.agents.search_agent import run_search_agent
from fishclaw.memory import merge_sources, short_text
from fishclaw.state import FishState


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

