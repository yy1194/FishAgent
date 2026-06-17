"""Fishclaw LangGraph 构建入口。"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from fishclaw.graph.nodes import context_compressor_node, context_gate_node, context_route, final_node, planner_node
from fishclaw.state import FishState


def build_workflow():
    """构建：planner -> context_gate -> compressor/final/planner。"""
    graph = StateGraph(FishState)
    graph.add_node("planner", planner_node)
    graph.add_node("context_gate", context_gate_node)
    graph.add_node("context_compressor", context_compressor_node)
    graph.add_node("final", final_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "context_gate")
    graph.add_conditional_edges(
        "context_gate",
        context_route,
        {"planner": "planner", "context_compressor": "context_compressor", "final": "final"},
    )
    graph.add_edge("context_compressor", "planner")
    graph.add_edge("final", END)
    return graph.compile()

