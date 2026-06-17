"""Fishclaw LangGraph 节点和路由函数。"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from fishclaw.agents import PLANNER_PROMPT, build_planner_tools
from fishclaw.graph.state_io import _message_text
from fishclaw.memory import FishStore, build_memory, format_memory, json_safe, short_text
from fishclaw.model import create_model
from fishclaw.state import FishState


COMPRESS_PROMPT = """你是 Fishclaw 的 context_compressor。
请把旧消息、搜索笔记、代码摘要、handoff 和历史摘要压缩成一段可继续工作的中文上下文。
保留：用户目标、已完成工作、重要文件、来源链接、未完成事项、风险。
只输出压缩后的中文摘要。"""


def planner_node(state: FishState) -> dict[str, Any]:
    """planner 节点：让模型选择 SearchAgentTool 或 CodeAgentTool。"""
    writer = _writer()
    working: FishState = {**state}
    memory = build_memory(working)
    recent_messages = "\n\n".join(
        short_text(_message_text(message), 700)
        for message in working.get("messages", [])[-6:]
        if _message_text(message).strip()
    )
    model = create_model().bind_tools(build_planner_tools(working, writer))
    prompt = f"用户任务：{working.get('task', '')}\n\n上下文工程记忆：\n{format_memory(memory)}"
    if recent_messages:
        prompt += f"\n\n最近图消息和工具结果摘要：\n{recent_messages}"
    messages: list[Any] = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = model.invoke(messages)
    produced: list[Any] = [response]
    tool_calls = getattr(response, "tool_calls", None) or []
    writer({"type": "planner_round", "round": working.get("planner_rounds", 0) + 1, "tool_calls": len(tool_calls)})
    for call in tool_calls:
        tool_message = _execute_planner_tool(working, writer, call)
        produced.append(tool_message)
        messages.append(tool_message)
    done = not tool_calls
    final_answer = str(getattr(response, "content", "") or "").strip() if done else ""
    return {
        "messages": produced,
        "planner_rounds": working.get("planner_rounds", 0) + 1,
        "since_compression": working.get("since_compression", 0) + 1,
        "done": done,
        "final_answer": final_answer,
        "search_notes": working.get("search_notes", ""),
        "sources": working.get("sources", []),
        "code_summary": working.get("code_summary", ""),
        "handoffs": working.get("handoffs", []),
        "metadata": {"last_planner_response": final_answer or short_text(getattr(response, "content", ""), 800)},
    }


def context_gate_node(state: FishState) -> dict[str, Any]:
    """上下文门控：达到次数阈值后触发自动压缩。"""
    runtime = state["runtime"]
    should_compress = (
        not state.get("done", False)
        and state.get("since_compression", 0) >= runtime.compress_every
        and bool(state.get("messages"))
    )
    FishStore(runtime).save_state(state, status="gate")
    _writer()(
        {
            "type": "context_gate",
            "since_compression": state.get("since_compression", 0),
            "compress_every": runtime.compress_every,
            "should_compress": should_compress,
        }
    )
    return {"should_compress": should_compress}


def context_route(state: FishState) -> str:
    """决定继续 planner、进入压缩，还是结束。"""
    if state.get("done") or state.get("planner_rounds", 0) >= state["runtime"].max_planner_rounds:
        return "final"
    if state.get("should_compress"):
        return "context_compressor"
    return "planner"


def context_compressor_node(state: FishState) -> dict[str, Any]:
    """自动压缩上下文，并把摘要持久化到 HISTORY.md。"""
    memory = build_memory(state)
    message_text = "\n\n".join(short_text(_message_text(message), 800) for message in state.get("messages", []))
    payload = f"当前 memory:\n{format_memory(memory)}\n\n旧消息摘要素材：\n{message_text}"
    try:
        response = create_model(temperature=0.0).invoke([SystemMessage(content=COMPRESS_PROMPT), HumanMessage(content=payload)])
        summary = str(getattr(response, "content", "") or "").strip()
    except Exception as exc:
        summary = f"压缩模型不可用，使用兜底摘要。\n\n{short_text(json.dumps(json_safe(memory), ensure_ascii=False), 2400)}\n\n错误：{type(exc).__name__}: {exc}"
    if not summary:
        summary = short_text(json.dumps(json_safe(memory), ensure_ascii=False), 2400)
    FishStore(state["runtime"]).write_history(summary)
    _writer()({"type": "context_compressed", "summary": short_text(summary, 1000)})
    return {
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), AIMessage(content=summary)],
        "context_summary": summary,
        "history_summary": summary,
        "since_compression": 0,
        "should_compress": False,
    }


def final_node(state: FishState) -> dict[str, Any]:
    """输出最终摘要，并保存最终状态。"""
    answer = state.get("final_answer", "").strip()
    if not answer:
        answer = (
            "Fishclaw 已停止继续规划。\n\n"
            f"代码摘要：{state.get('code_summary', '') or '(无)'}\n\n"
            f"搜索摘要：{state.get('search_notes', '') or '(无)'}"
        )
    FishStore(state["runtime"]).save_state({**state, "final_answer": answer, "done": True}, status="finished")
    return {"final_answer": answer, "done": True}


def _execute_planner_tool(state: FishState, writer, call: dict[str, Any]) -> ToolMessage:
    """执行 planner 的 SearchAgentTool/CodeAgentTool 调用。"""
    tools = {tool.name: tool for tool in build_planner_tools(state, writer)}
    name = str(call.get("name", ""))
    args = call.get("args") or {}
    writer({"type": "tool_call", "node": "planner", "name": name, "args": args})
    tool = tools.get(name)
    if tool is None:
        result = {"ok": False, "error": f"unknown planner tool: {name}"}
    else:
        try:
            result = tool.invoke(args)
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    tool_message = ToolMessage(content=json.dumps(result, ensure_ascii=False, default=str), name=name, tool_call_id=call.get("id") or f"{name}-call")
    writer({"type": "tool_result", "node": "planner", "name": name, "result": result})
    return tool_message


def _writer():
    """获取 LangGraph custom event writer；非图环境下返回空函数。"""
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _: None
