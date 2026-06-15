"""Fishclaw 的简化 LangGraph 工作流。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from fishclaw.agents import PLANNER_PROMPT, build_planner_tools
from fishclaw.memory import FishStore, build_memory, format_memory, json_safe, short_text
from fishclaw.model import create_model
from fishclaw.state import FishRuntime, FishState, new_workspace


COMPRESS_PROMPT = """你是 Fishclaw 的 context_compressor。
请把旧消息、搜索笔记、代码摘要、handoff 和历史摘要压缩成一段可继续工作的中文上下文。
保留：用户目标、已完成工作、重要文件、来源链接、未完成事项、风险。
只输出压缩后的中文摘要。"""


def create_runtime(
    workspace: Path | None = None,
    *,
    max_planner_rounds: int = 8,
    compress_every: int | None = None,
) -> FishRuntime:
    """创建 Fishclaw runtime。"""
    runtime = FishRuntime(workspace=workspace or new_workspace(), max_planner_rounds=max_planner_rounds)
    if compress_every is not None:
        runtime.compress_every = max(1, int(compress_every))
    return runtime


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


def stream_fishclaw_events(
    task: str,
    *,
    workspace: Path | None = None,
    max_planner_rounds: int = 8,
    compress_every: int | None = None,
) -> Iterator[dict[str, Any]]:
    """从用户任务开始，流式运行 Fishclaw 工作流。"""
    runtime = create_runtime(workspace, max_planner_rounds=max_planner_rounds, compress_every=compress_every)
    store = FishStore(runtime)
    inputs: FishState = {
        "task": task,
        "runtime": runtime,
        "messages": [],
        "planner_rounds": 0,
        "since_compression": 0,
        "context_summary": "",
        "history_summary": store.read_history(),
        "sources": [],
        "handoffs": [],
    }
    store.append_event({"type": "run_start", "task": task, "workspace": str(runtime.workspace)})
    store.save_state(inputs, status="started")
    yield {"type": "workspace", "path": str(runtime.workspace)}

    current_state: FishState = dict(inputs)
    try:
        for mode, event in build_workflow().stream(inputs, stream_mode=["updates", "custom"]):
            if mode == "custom":
                store.append_event(event if isinstance(event, dict) else {"type": "custom", "payload": event})
                yield {"type": "custom_event", "event": event}
            else:
                _merge_update(current_state, event)
                store.save_state(current_state, status="running")
                yield {"type": "graph_event", "event": event}
    finally:
        store.save_state(current_state, status="finished" if current_state.get("done") else "stopped")
        store.append_event({"type": "run_end", "done": current_state.get("done", False)})


def planner_node(state: FishState) -> dict[str, Any]:
    """planner 节点：让模型选择 SearchAgentTool 或 CodeAgentTool。"""
    writer = _writer()
    working: FishState = {**state}
    memory = build_memory(working)
    model = create_model().bind_tools(build_planner_tools(working, writer))
    messages: list[Any] = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"用户任务：{working.get('task', '')}\n\n上下文工程记忆：\n{format_memory(memory)}"),
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


def _execute_planner_tool(state: FishState, writer, call: dict[str, Any]):
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
    from langchain_core.messages import ToolMessage

    tool_message = ToolMessage(content=json.dumps(result, ensure_ascii=False, default=str), name=name, tool_call_id=call.get("id") or f"{name}-call")
    writer({"type": "tool_result", "node": "planner", "name": name, "result": result})
    return tool_message


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


def _message_text(message: Any) -> str:
    """把消息对象转换成文本。"""
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)


def _writer():
    """获取 LangGraph custom event writer；非图环境下返回空函数。"""
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _: None
