"""Fishclaw 工作流运行和恢复的流式入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from fishclaw.graph.builder import build_workflow
from fishclaw.graph.state_io import _merge_update, _restore_saved_state
from fishclaw.memory import FishStore
from fishclaw.state import FishRuntime, FishState, new_workspace


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
    yield from _stream_graph_events(
        inputs=inputs,
        store=store,
        start_event={"type": "workspace", "path": str(runtime.workspace)},
        start_log={"type": "run_start", "task": task, "workspace": str(runtime.workspace)},
        started_status="started",
    )


def resume_fishclaw_events(
    workspace: Path,
    *,
    max_planner_rounds: int = 8,
    compress_every: int | None = None,
) -> Iterator[dict[str, Any]]:
    """从已有 workspace 的 state.json 恢复运行。"""
    runtime = create_runtime(workspace, max_planner_rounds=max_planner_rounds, compress_every=compress_every)
    store = FishStore(runtime)
    payload = store.load_state()
    if payload.get("done") or payload.get("status") == "finished":
        yield {"type": "resume_skipped", "reason": "state is already finished", "path": str(runtime.workspace)}
        return
    inputs = _restore_saved_state(payload, runtime)
    yield from _stream_graph_events(
        inputs,
        store,
        start_event={"type": "workspace", "path": str(runtime.workspace), "resume": True},
        start_log={"type": "run_resume", "task": inputs.get("task", ""), "workspace": str(runtime.workspace)},
        started_status="resumed",
    )


def _stream_graph_events(
    inputs: FishState,
    store: FishStore,
    *,
    start_event: dict[str, Any],
    start_log: dict[str, Any],
    started_status: str,
) -> Iterator[dict[str, Any]]:
    """流式处理图事件。"""
    store.append_event(start_log)
    store.save_state(inputs, status=started_status)
    yield start_event

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

