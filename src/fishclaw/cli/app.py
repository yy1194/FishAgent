"""Fishclaw 的轻量命令行入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import json
from rich import box
from rich.console import Console
from rich.panel import Panel

from fishclaw.graph import stream_fishclaw_events, resume_fishclaw_events
from fishclaw.memory import FishStore
from fishclaw.state import FishRuntime



app = typer.Typer(help='Fishclaw: 简洁版 MultiAgent + Context/Harness Engineering。')
console = Console()


@app.callback()
def main() -> None:
    """Fishclaw 命令行入口。"""

@app.command("run")
def run_task(
    task: Annotated[str, typer.Argument(help="用户任务，从这里进入 planner 节点。")],
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w", help="Fishclaw workspace。")] = None,
    max_rounds: Annotated[int, typer.Option("--max-rounds", help="planner 最大轮数。")] = 8,
    compress_every: Annotated[int, typer.Option("--compress-every", help="每 N 轮自动压缩上下文。")] = 4,
) -> None:
    """运行一次 Fishclaw 工作流。"""
    for event in stream_fishclaw_events(
        task,
        workspace=workspace,
        max_planner_rounds=max_rounds,
        compress_every=compress_every,
    ):
        _print_event(event)

@app.command("inspect")
def inspect_workspace(workspace: Annotated[Path, typer.Argument(help="Fishclaw workspace 路径。")]) -> None:
    """查看 workspace 当前 state 摘要。"""
    store = _store_for_workspace(workspace)
    state = store.load_state()
    summary = {
        "status": state.get("status"),
        "task": state.get("task"),
        "planner_rounds": state.get("planner_rounds"),
        "since_compression": state.get("since_compression"),
        "done": state.get("done"),
        "messages_count": len(state.get("messages", [])),
        "sources_count": len(state.get("sources", [])),
        "handoffs_count": len(state.get("handoffs", [])),
        "updated_at": state.get("updated_at"),
        "final_answer": _compact_payload(state.get("final_answer", "")),
    }
    console.print(Panel(json.dumps(summary, ensure_ascii=False, indent=2), title="Fishclaw State"))

@app.command("events")
def show_events(
        workspace: Annotated[Path, typer.Argument(help="Fishclaw workspace 路径。")],
        limit: Annotated[int, typer.Option("--limit", "-n", help="只显示最后 N 条事件。")] = 30,
) -> None:
    """显示 workspace 中的事件。"""
    store = _store_for_workspace(workspace)
    for item in store.iter_events(limit=limit):
        console.print(_compact_payload(item, limit=200))

@app.command("resume")
def resume_workspace(
    workspace: Annotated[Path, typer.Argument(help="Fishclaw workspace 路径。")],
    max_rounds: Annotated[int, typer.Option("--max-rounds", help="planner 最大轮数。")] = 8,
    compress_every: Annotated[int, typer.Option("--compress-every", help="每 N 轮自动压缩上下文。")] = 4,
) -> None:
    """从已有 workspace 的 state.json 恢复运行。"""
    for event in resume_fishclaw_events(
        workspace,
        max_planner_rounds=max_rounds,
        compress_every=compress_every,
    ):
        _print_event(event)

def _print_event(event: dict) -> None:
    """把流式事件渲染成简洁 Rich 面板。"""
    event_type = event.get("type")
    if event_type == "workspace":
        console.print(Panel(str(event.get("path", "")), title="Workspace", border_style="blue", box=box.ROUNDED))
        return
    payload = event.get("event", event)
    if event_type == "custom_event" and isinstance(payload, dict):
        title = str(payload.get("type", "event"))
        body = _compact_payload(payload)
        style = "magenta" if title in {"tool_call", "handoff"} else "cyan"
        console.print(Panel(body, title=title, border_style=style, box=box.ROUNDED))
        return
    if event_type == "graph_event" and isinstance(payload, dict):
        for node, update in payload.items():
            console.print(Panel(_compact_payload(update), title=str(node), border_style="green" if node == "final" else "yellow"))
        return
    console.print(event)


def _compact_payload(value, limit: int = 1200) -> str:
    """裁剪事件文本，避免终端输出过长。"""
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."

def _store_for_workspace(workspace: Path) -> FishStore:
    if not (workspace / ".fishclaw").exists():
        console.print(f"不是有效的 Fishclaw workspace: {workspace}")
        raise typer.Exit(code=1)
    return FishStore(FishRuntime(workspace=workspace))