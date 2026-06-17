"""Fishclaw 的运行时配置和 LangGraph 状态契约。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


def new_workspace(root: Path | None = None) -> Path:
    """创建一个新的 Fishclaw workspace 路径。"""
    base = root or Path.cwd()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return base / ".fishclaw" / "workspaces" / f"workspace-{stamp}-{uuid4().hex[:6]}"


def _env_int(name: str, default: int) -> int:
    """读取正整数环境变量，非法时回退到默认值。"""
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


@dataclass
class FishRuntime:
    """一次 Fishclaw 运行共享的配置和工具安全状态。"""

    workspace: Path
    max_planner_rounds: int = 8
    max_agent_loops: int = 6
    compress_every: int = field(default_factory=lambda: _env_int("FISHCLAW_COMPRESS_EVERY", 4))
    bash_timeout_seconds: int = field(default_factory=lambda: _env_int("FISHCLAW_BASH_TIMEOUT_SECONDS", 120))
    max_output_chars: int = field(default_factory=lambda: _env_int("FISHCLAW_MAX_OUTPUT_CHARS", 6000))
    read_snapshots: dict[Path, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """保证 workspace 已存在，并把路径规范成绝对路径。"""
        self.workspace = self.workspace.expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def assert_workspace_path(self, path: Path) -> Path:
        """拒绝逃出 workspace 的文件访问。"""
        resolved = path.expanduser().resolve()
        workspace = self.workspace.resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise ValueError(f"path must stay inside workspace: {workspace}")
        return resolved

    def record_read(self, path: Path) -> None:
        """记录文件读取快照，后续写入时用于乐观锁校验。"""
        resolved = self.assert_workspace_path(path)
        if resolved.exists():
            self.read_snapshots[resolved] = resolved.stat().st_mtime_ns


class FishState(TypedDict, total=False):
    """Fishclaw 图中所有节点共享的状态。"""

    task: str
    runtime: FishRuntime
    messages: Annotated[list[BaseMessage], add_messages]
    planner_rounds: int
    since_compression: int
    should_compress: bool
    done: bool
    final_answer: str
    context_summary: str
    history_summary: str
    search_notes: str
    sources: list[dict[str, Any]]
    code_summary: str
    handoffs: list[dict[str, Any]]
    metadata: dict[str, Any]
