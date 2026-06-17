"""Fishclaw 的 Context Engineering 和信息持久化。"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from langchain_core.messages import BaseMessage, message_to_dict

from fishclaw.state import FishRuntime, FishState
from fishclaw.state.reducers import merge_sources as merge_source_lists


MAX_MEMORY_TEXT = 1600


def utc_now() -> str:
    """返回 UTC ISO 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


def short_text(value: Any, limit: int = MAX_MEMORY_TEXT) -> str:
    """把任意值压成适合 prompt 的短文本。"""
    text = value if isinstance(value, str) else json.dumps(json_safe(value), ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def json_safe(value: Any) -> Any:
    """把状态、消息、Path、dataclass 转为可 JSON 化结构。"""
    if isinstance(value, BaseMessage):
        return message_to_dict(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class FishStore:
    """workspace 内的轻量持久化仓库。"""

    def __init__(self, runtime: FishRuntime) -> None:
        self.runtime = runtime
        self.root = runtime.workspace / ".fishclaw"
        self.state_file = self.root / "state.json"
        self.events_file = self.root / "events.jsonl"
        self.history_file = self.root / "HISTORY.md"
        self.root.mkdir(parents=True, exist_ok=True)

    def append_event(self, event: dict[str, Any]) -> None:
        """追加 JSONL 事件，便于运行后排查。"""
        line = {"time": utc_now(), "event": json_safe(event)}
        with self.events_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")

    def save_state(self, state: FishState, *, status: str = "running") -> None:
        """保存当前图状态摘要，不保存 runtime 对象本身。"""
        payload = {key: value for key, value in state.items() if key != "runtime"}
        payload["status"] = status
        payload["updated_at"] = utc_now()
        self._write_json(self.state_file, json_safe(payload))

    def read_history(self) -> str:
        """读取持久上下文摘要。"""
        if not self.history_file.exists():
            return ""
        return short_text(self.history_file.read_text(encoding="utf-8", errors="replace"), 2400)

    def write_history(self, summary: str) -> None:
        """写入最新上下文摘要。"""
        content = f"# Fishclaw History\n\n_Updated: {utc_now()}_\n\n{summary.strip()}\n"
        self.history_file.write_text(content, encoding="utf-8")

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        """用临时文件替换方式写 JSON，减少半写入风险。"""
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)

    def load_state(self) -> dict[str, Any]:
        """加载并返回当前状态。"""
        if not self.state_file.exists():
            raise FileNotFoundError(f"State file does not exist: {self.state_file}")
        return json.loads(self.state_file.read_text(encoding="utf-8"))

    def iter_events(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        """读取 events.jsonl，limit 表示只取最后 N 条。"""
        if not self.events_file.exists():
            return
        lines = self.events_file.read_text(encoding="utf-8", errors="replace").splitlines()
        if limit is not None:
            lines = lines[-max(1, int(limit)) :]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_memory(state: FishState) -> dict[str, Any]:
    """组装 planner/search/code/compressor 共用的 layered memory。"""
    runtime = state["runtime"]
    history = FishStore(runtime).read_history()
    return {
        "rules": [
            "所有文件和命令都必须限制在当前 workspace 内。",
            "planner 只能通过 SearchAgentTool 和 CodeAgentTool 委派工作。",
            "上下文达到阈值后由 context_compressor 压缩，并写入 .fishclaw/HISTORY.md。",
        ],
        "task": state.get("task", ""),
        "workspace": str(runtime.workspace),
        "planner_rounds": state.get("planner_rounds", 0),
        "context_summary": short_text(state.get("context_summary", ""), 1800),
        "history_summary": short_text(state.get("history_summary") or history, 1800),
        "search_notes": short_text(state.get("search_notes", ""), 1200),
        "sources": state.get("sources", [])[-8:],
        "code_summary": short_text(state.get("code_summary", ""), 1200),
        "handoffs": state.get("handoffs", [])[-6:],
    }


def format_memory(memory: dict[str, Any]) -> str:
    """把 layered memory 渲染成稳定 JSON prompt。"""
    return json.dumps(memory, ensure_ascii=False, indent=2, default=str)


def merge_sources(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 URL 去重合并来源列表。"""
    merged: list[dict[str, Any]] = []
    for group in groups:
        merged = merge_source_lists(merged, group)
    return merged
