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
            "planner 只能通过 SearchAgentTool、CodeAgentTool、ReviewAgentTool 和 TestAgentTool 委派工作。",
            "结构化 task_plan 和 active_task_id 是任务状态的唯一权威来源；压缩摘要和历史摘要只作为背景。",
            "SearchAgentTool 只收集资料和来源；任何创建文件任务都交给 CodeAgentTool。",
            "latest_research_assessment.status=incomplete 时，只有 proposed_tasks 尚未对应到 task_plan 的 pending/in_progress 任务，才追加新计划；已有则直接分发。",
            "blocked 任务最多只能拆分一次；blocked_split_depth>=1 的任务再次 blocked 后应交给人工或跳过，继续下一个 pending 任务。",
            "上下文达到阈值后由 context_compressor 压缩，并写入 .fishclaw/HISTORY.md。",
        ],
        "task": state.get("task", ""),
        "workspace": str(runtime.workspace),
        "planner_rounds": state.get("planner_rounds", 0),
        "task_plan": _compact_task_plan(state.get("task_plan", [])),
        "active_task_id": state.get("active_task_id", ""),
        "plan_summary": short_text(state.get("plan_summary", ""), 800),
        "context_summary": short_text(state.get("context_summary", ""), 1800),
        "history_summary": short_text(state.get("history_summary") or history, 1800),
        "search_notes": short_text(state.get("search_notes", ""), 1200),
        "sources": state.get("sources", [])[-8:],
        "research_artifacts": state.get("research_artifacts", [])[-6:],
        "latest_research_batch": _compact_research_batch(state.get("latest_research_batch", {})),
        "research_batches": [_compact_research_batch(item) for item in state.get("research_batches", [])[-4:]],
        "latest_research_assessment": _compact_research_assessment(state.get("latest_research_assessment", {})),
        "research_assessments": [
            _compact_research_assessment(item) for item in state.get("research_assessments", [])[-4:]
        ],
        "code_summary": short_text(state.get("code_summary", ""), 1200),
        "handoffs": state.get("handoffs", [])[-6:],
        "review_notes": short_text(state.get("review_notes", ""), 1200),
        "test_notes": short_text(state.get("test_notes", ""), 1200),
        "verification_status": state.get("verification_status", "not_started"),
        "verified": state.get("verified", False),
        "code_dirty": state.get("code_dirty", False),
    }


def build_agent_memory(
    state: FishState,
    *,
    instruction: str = "",
    active_task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a scoped memory view for child agents.

    Child agents should execute only the planner instruction/current task.
    They may still read scoped upstream context, but they should not receive the
    full task plan as actionable work.
    """
    runtime = state["runtime"]
    history = FishStore(runtime).read_history()
    current_task = _compact_task(active_task) or _current_task_from_state(state)
    target_agent = str(current_task.get("agent", "") or "")
    latest_assessment = _compact_research_assessment(state.get("latest_research_assessment", {}))
    research_assessments = [
        _compact_research_assessment(item) for item in state.get("research_assessments", [])[-4:]
    ]
    latest_batch = _compact_research_batch(state.get("latest_research_batch", {}))
    research_batches = [_compact_research_batch(item) for item in state.get("research_batches", [])[-4:]]
    if target_agent == "codeAgent":
        search_notes = ""
        sources: list[dict[str, Any]] = []
        research_artifacts: list[dict[str, Any]] = []
        research_context = _research_context(latest_assessment, research_assessments, latest_batch, research_batches)
    else:
        search_notes = short_text(state.get("search_notes", ""), 1400)
        sources = state.get("sources", [])[-8:]
        research_artifacts = state.get("research_artifacts", [])[-6:]
        research_context = ""
    return {
        "rules": [
            "All file and command operations must stay inside the current workspace.",
            "Only execute the planner_instruction/current_task delegated in this handoff.",
            "Do not start other task-plan items unless the planner delegates them explicitly.",
        ],
        "overall_goal": short_text(state.get("task", ""), 800),
        "workspace": str(runtime.workspace),
        "planner_rounds": state.get("planner_rounds", 0),
        "planner_instruction": short_text(instruction, 1200),
        "current_task": current_task,
        "plan_summary": short_text(state.get("plan_summary", ""), 800),
        "context_summary": short_text(state.get("context_summary", ""), 1800),
        "history_summary": short_text(state.get("history_summary") or history, 1800),
        "search_notes": search_notes,
        "sources": sources,
        "research_artifacts": research_artifacts,
        "research_context": research_context,
        "latest_research_batch": latest_batch,
        "research_batches": research_batches,
        "latest_research_assessment": latest_assessment,
        "research_assessments": research_assessments,
        "code_summary": short_text(state.get("code_summary", ""), 1400),
        "review_notes": short_text(state.get("review_notes", ""), 1200),
        "test_notes": short_text(state.get("test_notes", ""), 1200),
        "verification_status": state.get("verification_status", "not_started"),
        "verified": state.get("verified", False),
        "code_dirty": state.get("code_dirty", False),
    }


def _compact_task_plan(task_plan: Any) -> list[dict[str, Any]]:
    """Trim task-plan items for prompt memory."""
    if not isinstance(task_plan, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in task_plan[:20]:
        if not isinstance(item, dict):
            continue
        compact_item = {
            "id": item.get("id", ""),
            "title": short_text(item.get("title", ""), 160),
            "status": item.get("status", "pending"),
            "agent": item.get("agent", ""),
            "instruction": short_text(item.get("instruction", ""), 360),
            "result": short_text(item.get("result", ""), 240),
        }
        _copy_compact_task_metadata(item, compact_item)
        compact.append(compact_item)
    return compact


def _copy_compact_task_metadata(source: dict[str, Any], target: dict[str, Any]) -> None:
    """Copy task-control metadata into prompt memory."""
    for key in ("parent_task_id", "blocked_split_depth", "blocked_split_count", "manual_required", "skip_reason"):
        value = source.get(key)
        if value not in (None, "", False):
            target[key] = value

def _current_task_from_state(state: FishState) -> dict[str, Any]:
    """Return the active task only, without exposing the whole plan."""
    task_id = str(state.get("active_task_id", "") or "").strip()
    task_plan = state.get("task_plan", [])
    if not task_id or not isinstance(task_plan, list):
        return {}
    for item in task_plan:
        if isinstance(item, dict) and str(item.get("id", "")) == task_id:
            return _compact_task(item)
    return {}


def _compact_research_assessment(item: Any) -> dict[str, Any]:
    """Trim a research assessment record for prompt memory."""
    if not isinstance(item, dict):
        return {}
    return {
        "task_id": item.get("task_id", ""),
        "status": item.get("status", ""),
        "confidence": item.get("confidence", 0.0),
        "summary": short_text(item.get("summary", ""), 360),
        "answered": _short_list(item.get("answered", []), 5, 160),
        "open_questions": _short_list(item.get("open_questions", []), 5, 160),
        "evidence_gaps": _short_list(item.get("evidence_gaps", []), 5, 160),
        "next_queries": _short_list(item.get("next_queries", []), 6, 160),
        "proposed_tasks": _compact_proposed_tasks(item.get("proposed_tasks", [])),
        "stop_reason": item.get("stop_reason", ""),
        "artifact_paths": _short_list(item.get("artifact_paths", []), 6, 160),
    }


def _compact_research_batch(item: Any) -> dict[str, Any]:
    """Trim one SearchAgent batch for prompt memory."""
    if not isinstance(item, dict):
        return {}
    return {
        "batch_id": item.get("batch_id", ""),
        "task_id": item.get("task_id", ""),
        "status": item.get("status", ""),
        "confidence": item.get("confidence", 0.0),
        "summary": short_text(item.get("summary", ""), 500),
        "answered": _short_list(item.get("answered", []), 8, 180),
        "entities": _compact_batch_entities(item.get("entities", [])),
        "open_questions": _short_list(item.get("open_questions", []), 6, 180),
        "evidence_gaps": _short_list(item.get("evidence_gaps", []), 6, 180),
        "next_queries": _short_list(item.get("next_queries", []), 6, 160),
        "proposed_tasks": _compact_proposed_tasks(item.get("proposed_tasks", [])),
        "source_refs": _compact_batch_source_refs(item.get("source_refs", [])),
        "search_stats": item.get("search_stats", {}),
    }


def _compact_proposed_tasks(value: Any) -> list[dict[str, Any]]:
    """Trim ResearchEvaluator task proposals for prompt memory."""
    if not isinstance(value, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in value[:6]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "title": short_text(item.get("title", ""), 140),
                "agent": item.get("agent", ""),
                "instruction": short_text(item.get("instruction", ""), 320),
                "reason": short_text(item.get("reason", ""), 180),
            }
        )
    return compact

def _compact_batch_entities(value: Any) -> list[dict[str, Any]]:
    """Trim batch entities for downstream agents."""
    if not isinstance(value, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in value[:12]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "name": short_text(item.get("name", ""), 120),
                "type": short_text(item.get("type", ""), 60),
                "status": item.get("status", ""),
                "facts": _short_list(item.get("facts", []), 5, 160),
                "source_urls": _short_list(item.get("source_urls", []), 5, 180),
            }
        )
    return compact


def _compact_batch_source_refs(value: Any) -> list[dict[str, Any]]:
    """Trim batch source references for prompt memory."""
    if not isinstance(value, list):
        return []
    refs: list[dict[str, Any]] = []
    for item in value[:8]:
        if not isinstance(item, dict):
            continue
        refs.append(
            {
                "title": short_text(item.get("title", ""), 120),
                "url": short_text(item.get("url", ""), 180),
                "content_preview": short_text(item.get("content_preview", ""), 180),
            }
        )
    return refs


def _research_context(
    latest: dict[str, Any],
    assessments: list[dict[str, Any]],
    latest_batch: dict[str, Any] | None = None,
    batches: list[dict[str, Any]] | None = None,
) -> str:
    """Build compact research context for CodeAgent without raw sources."""
    if latest_batch:
        pieces = [
            f"batch_id={latest_batch.get('batch_id', '')}",
            f"status={latest_batch.get('status', '')}",
            f"confidence={latest_batch.get('confidence', 0.0)}",
            f"summary={latest_batch.get('summary', '')}",
        ]
        answered = latest_batch.get("answered", [])
        entities = latest_batch.get("entities", [])
        open_questions = latest_batch.get("open_questions", [])
        evidence_gaps = latest_batch.get("evidence_gaps", [])
        proposed_tasks = latest_batch.get("proposed_tasks", [])
        if answered:
            pieces.append(f"answered={answered}")
        if entities:
            pieces.append(f"entities={entities}")
        if open_questions:
            pieces.append(f"open_questions={open_questions}")
        if evidence_gaps:
            pieces.append(f"evidence_gaps={evidence_gaps}")
        if proposed_tasks:
            pieces.append(f"proposed_tasks={proposed_tasks}")
        return short_text("; ".join(str(piece) for piece in pieces if piece), 1400)
    if latest:
        pieces = [
            f"status={latest.get('status', '')}",
            f"confidence={latest.get('confidence', 0.0)}",
            f"summary={latest.get('summary', '')}",
        ]
        open_questions = latest.get("open_questions", [])
        evidence_gaps = latest.get("evidence_gaps", [])
        next_queries = latest.get("next_queries", [])
        proposed_tasks = latest.get("proposed_tasks", [])
        if open_questions:
            pieces.append(f"open_questions={open_questions}")
        if evidence_gaps:
            pieces.append(f"evidence_gaps={evidence_gaps}")
        if next_queries:
            pieces.append(f"next_queries={next_queries}")
        if proposed_tasks:
            pieces.append(f"proposed_tasks={proposed_tasks}")
        return short_text("; ".join(str(piece) for piece in pieces if piece), 1200)
    if batches:
        return short_text("; ".join(item.get("summary", "") for item in batches if item.get("summary")), 1200)
    if assessments:
        return short_text("; ".join(item.get("summary", "") for item in assessments if item.get("summary")), 1200)
    return ""


def _short_list(value: Any, limit: int, item_limit: int) -> list[str]:
    """Return a short text list for prompt memory."""
    if not isinstance(value, list):
        return []
    return [short_text(item, item_limit) for item in value[:limit]]


def _compact_task(item: Any) -> dict[str, Any]:
    """Trim one task item for child-agent prompt memory."""
    if not isinstance(item, dict):
        return {}
    compact = {
        "id": item.get("id", ""),
        "title": short_text(item.get("title", ""), 160),
        "status": item.get("status", "pending"),
        "agent": item.get("agent", ""),
        "instruction": short_text(item.get("instruction", ""), 600),
        "result": short_text(item.get("result", ""), 240),
    }
    _copy_compact_task_metadata(item, compact)
    return compact


def format_memory(memory: dict[str, Any]) -> str:
    """把 layered memory 渲染成稳定 JSON prompt。"""
    return json.dumps(memory, ensure_ascii=False, indent=2, default=str)


def merge_sources(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 URL 去重合并来源列表。"""
    merged: list[dict[str, Any]] = []
    for group in groups:
        merged = merge_source_lists(merged, group)
    return merged
