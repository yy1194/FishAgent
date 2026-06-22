"""Planner 可调用的子 agent 包装工具。"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.tools import StructuredTool

from fishclaw.agents.code_agent import run_code_agent
from fishclaw.agents.common import Writer
from fishclaw.agents.review_agent import run_review_agent
from fishclaw.agents.search_agent import run_search_agent
from fishclaw.agents.test_agent import run_test_agent
from fishclaw.memory import short_text
from fishclaw.state import FishState, apply_state_update


TASK_STATUSES = {"pending", "in_progress", "completed", "blocked"}
BLOCKED_SPLIT_LIMIT = 1
TASK_AGENTS = {"searchAgent", "codeAgent", "reviewAgent", "testAgent"}
TASK_AGENT_ALIASES = {
    "searchagent": "searchAgent",
    "searchagenttool": "searchAgent",
    "search": "searchAgent",
    "codeagent": "codeAgent",
    "codeagenttool": "codeAgent",
    "code": "codeAgent",
    "reviewagent": "reviewAgent",
    "reviewagenttool": "reviewAgent",
    "review": "reviewAgent",
    "testagent": "testAgent",
    "testagenttool": "testAgent",
    "test": "testAgent",
}
CODE_FILE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".php",
    ".rb",
    ".sh",
    ".ps1",
    ".bat",
    ".html",
    ".css",
}
DOCUMENT_FILE_EXTENSIONS = {".md", ".txt", ".rst", ".adoc"}
DOCUMENT_TASK_MARKERS = (
    "markdown",
    "document",
    "report",
    "notes",
    "research",
    "文档",
    "报告",
    "记录",
    "资料",
    "总结",
    "整理",
)


def build_planner_tools(state: FishState, writer: Writer, state_updates: FishState | None = None) -> list[StructuredTool]:
    """构建 planner 可见的任务清单和子 agent 包装工具。"""
    def write_task_plan(tasks: Any, plan_summary: str = "") -> dict[str, Any]:
        return _write_task_plan(state, writer, tasks, plan_summary, state_updates)

    def update_task_status(task_id: str, status: str, result: str = "") -> dict[str, Any]:
        return _update_task_status(state, writer, task_id, status, result, state_updates)

    def call_search_agent(instruction: str, task_id: str = "") -> dict[str, Any]:
        return _call_search_agent(state, writer, instruction, state_updates, task_id)

    def call_code_agent(instruction: str, task_id: str = "") -> dict[str, Any]:
        return _call_code_agent(state, writer, instruction, state_updates, task_id)

    def call_review_agent(instruction: str, task_id: str = "") -> dict[str, Any]:
        return _call_review_agent(state, writer, instruction, state_updates, task_id)

    def call_test_agent(instruction: str, task_id: str = "") -> dict[str, Any]:
        return _call_test_agent(state, writer, instruction, state_updates, task_id)

    return [
        StructuredTool.from_function(
            name="WriteTaskPlanTool",
            func=write_task_plan,
            description=(
                "仅在任务清单确实需要新增或更新时调用。参数：tasks, plan_summary。"
                "tasks 是任务对象列表，每项包含 id, title, status, agent, instruction。"
                "如果返回 changed=false，说明计划未变化或 blocked 拆分已达上限，应直接调用 next_task_id 对应的 AgentTool，不要重复调用本工具。"
            ),
        ),
        StructuredTool.from_function(
            name="UpdateTaskStatusTool",
            func=update_task_status,
            description="更新任务清单中某个任务的状态。参数：task_id, status, result。",
        ),
        StructuredTool.from_function(
            name="SearchAgentTool",
            func=call_search_agent,
            description="委派搜索研究任务。参数：instruction, task_id。",
        ),
        StructuredTool.from_function(
            name="CodeAgentTool",
            func=call_code_agent,
            description="委派 workspace 代码/文件/命令任务。参数：instruction, task_id。",
        ),
        StructuredTool.from_function(
            name="ReviewAgentTool",
            func=call_review_agent,
            description="委派代码审查任务。参数：instruction, task_id。",
        ),
        StructuredTool.from_function(
            name="TestAgentTool",
            func=call_test_agent,
            description="委派测试验证任务。参数：instruction, task_id。",
        ),
    ]


def _write_task_plan(
    state: FishState,
    writer: Writer,
    tasks: Any,
    plan_summary: str = "",
    state_updates: FishState | None = None,
) -> dict[str, Any]:
    """Write a structured task plan into state."""
    try:
        normalized = _normalize_task_plan(tasks)
        planned_items, policy_message = _apply_blocked_split_policy(state, normalized)
        items = _preserve_existing_task_statuses(state, planned_items)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    plan_summary = plan_summary or _task_plan_summary(items)
    if not items or not _task_plan_update_changes_state(state, items, plan_summary):
        next_task_id = _next_task_id(state)
        message = policy_message or "Task plan already contains these tasks; dispatch next_task_id instead of calling WriteTaskPlanTool again."
        writer(
            {
                "type": "task_plan_unchanged",
                "task_count": len(state.get("task_plan", []) or []),
                "next_task_id": next_task_id,
                "plan_summary": state.get("plan_summary", "") or plan_summary,
                "message": message,
            }
        )
        return {
            "ok": True,
            "changed": False,
            "task_count": len(state.get("task_plan", []) or []),
            "next_task_id": next_task_id,
            "plan_summary": state.get("plan_summary", "") or plan_summary,
            "message": message,
        }
    delta: FishState = {
        "task_plan": items,
        "plan_summary": plan_summary,
        "active_task_id": _active_task_id_after_plan_write(state, items),
    }
    _record_state_delta(state, state_updates, delta)
    writer({"type": "task_plan_updated", "task_count": len(items), "plan_summary": delta["plan_summary"]})
    return {
        "ok": True,
        "changed": True,
        "task_count": len(state.get("task_plan", []) or items),
        "next_task_id": _next_task_id(state),
        "plan_summary": delta["plan_summary"],
        "message": policy_message,
    }


def _update_task_status(
    state: FishState,
    writer: Writer,
    task_id: str,
    status: str,
    result: str = "",
    state_updates: FishState | None = None,
) -> dict[str, Any]:
    """Update one task-plan item status."""
    task_id = str(task_id or "").strip()
    status = _normalize_status(status)
    if not task_id:
        return {"ok": False, "error": "task_id must not be empty"}
    if status == "in_progress":
        return {
            "ok": False,
            "error": "in_progress is reserved for AgentTool handoff; call the matching AgentTool instead",
        }
    delta: FishState = {
        "task_plan": [{"id": task_id, "status": status, "result": short_text(result, 600)}],
        "active_task_id": "",
    }
    _record_state_delta(state, state_updates, delta)
    writer({"type": "task_status_updated", "task_id": task_id, "status": status})
    return {"ok": True, "task_id": task_id, "status": status}


def _call_search_agent(
    state: FishState,
    writer: Writer,
    instruction: str,
    state_updates: FishState | None = None,
    task_id: str = "",
) -> dict[str, Any]:
    """执行 SearchAgentTool，并把结果合并到 planner 工作状态。"""
    active_task = _activate_task(state, state_updates, "searchAgent", task_id, instruction)
    if not active_task and _planned_task_required(state, "searchAgent", task_id):
        return _blocked_handoff("searchAgent", task_id, instruction, writer)
    instruction = _instruction_for_task(instruction, active_task)
    writer({"type": "handoff", "from": "planner", "to": "searchAgent", "task_id": active_task.get("id", ""), "instruction": instruction})
    result = run_search_agent(state, instruction, writer=writer, active_task=active_task)
    assessment_record = _research_assessment_record(active_task, instruction, result)
    research_batch = result.get("research_batch") or _fallback_research_batch(
        state,
        active_task,
        instruction,
        result,
        assessment_record,
    )
    delta: FishState = {
        "search_notes": result.get("summary", ""),
        "sources": result.get("sources", []) or [],
        "research_artifacts": result.get("research_artifacts", []) or [],
        "research_batches": [research_batch],
        "latest_research_batch": research_batch,
        "research_assessments": [assessment_record],
        "latest_research_assessment": assessment_record,
        "handoffs": [
            {
                "to": "searchAgent",
                "task_id": active_task.get("id", ""),
                "instruction": instruction,
                "summary": short_text(result.get("summary", ""), 600),
            }
        ],
        **_task_finished_delta(active_task, result),
    }
    _record_state_delta(state, state_updates, delta)
    writer({"type": "handoff_result", "from": "searchAgent", "to": "planner", "summary": result.get("summary", "")})
    return {
        "ok": bool(result.get("ok", True)),
        "summary": result.get("summary", ""),
        "sources": state.get("sources", []),
        "research_artifacts": state.get("research_artifacts", []),
        "research_status": result.get("research_status", "complete"),
        "latest_research_batch": research_batch,
        "confidence": result.get("confidence", 0.0),
        "open_questions": result.get("open_questions", []),
        "evidence_gaps": result.get("evidence_gaps", []),
        "next_queries": result.get("next_queries", []),
        "proposed_tasks": result.get("proposed_tasks", []) or [],
        "entities": result.get("entities", []) or [],
        "stop_reason": result.get("stop_reason", ""),
        "latest_research_assessment": assessment_record,
    }


def _call_code_agent(
    state: FishState,
    writer: Writer,
    instruction: str,
    state_updates: FishState | None = None,
    task_id: str = "",
) -> dict[str, Any]:
    """执行 CodeAgentTool，并把实现摘要合并到 planner 工作状态。"""
    active_task = _activate_task(state, state_updates, "codeAgent", task_id, instruction)
    if not active_task and _planned_task_required(state, "codeAgent", task_id):
        return _blocked_handoff("codeAgent", task_id, instruction, writer)
    instruction = _instruction_for_task(instruction, active_task)
    writer({"type": "handoff", "from": "planner", "to": "codeAgent", "task_id": active_task.get("id", ""), "instruction": instruction})
    result = run_code_agent(state, instruction, writer=writer, active_task=active_task)
    requires_verification = _code_agent_requires_verification(instruction, result)
    delta: FishState = {
        "code_summary": result.get("summary", ""),
        "handoffs": [
            {
                "to": "codeAgent",
                "task_id": active_task.get("id", ""),
                "instruction": instruction,
                "summary": short_text(result.get("summary", ""), 600),
            }
        ],
        **_task_finished_delta(active_task, result),
    }
    if requires_verification:
        delta.update(
            {
                "code_dirty": True,
                "verified": False,
                "verification_status": "needs_verification",
            }
        )
    _record_state_delta(state, state_updates, delta)
    writer({"type": "handoff_result", "from": "codeAgent", "to": "planner", "summary": result.get("summary", "")})
    return {"ok": True, "summary": result.get("summary", "")}

def _call_review_agent(
    state: FishState,
    writer: Writer,
    instruction: str,
    state_updates: FishState | None = None,
    task_id: str = "",
) -> dict[str, Any]:
    """执行 ReviewAgentTool，并把审查摘要合并到 planner 工作状态。"""
    active_task = _activate_task(state, state_updates, "reviewAgent", task_id, instruction)
    if not active_task and _planned_task_required(state, "reviewAgent", task_id):
        return _blocked_handoff("reviewAgent", task_id, instruction, writer)
    instruction = _instruction_for_task(instruction, active_task)
    writer({"type": "handoff", "from": "planner", "to": "reviewAgent", "task_id": active_task.get("id", ""), "instruction": instruction})
    result = run_review_agent(state, instruction, writer=writer, active_task=active_task)
    passed = bool(result.get("passed", False))
    delta: FishState = {
        "review_notes": result.get("summary", ""),
        "handoffs": [
            {
                "to": "reviewAgent",
                "task_id": active_task.get("id", ""),
                "instruction": instruction,
                "summary": short_text(result.get("summary", ""), 600),
            }
        ],
        "verification_status": "review_passed" if passed else "review_failed",
        **_task_finished_delta(active_task, result, completed=passed),
    }
    _record_state_delta(state, state_updates, delta)
    writer({"type": "handoff_result", "from": "reviewAgent", "to": "planner", "summary": result.get("summary", "")})
    return {"ok": True, "summary": result.get("summary", ""), "passed": passed}

def _call_test_agent(
    state: FishState,
    writer: Writer,
    instruction: str,
    state_updates: FishState | None = None,
    task_id: str = "",
) -> dict[str, Any]:
    """执行 TestAgentTool，并把测试摘要合并到 planner 工作状态。"""
    active_task = _activate_task(state, state_updates, "testAgent", task_id, instruction)
    if not active_task and _planned_task_required(state, "testAgent", task_id):
        return _blocked_handoff("testAgent", task_id, instruction, writer)
    instruction = _instruction_for_task(instruction, active_task)
    writer({"type": "handoff", "from": "planner", "to": "testAgent", "task_id": active_task.get("id", ""), "instruction": instruction})
    result = run_test_agent(state, instruction, writer=writer, active_task=active_task)
    passed = bool(result.get("passed", False))
    delta: FishState = {
        "test_notes": result.get("summary", ""),
        "handoffs": [
            {
                "to": "testAgent",
                "task_id": active_task.get("id", ""),
                "instruction": instruction,
                "summary": short_text(result.get("summary", ""), 600),
            }
        ],
        "verified": passed,
        "code_dirty": not passed,
        "verification_status": "passed" if passed else "failed",
        **_task_finished_delta(active_task, result, completed=passed),
    }
    _record_state_delta(state, state_updates, delta)
    writer({"type": "handoff_result", "from": "testAgent", "to": "planner", "summary": result.get("summary", "")})
    return {
        "ok": True,
        "summary": result.get("summary", ""),
        "passed": passed,
        "commands": result.get("commands", []),
    }

def _record_state_delta(state: FishState, state_updates: FishState | None, delta: FishState) -> None:
    """Apply a child-agent delta to the working state and optional return update."""
    apply_state_update(state, delta)
    if state_updates is not None:
        apply_state_update(state_updates, delta)


def _code_agent_requires_verification(instruction: str, result: dict[str, Any]) -> bool:
    """Return whether a CodeAgent handoff changed code and needs review/test."""
    touched_paths = _code_agent_touched_paths(result)
    if not touched_paths:
        return True
    normalized = [path.replace("\\", "/").lower() for path in touched_paths]
    if any(_path_has_extension(path, CODE_FILE_EXTENSIONS) for path in normalized):
        return True
    document_like = any(marker in instruction.lower() for marker in DOCUMENT_TASK_MARKERS)
    if document_like and all(_is_document_material_path(path) for path in normalized):
        return False
    return True


def _code_agent_touched_paths(result: dict[str, Any]) -> list[str]:
    """Extract touched file paths from CodeAgent tool events."""
    paths: list[str] = []
    for event in result.get("tool_events", []) or []:
        if not isinstance(event, dict):
            continue
        payload = event.get("result")
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            continue
        for key in ("path", "file_path"):
            value = str(payload.get(key, "") or "").strip()
            if value:
                paths.append(value)
    return paths


def _is_document_material_path(path: str) -> bool:
    """Return whether a path is a non-code material/document output."""
    return path.startswith("research/") or _path_has_extension(path, DOCUMENT_FILE_EXTENSIONS)


def _path_has_extension(path: str, extensions: set[str]) -> bool:
    """Return whether a path ends with one of the provided extensions."""
    return any(path.endswith(extension) for extension in extensions)


def _research_assessment_record(
    active_task: dict[str, Any],
    instruction: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Build a durable research assessment record for downstream planning."""
    artifacts = result.get("research_artifacts", []) or []
    artifact_paths = [
        str(item.get("path", ""))
        for item in artifacts
        if isinstance(item, dict) and str(item.get("path", "")).strip()
    ]
    sources = result.get("sources", []) or []
    return {
        "task_id": str(active_task.get("id", "")),
        "instruction": short_text(instruction, 600),
        "status": result.get("research_status", "complete"),
        "ok": bool(result.get("ok", True)),
        "confidence": result.get("confidence", 0.0),
        "summary": short_text(result.get("assessment_summary") or result.get("summary", ""), 800),
        "answered": result.get("answered", []) or [],
        "open_questions": result.get("open_questions", []) or [],
        "evidence_gaps": result.get("evidence_gaps", []) or [],
        "next_queries": result.get("next_queries", []) or [],
        "proposed_tasks": result.get("proposed_tasks", []) or [],
        "entities": result.get("entities", []) or [],
        "stop_reason": result.get("stop_reason", ""),
        "artifact_paths": artifact_paths,
        "source_count": len(sources) if isinstance(sources, list) else 0,
    }


def _fallback_research_batch(
    state: FishState,
    active_task: dict[str, Any],
    instruction: str,
    result: dict[str, Any],
    assessment_record: dict[str, Any],
) -> dict[str, Any]:
    """Build a minimal structured batch when a test double or legacy agent omits one."""
    sources = result.get("sources", []) or []
    task_id = str(active_task.get("id", "") or "").strip()
    sequence = len(state.get("research_batches", []) or []) + 1
    batch_id_prefix = _task_id_from_title(task_id or instruction or "research batch", sequence)
    return {
        "batch_id": f"{batch_id_prefix}-{sequence:03d}",
        "task_id": task_id,
        "instruction": short_text(instruction, 1000),
        "status": result.get("research_status", "complete"),
        "confidence": result.get("confidence", 0.0),
        "summary": assessment_record.get("summary", ""),
        "answered": result.get("answered", []) or [],
        "entities": result.get("entities", []) or assessment_record.get("entities", []) or [],
        "open_questions": result.get("open_questions", []) or [],
        "evidence_gaps": result.get("evidence_gaps", []) or [],
        "next_queries": result.get("next_queries", []) or [],
        "proposed_tasks": result.get("proposed_tasks", []) or assessment_record.get("proposed_tasks", []) or [],
        "source_refs": _compact_batch_source_refs(sources),
        "search_stats": {
            "search_calls": result.get("search_calls", 0),
            "blocked_tool_calls": result.get("blocked_tool_calls", 0),
            "source_count": len(sources) if isinstance(sources, list) else 0,
        },
    }


def _compact_batch_source_refs(sources: Any) -> list[dict[str, Any]]:
    """Build compact source references without carrying full raw source text."""
    if not isinstance(sources, list):
        return []
    refs: list[dict[str, Any]] = []
    for source in sources[:20]:
        if not isinstance(source, dict):
            continue
        url = str(source.get("url", "") or "").strip()
        if not url:
            continue
        refs.append(
            {
                "title": short_text(source.get("title", "") or url, 160),
                "url": url,
                "content_preview": short_text(source.get("content", ""), 260),
                "score": source.get("score"),
            }
        )
    return refs


def _normalize_task_plan(tasks: Any) -> list[dict[str, Any]]:
    """Normalize user/model-provided task-plan payload into state items."""
    if isinstance(tasks, str):
        try:
            tasks = json.loads(tasks)
        except json.JSONDecodeError as exc:
            raise ValueError(f"tasks must be a JSON list or list object: {exc}") from exc
    if isinstance(tasks, dict):
        tasks = tasks.get("tasks") or tasks.get("items") or [tasks]
    if not isinstance(tasks, list):
        raise ValueError("tasks must be a list")

    normalized: list[dict[str, Any]] = []
    for index, raw_item in enumerate(tasks, start=1):
        if not isinstance(raw_item, dict):
            continue
        title = str(raw_item.get("title") or raw_item.get("task") or raw_item.get("instruction") or "").strip()
        instruction = str(raw_item.get("instruction") or title).strip()
        task_id = str(raw_item.get("id") or _task_id_from_title(title, index)).strip()
        if not title:
            title = f"Task {index}"
        item = {
            "id": task_id,
            "title": title,
            "status": _normalize_status(raw_item.get("status", "pending")),
            "agent": _normalize_agent(raw_item.get("agent", "")),
            "instruction": instruction,
            "result": str(raw_item.get("result", "")).strip(),
        }
        _copy_optional_task_metadata(raw_item, item)
        normalized.append(item)
    if not normalized:
        raise ValueError("tasks must contain at least one task object")
    return normalized[:20]


def _apply_blocked_split_policy(
    state: FishState,
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Allow each blocked task to be split only once."""
    parent = _blocked_split_parent(state, items)
    if not parent:
        return items, ""
    parent_id = str(parent.get("id", "") or "").strip()
    parent_depth = _blocked_split_depth(parent)
    parent_split_count = _blocked_split_count(parent)
    if parent_depth >= BLOCKED_SPLIT_LIMIT or parent_split_count >= BLOCKED_SPLIT_LIMIT:
        return [], (
            f"Blocked task {parent_id} already reached the split limit; "
            "leave it blocked for human handling or skip to next_task_id."
        )

    child_depth = parent_depth + 1
    child_items: list[dict[str, Any]] = []
    for item in items:
        item = dict(item)
        if str(item.get("id", "") or "").strip() == parent_id:
            continue
        item["parent_task_id"] = str(item.get("parent_task_id") or parent_id)
        item["blocked_split_depth"] = child_depth
        item["split_from_blocked"] = True
        child_items.append(item)
    if not child_items:
        return [], f"No child tasks were provided for blocked task {parent_id}."
    child_items.append(
        {
            "id": parent_id,
            "status": "blocked",
            "blocked_split_count": parent_split_count + 1,
            "result": str(parent.get("result", "") or "").strip(),
        }
    )
    return child_items, f"Split blocked task {parent_id}; further blocked splits for this lineage are disabled."


def _blocked_split_parent(state: FishState, items: list[dict[str, Any]]) -> dict[str, Any]:
    """Find the blocked task that incoming items are trying to split."""
    task_plan = state.get("task_plan", [])
    if not isinstance(task_plan, list):
        return {}
    tasks_by_id = {
        str(item.get("id", "") or "").strip(): item
        for item in task_plan
        if isinstance(item, dict) and str(item.get("id", "") or "").strip()
    }
    for item in items:
        parent_id = str(item.get("parent_task_id", "") or "").strip()
        parent = tasks_by_id.get(parent_id)
        if _is_blocked_task(parent):
            return dict(parent)
    latest_assessment = state.get("latest_research_assessment", {})
    if isinstance(latest_assessment, dict):
        parent_id = str(latest_assessment.get("task_id", "") or "").strip()
        parent = tasks_by_id.get(parent_id)
        if _is_blocked_task(parent):
            return dict(parent)
    return {}


def _is_blocked_task(task: Any) -> bool:
    """Return whether a task-plan item is currently blocked."""
    return isinstance(task, dict) and _normalize_status(task.get("status", "")) == "blocked"


def _blocked_split_depth(task: dict[str, Any]) -> int:
    """Return a task's blocked-split lineage depth."""
    return _coerce_nonnegative_int(task.get("blocked_split_depth", 0))


def _blocked_split_count(task: dict[str, Any]) -> int:
    """Return how many times this blocked task has already been split."""
    return _coerce_nonnegative_int(task.get("blocked_split_count", 0))


def _coerce_nonnegative_int(value: Any) -> int:
    """Coerce model/user metadata into a non-negative integer."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def _copy_optional_task_metadata(raw_item: dict[str, Any], item: dict[str, Any]) -> None:
    """Copy supported task metadata fields into a normalized task item."""
    parent_task_id = str(raw_item.get("parent_task_id", "") or "").strip()
    if parent_task_id:
        item["parent_task_id"] = parent_task_id
    for key in ("blocked_split_depth", "blocked_split_count"):
        if key in raw_item:
            item[key] = _coerce_nonnegative_int(raw_item.get(key))
    for key in ("manual_required", "split_from_blocked"):
        if key in raw_item:
            item[key] = bool(raw_item.get(key))
    skip_reason = str(raw_item.get("skip_reason", "") or "").strip()
    if skip_reason:
        item["skip_reason"] = skip_reason

def _preserve_existing_task_statuses(state: FishState, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Avoid resetting progressed or already-planned tasks during replanning."""
    existing_plan = state.get("task_plan", [])
    if not isinstance(existing_plan, list):
        return items
    existing_by_id = {
        str(item.get("id", "")): item
        for item in existing_plan
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    }
    existing_by_signature = {
        signature: item
        for item in existing_plan
        if isinstance(item, dict)
        for signature in [_task_signature(item)]
        if signature
    }
    preserved: list[dict[str, Any]] = []
    emitted_ids: set[str] = set()
    for item in items:
        item = dict(item)
        original_id = str(item.get("id", "") or "").strip()
        existing = existing_by_id.get(original_id)
        matched_by_signature = False
        if not isinstance(existing, dict):
            existing = existing_by_signature.get(_task_signature(item))
            matched_by_signature = isinstance(existing, dict)
        if isinstance(existing, dict):
            existing_id = str(existing.get("id", "") or "").strip()
            if existing_id:
                item["id"] = existing_id
            if matched_by_signature:
                item["title"] = existing.get("title", item.get("title", ""))
                item["agent"] = existing.get("agent", item.get("agent", ""))
                item["instruction"] = existing.get("instruction", item.get("instruction", ""))
            existing_status = _normalize_status(existing.get("status", "pending"))
            incoming_status = _normalize_status(item.get("status", "pending"))
            if incoming_status == "pending" and existing_status in {"pending", "in_progress", "completed", "blocked"}:
                item["status"] = existing_status
                if existing.get("result") and not item.get("result"):
                    item["result"] = str(existing.get("result", "")).strip()
        item_id = str(item.get("id", "") or "").strip()
        if item_id and item_id in emitted_ids:
            continue
        if item_id:
            emitted_ids.add(item_id)
        preserved.append(item)
    return preserved


def _task_plan_update_changes_state(state: FishState, items: list[dict[str, Any]], plan_summary: str) -> bool:
    """Return whether a task-plan write would materially change state."""
    preview: FishState = {"task_plan": list(state.get("task_plan", []) or [])}
    apply_state_update(preview, {"task_plan": items})
    current_snapshot = _task_plan_snapshot(state.get("task_plan", []))
    preview_snapshot = _task_plan_snapshot(preview.get("task_plan", []))
    if current_snapshot != preview_snapshot:
        return True
    current_summary = str(state.get("plan_summary", "") or "").strip()
    return bool(plan_summary and current_summary and plan_summary != current_summary)


def _task_plan_snapshot(value: Any) -> list[tuple[str, str, str, str, str, str, str, int, int, bool, str]]:
    """Build a comparable task-plan representation."""
    if not isinstance(value, list):
        return []
    snapshot: list[tuple[str, str, str, str, str, str, str, int, int, bool, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        snapshot.append(
            (
                str(item.get("id", "") or "").strip(),
                short_text(item.get("title", ""), 300),
                _normalize_status(item.get("status", "pending")),
                _normalize_agent(item.get("agent", "")),
                short_text(item.get("instruction", ""), 1000),
                short_text(item.get("result", ""), 600),
                str(item.get("parent_task_id", "") or "").strip(),
                _coerce_nonnegative_int(item.get("blocked_split_depth", 0)),
                _coerce_nonnegative_int(item.get("blocked_split_count", 0)),
                bool(item.get("manual_required", False)),
                str(item.get("skip_reason", "") or "").strip(),
            )
        )
    return snapshot


def _task_signature(item: dict[str, Any]) -> tuple[str, str] | None:
    """Return a stable semantic signature for task deduplication."""
    agent = _normalize_agent(item.get("agent", ""))
    instruction = str(item.get("instruction") or item.get("title") or "").strip().lower()
    instruction = re.sub(r"\s+", " ", instruction)
    if not agent or not instruction:
        return None
    return agent, instruction


def _active_task_id_after_plan_write(state: FishState, items: list[dict[str, Any]]) -> str:
    """Keep an in-progress active task when replanning preserves it."""
    active_task_id = str(state.get("active_task_id", "") or "").strip()
    if not active_task_id:
        return ""
    for item in items:
        if str(item.get("id", "")) == active_task_id and item.get("status") == "in_progress":
            return active_task_id
    return ""


def _task_id_from_title(title: str, index: int) -> str:
    """Create a stable-enough task id from title text."""
    words = re.findall(r"[A-Za-z0-9]+", title.lower())
    if words:
        return "-".join(words[:5])
    return f"task-{index}"


def _normalize_status(status: Any) -> str:
    """Normalize task status."""
    value = str(status or "pending").strip().lower()
    return value if value in TASK_STATUSES else "pending"


def _normalize_agent(agent: Any) -> str:
    """Normalize task owner agent."""
    value = str(agent or "").strip()
    if value in TASK_AGENTS:
        return value
    key = re.sub(r"[^a-z]", "", value.lower())
    return TASK_AGENT_ALIASES.get(key, "")


def _task_plan_summary(items: list[dict[str, Any]]) -> str:
    """Build a compact task-plan summary."""
    return "; ".join(f"{item['id']}={item['status']}" for item in items)


def _next_task_id(state: FishState, agent: str = "") -> str:
    """Return the next pending/in-progress task id."""
    task = _select_task(state, agent)
    return str(task.get("id", "")) if task else ""


def _select_task(state: FishState, agent: str = "", task_id: str = "") -> dict[str, Any]:
    """Select the explicit or next open task for an agent."""
    task_plan = state.get("task_plan", [])
    if not isinstance(task_plan, list):
        return {}
    explicit_task_id = str(task_id or "").strip()
    if explicit_task_id:
        task = _find_task_by_id(task_plan, explicit_task_id)
        if task and _task_matches_agent(task, agent):
            return task
        return {}

    active_task_id = str(state.get("active_task_id", "") or "").strip()
    if active_task_id:
        task = _find_task_by_id(task_plan, active_task_id)
        if task and _task_matches_agent(task, agent):
            return task

    for status in ("in_progress", "pending"):
        for item in task_plan:
            if not isinstance(item, dict):
                continue
            if item.get("status", "pending") != status:
                continue
            if item.get("manual_required"):
                continue
            if not _task_matches_agent(item, agent):
                continue
            return dict(item)
    return {}


def _find_task_by_id(task_plan: list[Any], task_id: str) -> dict[str, Any]:
    """Find a task by id."""
    for item in task_plan:
        if isinstance(item, dict) and str(item.get("id", "")) == task_id:
            return dict(item)
    return {}


def _task_matches_agent(task: dict[str, Any], agent: str = "") -> bool:
    """Return whether a task can be handled by the requested agent."""
    if not agent:
        return True
    task_agent = str(task.get("agent", "") or "").strip()
    return task_agent == agent


def _planned_task_required(state: FishState, agent: str, task_id: str = "") -> bool:
    """Return whether a failed activation should block the handoff."""
    if str(task_id or "").strip():
        return True
    if agent not in {"searchAgent", "codeAgent"}:
        return False
    task_plan = state.get("task_plan", [])
    return isinstance(task_plan, list) and any(isinstance(item, dict) for item in task_plan)


def _blocked_handoff(agent: str, task_id: str, instruction: str, writer: Writer) -> dict[str, Any]:
    """Report a planner handoff that could not be matched to a valid task."""
    message = (
        f"No {agent} task matches task_id={task_id!r}. "
        "Use the task id owned by this agent, or update the task plan with the correct agent."
    )
    writer(
        {
            "type": "handoff_blocked",
            "to": agent,
            "task_id": str(task_id or ""),
            "instruction": short_text(instruction, 600),
            "error": message,
        }
    )
    return {"ok": False, "error": message}


def _activate_task(
    state: FishState,
    state_updates: FishState | None,
    agent: str,
    task_id: str,
    instruction: str,
) -> dict[str, Any]:
    """Mark the selected task in progress before calling a child agent."""
    active_task = _select_task(state, agent, task_id)
    if not active_task:
        return {}
    task_id = str(active_task.get("id", ""))
    delta: FishState = {
        "active_task_id": task_id,
        "task_plan": [
            {
                "id": task_id,
                "status": "in_progress",
                "agent": active_task.get("agent") or agent,
                "instruction": active_task.get("instruction") or instruction,
            }
        ],
    }
    _record_state_delta(state, state_updates, delta)
    return active_task


def _instruction_for_task(instruction: str, active_task: dict[str, Any]) -> str:
    """Prefer explicit instruction, falling back to the selected task instruction."""
    instruction = str(instruction or "").strip()
    if instruction:
        return instruction
    return str(active_task.get("instruction", "")).strip()


def _task_finished_delta(
    active_task: dict[str, Any],
    result: dict[str, Any],
    *,
    completed: bool | None = None,
) -> FishState:
    """Build task-plan update after a child-agent result."""
    task_id = str(active_task.get("id", "")).strip()
    if not task_id:
        return {}
    if completed is None:
        completed = bool(result.get("ok", True))
        if result.get("save_required") and not result.get("saved"):
            completed = False
        if _research_result_has_unresolved_work(result):
            completed = False
    status = "completed" if completed else "blocked"
    task_update: dict[str, Any] = {
        "id": task_id,
        "status": status,
        "result": short_text(result.get("summary", "") or result.get("error", ""), 600),
    }
    depth = _blocked_split_depth(active_task)
    if depth:
        task_update["blocked_split_depth"] = depth
        parent_task_id = str(active_task.get("parent_task_id", "") or "").strip()
        if parent_task_id:
            task_update["parent_task_id"] = parent_task_id
    if status == "blocked" and depth >= BLOCKED_SPLIT_LIMIT:
        task_update["manual_required"] = True
        task_update["skip_reason"] = "blocked_after_split_limit"
    return {
        "active_task_id": "",
        "task_plan": [task_update],
    }


def _research_result_has_unresolved_work(result: dict[str, Any]) -> bool:
    """Return whether a SearchAgent-style result still has research gaps."""
    research_status = str(result.get("research_status", "") or "").strip()
    if research_status and research_status != "complete":
        return True
    return bool(_meaningful_items(result.get("open_questions", [])) or _meaningful_items(result.get("evidence_gaps", [])))


def _meaningful_items(value: Any) -> list[str]:
    """Normalize issue-like fields and ignore explicit no-issue placeholders."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    no_issue_markers = {"", "none", "n/a", "na", "null", "无", "无。", "暂无", "没有", "无明显缺口"}
    items: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text.lower() not in no_issue_markers:
            items.append(text)
    return items
