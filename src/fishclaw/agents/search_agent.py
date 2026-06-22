"""SearchAgent：负责搜索、资料收集和来源整理。"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from fishclaw.agents.common import Writer, compact_text, last_ai_text, tool_event
from fishclaw.agents.prompts import SEARCH_PROMPT
from fishclaw.memory import build_agent_memory, format_memory, merge_sources
from fishclaw.model import create_model
from fishclaw.state import FishState
from fishclaw.tools.harness import build_search_tools, tool_result_json


RESEARCH_STATUSES = {"complete", "incomplete"}
MAX_ASSESSMENT_SOURCES = 12
MAX_ASSESSMENT_ANSWERS = 8

RESEARCH_ASSESSMENT_PROMPT = """你是 Fishclaw 的 ResearchEvaluator。
你只评估研究进度，不调用工具，不补充事实，不编造来源。

请根据用户任务、planner 指令、当前搜索摘要、搜索答案片段和来源摘要，判断当前研究是否足够完成该指令。

必须只输出 JSON，字段如下：
{
  "status": "complete | incomplete",
  "confidence": 0.0,
  "answered": ["已经回答的问题"],
  "entities": [
    {
      "name": "实体名称，例如球队、公司、论文、产品",
      "type": "team | company | paper | product | person | place | other",
      "status": "found | partial | missing",
      "facts": ["可由来源支撑的事实"],
      "source_urls": ["支撑这些事实的 URL"]
    }
  ],
  "open_questions": ["仍未解决的问题"],
  "evidence_gaps": ["证据不足、来源不权威、信息过时、缺少交叉验证等问题"],
  "next_queries": ["下一轮建议搜索词"],
  "proposed_tasks": [
    {
      "title": "给 planner 的下一步任务标题",
      "agent": "searchAgent | codeAgent",
      "instruction": "可以直接分发给对应 agent 的具体任务指令",
      "reason": "为什么需要这个后续任务"
    }
  ],
  "stop_reason": "enough_evidence | budget_exhausted | no_useful_next_query | incomplete_available | unresolved_research_gaps",
  "summary": "面向后续 planner/CodeAgent 的简洁研究上下文，不要粘贴大段网页原文"
}

判断规则：
- 如果已经可以直接回答 planner 指令，且来源足够支撑结论，status=complete。
- 只要还有明确缺口、证据不足、信息未公开、查询方向未穷尽或只能形成阶段性结果，status=incomplete。
- next_queries 必须具体、可搜索、非重复；不要给泛泛的“继续搜索”。
- status=incomplete 时必须尽量给出 proposed_tasks，供 planner 重新制定计划并重新分发任务。
- proposed_tasks 中，继续补资料用 searchAgent；基于已完成研究创建 Markdown/txt/report 或写文件用 codeAgent。
- 不要要求搜索已经 answered 的内容。
- entities 用来把长研究任务拆成可被下游复用的对象记录，例如球队、公司、论文、产品；没有明确对象时输出空数组。
- summary 应整合关键发现、可用事实范围和重要限制，保持简洁；后续 CodeAgent 会优先看到结构化 research_batch，而不会看到完整 sources 原文。
"""


def run_search_agent(
    state: FishState,
    instruction: str,
    *,
    writer: Writer | None = None,
    active_task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """运行 SearchAgent，并返回研究摘要和来源。"""
    writer = writer or (lambda _: None)
    runtime = state["runtime"]
    model = create_model().bind_tools(build_search_tools(state["runtime"]))
    assessment_model = create_model(temperature=0.0)
    messages: list[Any] = [
        SystemMessage(content=SEARCH_PROMPT),
        HumanMessage(
            content=(
                f"planner 指令：{instruction}\n\n"
                f"当前任务上下文：\n"
                f"{format_memory(build_agent_memory(state, instruction=instruction, active_task=active_task))}"
            )
        ),
    ]
    produced: list[Any] = []
    sources: list[dict[str, Any]] = []
    answers: list[str] = []
    search_calls = 0
    blocked_tool_calls = 0
    save_required = False
    assessment = _default_research_assessment()
    seen_next_queries: set[tuple[str, ...]] = set()
    for _ in range(max(1, runtime.max_agent_loops)):
        response = model.invoke(messages)
        produced.append(response)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            for index, call in enumerate(tool_calls):
                writer({"type": "tool_call", "node": "searchAgent", "name": call.get("name"), "args": call.get("args", {})})
                name = str(call.get("name", ""))
                if index >= runtime.max_search_tool_calls_per_turn:
                    blocked_tool_calls += 1
                    tool_message = _budget_tool_message(call, "per-turn search tool call budget exceeded")
                elif name == "WebSearchTool" and search_calls >= runtime.max_search_tool_calls:
                    blocked_tool_calls += 1
                    tool_message = _budget_tool_message(call, "total WebSearchTool budget exceeded")
                else:
                    if name == "WebSearchTool":
                        search_calls += 1
                    tool_message = _execute_search_tool(state, call)
                produced.append(tool_message)
                messages.append(tool_message)
                event = tool_event("searchAgent", tool_message)
                writer(event)
                parsed = event.get("result") if isinstance(event.get("result"), dict) else {}
                if isinstance(parsed, dict):
                    if parsed.get("answer"):
                        answers.append(str(parsed.get("answer")))
                    sources.extend([item for item in parsed.get("results", []) or [] if isinstance(item, dict)])

        summary = _interim_research_summary(last_ai_text(produced), answers, sources)
        assessment = _assess_research_progress(
            assessment_model,
            state,
            instruction,
            active_task,
            summary,
            answers,
            sources,
            search_calls,
            runtime.max_search_tool_calls,
            blocked_tool_calls,
            save_required,
        )
        writer({"type": "research_assessment", **assessment})

        if assessment["status"] == "complete":
            break

        next_queries = _normalized_queries(assessment.get("next_queries", []))
        if search_calls >= runtime.max_search_tool_calls:
            assessment = _incomplete_assessment(assessment, "budget_exhausted")
            break
        if not next_queries:
            if assessment.get("proposed_tasks") or assessment.get("open_questions") or assessment.get("evidence_gaps"):
                break
            assessment = _incomplete_assessment(assessment, "no_useful_next_query")
            break
        query_key = tuple(next_queries)
        if query_key in seen_next_queries:
            assessment = _incomplete_assessment(assessment, "no_new_next_query")
            break
        seen_next_queries.add(query_key)
        messages.append(HumanMessage(content=_assessment_followup_prompt(assessment, runtime.max_search_tool_calls - search_calls)))
    if assessment["status"] == "incomplete":
        assessment = _incomplete_assessment(assessment, assessment.get("stop_reason") or "agent_loop_limit")
    merged_sources = merge_sources(sources)
    summary = _assessment_context_summary(
        instruction,
        merged_sources,
        assessment,
        search_calls,
        blocked_tool_calls,
    )
    research_status = "complete" if _assessment_is_complete(assessment) else "incomplete"
    research_batch = _build_research_batch(
        state,
        instruction,
        active_task,
        assessment,
        merged_sources,
        search_calls,
        blocked_tool_calls,
        research_status,
    )
    return {
        "ok": research_status == "complete",
        "summary": summary,
        "sources": merged_sources,
        "research_batch": research_batch,
        "research_artifacts": [],
        "save_required": save_required,
        "saved": False,
        "research_status": research_status,
        "confidence": assessment.get("confidence", 0.0),
        "answered": assessment.get("answered", []),
        "open_questions": assessment.get("open_questions", []),
        "evidence_gaps": assessment.get("evidence_gaps", []),
        "next_queries": assessment.get("next_queries", []),
        "proposed_tasks": assessment.get("proposed_tasks", []),
        "entities": assessment.get("entities", []),
        "stop_reason": assessment.get("stop_reason", ""),
        "assessment_summary": assessment.get("summary", ""),
        "search_calls": search_calls,
        "blocked_tool_calls": blocked_tool_calls,
        "messages": produced,
    }


def _build_research_batch(
    state: FishState,
    instruction: str,
    active_task: dict[str, Any] | None,
    assessment: dict[str, Any],
    sources: list[dict[str, Any]],
    search_calls: int,
    blocked_tool_calls: int,
    research_status: str,
) -> dict[str, Any]:
    """Build a structured, non-file research batch for downstream agents."""
    task = _assessment_task(active_task)
    task_id = str(task.get("id", "") or "").strip()
    sequence = len(state.get("research_batches", []) or []) + 1
    batch_id = f"{_batch_slug(task_id or instruction)}-{sequence:03d}"
    return {
        "batch_id": batch_id,
        "task_id": task_id,
        "instruction": compact_text(instruction, 1000),
        "status": research_status,
        "confidence": assessment.get("confidence", 0.0),
        "summary": str(assessment.get("summary", "") or "").strip(),
        "answered": _string_list(assessment.get("answered", []), limit=12),
        "entities": _entity_list(assessment.get("entities", []), sources),
        "open_questions": _issue_list(assessment.get("open_questions", []), limit=12),
        "evidence_gaps": _issue_list(assessment.get("evidence_gaps", []), limit=12),
        "next_queries": _string_list(assessment.get("next_queries", []), limit=8),
        "proposed_tasks": _proposed_task_list(assessment.get("proposed_tasks", []), assessment.get("next_queries", []), assessment.get("open_questions", []), assessment.get("evidence_gaps", [])),
        "source_refs": _source_refs(sources),
        "search_stats": {
            "search_calls": search_calls,
            "blocked_tool_calls": blocked_tool_calls,
            "source_count": len(sources),
        },
    }


def _batch_slug(value: str) -> str:
    """Create a stable batch id prefix without relying on file-system paths."""
    words = re.findall(r"[A-Za-z0-9]+", value.lower())
    if words:
        return "-".join(words[:6])
    return "research-batch"


def _interim_research_summary(model_summary: str, answers: list[str], sources: list[dict[str, Any]]) -> str:
    """Build a short assessment input from collected material."""
    if answers:
        return "\n".join(compact_text(answer, 700) for answer in answers[-MAX_ASSESSMENT_ANSWERS:])
    if sources:
        labels = [_source_label(source) for source in merge_sources(sources)[-MAX_ASSESSMENT_SOURCES:]]
        labels = [label for label in labels if label]
        if labels:
            return "已收集来源：" + "；".join(labels)
    return model_summary.strip()


def _assessment_context_summary(
    instruction: str,
    sources: list[dict[str, Any]],
    assessment: dict[str, Any],
    search_calls: int,
    blocked_tool_calls: int,
) -> str:
    """Build the compact research context passed downstream."""
    lines: list[str] = []
    status = assessment.get("status", "incomplete")
    confidence = assessment.get("confidence", 0.0)
    stop_reason = assessment.get("stop_reason", "") or "not_provided"
    lines.append("ResearchEvaluator 摘要：")
    lines.append(f"- 状态：{status}；置信度：{confidence}；停止原因：{stop_reason}。")

    assessment_summary = str(assessment.get("summary", "") or "").strip()
    if assessment_summary:
        lines.append(f"- 总结：{compact_text(assessment_summary, 800)}")
    else:
        lines.append(f"- 总结：已完成资料收集评估，原始指令为 {compact_text(instruction, 240)}")

    answered = _string_list(assessment.get("answered", []), limit=8)
    if answered:
        lines.append("- 已覆盖：" + "；".join(compact_text(item, 180) for item in answered))

    open_questions = _string_list(assessment.get("open_questions", []), limit=8)
    if open_questions:
        lines.append("- 未解决：" + "；".join(compact_text(item, 180) for item in open_questions))

    evidence_gaps = _string_list(assessment.get("evidence_gaps", []), limit=8)
    if evidence_gaps:
        lines.append("- 证据缺口：" + "；".join(compact_text(item, 180) for item in evidence_gaps))

    next_queries = _string_list(assessment.get("next_queries", []), limit=6)
    if next_queries:
        lines.append("- 建议后续查询：" + "；".join(compact_text(item, 160) for item in next_queries))

    proposed_tasks = _proposed_task_list(
        assessment.get("proposed_tasks", []),
        assessment.get("next_queries", []),
        assessment.get("open_questions", []),
        assessment.get("evidence_gaps", []),
        limit=5,
    )
    if proposed_tasks:
        proposals = [
            f"{item.get('agent', '')}:{compact_text(item.get('instruction', ''), 180)}"
            for item in proposed_tasks
        ]
        lines.append("- 建议 planner 新任务：" + "；".join(proposals))

    lines.append(f"- 来源数量：{len(sources)}；搜索调用：{search_calls} 次；被预算拦截：{blocked_tool_calls} 次。")
    return "\n".join(lines).strip()


def _assessment_is_complete(assessment: dict[str, Any]) -> bool:
    """Return whether the normalized assessment has no unresolved research gaps."""
    return (
        assessment.get("status") == "complete"
        and not _issue_list(assessment.get("open_questions", []), limit=12)
        and not _issue_list(assessment.get("evidence_gaps", []), limit=12)
    )


def _source_label(source: dict[str, Any]) -> str:
    """Create a compact source label for summaries."""
    title = compact_text(source.get("title") or source.get("url") or "", 120)
    url = compact_text(source.get("url") or "", 180)
    if title and url and title != url:
        return f"{title} ({url})"
    return title or url


def _assess_research_progress(
    model: Any,
    state: FishState,
    instruction: str,
    active_task: dict[str, Any] | None,
    summary: str,
    answers: list[str],
    sources: list[dict[str, Any]],
    search_calls: int,
    max_search_calls: int,
    blocked_tool_calls: int,
    save_required: bool,
) -> dict[str, Any]:
    """Ask a no-tool evaluator whether the current research should continue."""
    payload = {
        "planner_instruction": instruction,
        "current_task": _assessment_task(active_task),
        "current_summary": summary,
        "answers": answers[-MAX_ASSESSMENT_ANSWERS:],
        "sources": _compact_assessment_sources(merge_sources(sources)[-MAX_ASSESSMENT_SOURCES:]),
        "search_calls": search_calls,
        "max_search_calls": max_search_calls,
        "blocked_tool_calls": blocked_tool_calls,
        "save_required": save_required,
    }
    try:
        response = model.invoke(
            [
                SystemMessage(content=RESEARCH_ASSESSMENT_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2, default=str)),
            ]
        )
    except Exception as exc:
        return _fallback_assessment(summary, answers, sources, f"assessment_model_error:{type(exc).__name__}")
    content = str(getattr(response, "content", "") or "")
    return _parse_assessment_json(content, summary, answers, sources)


def _assessment_task(active_task: dict[str, Any] | None) -> dict[str, Any]:
    """Compact the delegated task for the research evaluator."""
    if not isinstance(active_task, dict):
        return {}
    return {
        "id": active_task.get("id", ""),
        "title": active_task.get("title", ""),
        "status": active_task.get("status", ""),
        "agent": active_task.get("agent", ""),
        "instruction": active_task.get("instruction", ""),
    }


def _default_research_assessment() -> dict[str, Any]:
    """Default assessment before the first evaluator call."""
    return {
        "status": "incomplete",
        "confidence": 0.0,
        "answered": [],
        "entities": [],
        "open_questions": [],
        "evidence_gaps": [],
        "next_queries": [],
        "proposed_tasks": [],
        "stop_reason": "not_assessed",
        "summary": "",
    }


def _parse_assessment_json(
    content: str,
    summary: str,
    answers: list[str],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Parse and normalize the evaluator JSON output."""
    text = _strip_json_fence(content)
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return _fallback_assessment(summary, answers, sources, "assessment_json_parse_failed")
    if not isinstance(raw, dict):
        return _fallback_assessment(summary, answers, sources, "assessment_json_not_object")
    return _normalize_assessment(raw, bool(summary.strip() or answers or sources))


def _strip_json_fence(content: str) -> str:
    """Remove common markdown JSON fences from model output."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _normalize_assessment(raw: dict[str, Any], has_material: bool) -> dict[str, Any]:
    """Normalize evaluator output into a predictable schema."""
    confidence = _coerce_confidence(raw.get("confidence", 0.0))
    open_questions = _issue_list(raw.get("open_questions", []), limit=12)
    evidence_gaps = _issue_list(raw.get("evidence_gaps", []), limit=12)
    next_queries = _string_list(raw.get("next_queries", []), limit=8)
    raw_status = str(raw.get("status", "") or "").strip().lower()
    if raw_status in {"complete", "completed", "done"}:
        status = "complete"
    elif raw_status in {"incomplete", "needs_more_search", "need_more_search", "partial", "blocked"}:
        status = "incomplete"
    else:
        status = "complete" if has_material and not open_questions and not evidence_gaps else "incomplete"

    downgraded = False
    if status == "complete" and (not has_material or open_questions or evidence_gaps):
        status = "incomplete"
        downgraded = True

    stop_reason = str(raw.get("stop_reason", "") or "").strip()
    if downgraded:
        stop_reason = "unresolved_research_gaps"
    elif status == "complete" and not stop_reason:
        stop_reason = "enough_evidence"
    elif status == "incomplete" and not stop_reason:
        stop_reason = "unresolved_research_gaps" if open_questions or evidence_gaps else "incomplete_available"

    proposed_tasks = _proposed_task_list(
        raw.get("proposed_tasks", []),
        next_queries,
        open_questions,
        evidence_gaps,
    ) if status == "incomplete" else []
    return {
        "status": status,
        "confidence": confidence,
        "answered": _string_list(raw.get("answered", []), limit=12),
        "entities": _entity_list(raw.get("entities", []), []),
        "open_questions": open_questions,
        "evidence_gaps": evidence_gaps,
        "next_queries": next_queries,
        "proposed_tasks": proposed_tasks,
        "stop_reason": stop_reason,
        "summary": str(raw.get("summary", "") or "").strip(),
    }


def _fallback_assessment(
    summary: str,
    answers: list[str],
    sources: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    """Keep SearchAgent stable when evaluator output is missing or invalid."""
    has_material = bool(summary.strip() or answers or sources)
    open_questions = [] if has_material else ["SearchAgent did not collect usable research material."]
    evidence_gaps = [reason]
    return {
        "status": "incomplete",
        "confidence": 0.0,
        "answered": [],
        "entities": [],
        "open_questions": open_questions,
        "evidence_gaps": evidence_gaps,
        "next_queries": [],
        "proposed_tasks": _proposed_task_list([], [], open_questions, evidence_gaps),
        "stop_reason": reason,
        "summary": "评估器不可用，已使用保守兜底判断。",
    }


def _incomplete_assessment(assessment: dict[str, Any], reason: str) -> dict[str, Any]:
    """Convert an in-progress assessment into an incomplete result."""
    updated = dict(assessment)
    updated["status"] = "incomplete"
    updated["stop_reason"] = reason
    gaps = _string_list(updated.get("evidence_gaps", []), limit=12)
    if reason in {"budget_exhausted", "no_useful_next_query", "no_new_next_query", "agent_loop_limit"} and reason not in gaps:
        gaps.append(reason)
    updated["evidence_gaps"] = gaps
    updated["next_queries"] = _string_list(updated.get("next_queries", []), limit=8)
    updated["proposed_tasks"] = _proposed_task_list(
        updated.get("proposed_tasks", []),
        updated.get("next_queries", []),
        updated.get("open_questions", []),
        updated.get("evidence_gaps", []),
    )
    return updated

def _proposed_task_list(
    value: Any,
    next_queries: Any,
    open_questions: Any,
    evidence_gaps: Any,
    *,
    limit: int = 8,
) -> list[dict[str, str]]:
    """Normalize evaluator task proposals for planner replanning."""
    raw_items: list[Any]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [value]
        raw_items = parsed if isinstance(parsed, list) else [parsed]
    elif isinstance(value, dict):
        nested = value.get("tasks") or value.get("items") or value.get("proposed_tasks")
        raw_items = nested if isinstance(nested, list) else [value]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = []

    proposals: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_items:
        if isinstance(item, dict):
            title = str(item.get("title") or item.get("task") or item.get("query") or "").strip()
            instruction = str(item.get("instruction") or item.get("query") or item.get("task") or title).strip()
            reason = str(item.get("reason") or item.get("rationale") or "").strip()
            agent = _normalize_proposal_agent(item.get("agent", ""))
        else:
            instruction = str(item or "").strip()
            title = instruction
            reason = "ResearchEvaluator proposed this follow-up from unresolved research gaps."
            agent = "searchAgent"
        _append_proposal(proposals, seen, title, agent, instruction, reason, limit)

    if not proposals:
        for query in _string_list(next_queries, limit=limit):
            _append_proposal(
                proposals,
                seen,
                f"Follow up research: {query}",
                "searchAgent",
                f"围绕查询词继续收集权威资料和来源：{query}",
                "ResearchEvaluator suggested this concrete next query.",
                limit,
            )
    if not proposals:
        issues = _issue_list(open_questions, limit=4) + _issue_list(evidence_gaps, limit=4)
        if issues:
            issue_text = "；".join(compact_text(item, 120) for item in issues)
            _append_proposal(
                proposals,
                seen,
                "补充未完成的研究缺口",
                "searchAgent",
                f"继续围绕以下未完成研究缺口收集权威资料和来源：{issue_text}",
                "ResearchEvaluator found unresolved research gaps but no concrete query was supplied.",
                limit,
            )
    return proposals[:limit]


def _append_proposal(
    proposals: list[dict[str, str]],
    seen: set[tuple[str, str]],
    title: str,
    agent: str,
    instruction: str,
    reason: str,
    limit: int,
) -> None:
    """Append a deduplicated planner proposal."""
    if len(proposals) >= limit:
        return
    instruction = str(instruction or "").strip()
    if not instruction:
        return
    agent = _normalize_proposal_agent(agent)
    title = str(title or instruction).strip()
    reason = str(reason or "").strip()
    key = (agent, instruction.lower())
    if key in seen:
        return
    seen.add(key)
    proposals.append(
        {
            "title": compact_text(title, 160),
            "agent": agent,
            "instruction": compact_text(instruction, 700),
            "reason": compact_text(reason, 260),
        }
    )


def _normalize_proposal_agent(value: Any) -> str:
    """Normalize a ResearchEvaluator proposal owner."""
    text = str(value or "").strip()
    key = re.sub(r"[^a-z]", "", text.lower())
    if key in {"code", "codeagent", "codeagenttool"}:
        return "codeAgent"
    return "searchAgent"

def _coerce_confidence(value: Any) -> float:
    """Coerce confidence into the 0..1 range."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(confidence, 1.0))


def _string_list(value: Any, *, limit: int) -> list[str]:
    """Normalize arbitrary model output into a short string list."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            items.append(text[:500])
        if len(items) >= limit:
            break
    return items


def _entity_list(value: Any, sources: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    """Normalize evaluator entity records into compact downstream data."""
    if not isinstance(value, list):
        return []
    source_urls = {str(source.get("url", "")).strip() for source in sources if isinstance(source, dict)}
    entities: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        if not name:
            continue
        urls = _string_list(item.get("source_urls", []), limit=8)
        if source_urls:
            urls = [url for url in urls if url in source_urls]
        entities.append(
            {
                "name": compact_text(name, 120),
                "type": compact_text(item.get("type", "other") or "other", 60),
                "status": _normalize_entity_status(item.get("status", "")),
                "facts": _string_list(item.get("facts", []), limit=8),
                "source_urls": urls,
            }
        )
        if len(entities) >= limit:
            break
    return entities


def _normalize_entity_status(value: Any) -> str:
    """Normalize entity coverage status for research batches."""
    status = str(value or "").strip().lower()
    return status if status in {"found", "partial", "missing"} else "found"


def _source_refs(sources: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    """Return compact source references for a research batch."""
    refs: list[dict[str, Any]] = []
    for source in sources[:limit]:
        if not isinstance(source, dict):
            continue
        url = str(source.get("url", "") or "").strip()
        if not url:
            continue
        refs.append(
            {
                "title": compact_text(source.get("title", "") or url, 160),
                "url": url,
                "content_preview": compact_text(source.get("content", ""), 260),
                "score": source.get("score"),
            }
        )
    return refs


def _issue_list(value: Any, *, limit: int) -> list[str]:
    """Normalize model issue lists and drop explicit no-issue placeholders."""
    no_issue_markers = {"", "none", "n/a", "na", "null", "无", "无。", "暂无", "没有", "无明显缺口"}
    issues: list[str] = []
    for item in _string_list(value, limit=limit):
        normalized = item.strip().lower()
        if normalized in no_issue_markers:
            continue
        issues.append(item)
    return issues


def _normalized_queries(value: Any) -> list[str]:
    """Normalize next query strings for loop-control comparisons."""
    return [query.lower() for query in _string_list(value, limit=8)]


def _compact_assessment_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep source evidence small enough for evaluator prompts."""
    compact: list[dict[str, Any]] = []
    for source in sources:
        compact.append(
            {
                "title": str(source.get("title", ""))[:180],
                "url": str(source.get("url", ""))[:300],
                "content": str(source.get("content", ""))[:700],
                "score": source.get("score"),
            }
        )
    return compact


def _assessment_followup_prompt(assessment: dict[str, Any], remaining_search_calls: int) -> str:
    """Tell SearchAgent how to continue after evaluator feedback."""
    return (
        "ResearchEvaluator 判断当前研究还需要继续搜索。\n"
        f"- 当前状态摘要：{assessment.get('summary', '')}\n"
        f"- 已回答：{assessment.get('answered', [])}\n"
        f"- 未解决问题：{assessment.get('open_questions', [])}\n"
        f"- 证据缺口：{assessment.get('evidence_gaps', [])}\n"
        f"- 剩余 WebSearchTool 预算：{max(0, remaining_search_calls)}\n"
        f"- 下一轮优先查询：{assessment.get('next_queries', [])}\n\n"
        "请下一轮只围绕 next_queries 中的高价值查询调用 WebSearchTool。"
        "如果确认没有更多可靠信息，请停止搜索并明确说明阶段性缺口。"
    )


def _execute_search_tool(state: FishState, call: dict[str, Any]) -> ToolMessage:
    """执行 SearchAgent 的搜索或研究保存工具调用。"""
    tools = {tool.name: tool for tool in build_search_tools(state["runtime"])}
    name = str(call.get("name", ""))
    args = call.get("args") or {}
    tool = tools.get(name)
    if tool is None:
        result = {"ok": False, "error": f"unknown tool: {name}"}
    else:
        try:
            result = tool.invoke(args)
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return ToolMessage(content=tool_result_json(result), name=name, tool_call_id=call.get("id") or f"{name}-call")


def _budget_tool_message(call: dict[str, Any], reason: str) -> ToolMessage:
    """Return a synthetic ToolMessage for skipped over-budget tool calls."""
    name = str(call.get("name", ""))
    result = {"ok": False, "error": reason, "skipped": True}
    return ToolMessage(content=tool_result_json(result), name=name, tool_call_id=call.get("id") or f"{name}-call")
