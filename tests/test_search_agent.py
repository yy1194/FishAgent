from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from fishclaw.agents import search_agent
from fishclaw.agents.search_agent import _execute_search_tool, run_search_agent
from fishclaw.state import FishRuntime, FishState
from fishclaw.tools.harness import tool_result_json


class _FakeModel:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = responses

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self.responses.pop(0)


class _RecordingAssessmentModel:
    def __init__(self) -> None:
        self.payload: dict = {}

    def invoke(self, messages):
        self.payload = json.loads(messages[1].content)
        return _assessment_response()


def _patch_models(monkeypatch, search_model: _FakeModel, assessment_model: _FakeModel) -> None:
    models = iter([search_model, assessment_model])
    monkeypatch.setattr(search_agent, "create_model", lambda *args, **kwargs: next(models))


def _assessment_response(
    status: str = "complete",
    *,
    next_queries: list[str] | None = None,
    open_questions: list[str] | None = None,
    evidence_gaps: list[str] | None = None,
    entities: list[dict] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=json.dumps(
            {
                "status": status,
                "confidence": 0.8,
                "answered": ["answered"],
                "entities": entities or [],
                "open_questions": open_questions or [],
                "evidence_gaps": evidence_gaps or [],
                "next_queries": next_queries or [],
                "stop_reason": "enough_evidence" if status == "complete" else "incomplete_available",
                "summary": "assessment summary",
            }
        ),
        tool_calls=[],
    )


def test_execute_search_tool_does_not_expose_save_research(tmp_path: Path) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {"runtime": runtime, "task": "demo"}
    call = {
        "name": "SaveResearchTool",
        "args": {
            "title": "World Cup Research",
            "file_name": "world-cup.md",
            "content": "Team and roster notes.",
            "sources": [{"title": "FIFA", "url": "https://www.fifa.com"}],
        },
        "id": "save-research-call",
    }

    message = _execute_search_tool(state, call)
    result = json.loads(str(message.content))

    assert result["ok"] is False
    assert "unknown tool" in result["error"]
    assert not (runtime.workspace / "research" / "world-cup.md").exists()


def test_run_search_agent_collects_without_creating_report(tmp_path: Path, monkeypatch) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {"runtime": runtime, "task": "查询资料并记录下来"}
    search_model = _FakeModel(
        [
            SimpleNamespace(
                content="I will search now.",
                tool_calls=[
                    {
                        "name": "WebSearchTool",
                        "args": {"query": "world cup squads", "max_results": 1},
                        "id": "search-1",
                    }
                ],
            )
        ]
    )
    assessment_model = _FakeModel(
        [
            _assessment_response(
                entities=[
                    {
                        "name": "France",
                        "type": "team",
                        "status": "partial",
                        "facts": ["Found roster notes."],
                        "source_urls": ["https://www.fifa.com"],
                    }
                ]
            )
        ]
    )
    original_execute = search_agent._execute_search_tool

    def fake_execute(state_arg: FishState, call: dict) -> ToolMessage:
        if call.get("name") == "WebSearchTool":
            return ToolMessage(
                content=tool_result_json(
                    {
                        "ok": True,
                        "query": call["args"]["query"],
                        "answer": "Found roster notes.",
                        "results": [{"title": "FIFA", "url": "https://www.fifa.com", "content": "official notes"}],
                    }
                ),
                name="WebSearchTool",
                tool_call_id=call.get("id") or "search-1",
            )
        return original_execute(state_arg, call)

    _patch_models(monkeypatch, search_model, assessment_model)
    monkeypatch.setattr(search_agent, "_execute_search_tool", fake_execute)

    result = run_search_agent(state, "请查询并记录下来")

    assert result["save_required"] is False
    assert result["saved"] is False
    assert result["research_status"] == "complete"
    assert "assessment summary" in result["summary"]
    assert "Found roster notes." not in result["summary"]
    assert "https://www.fifa.com" not in result["summary"]
    assert result["assessment_summary"] == "assessment summary"
    assert result["summary"] != "I will search now."
    assert result["search_calls"] == 1
    assert result["research_artifacts"] == []
    assert result["research_batch"]["batch_id"] == "research-batch-001"
    assert result["research_batch"]["status"] == "complete"
    assert result["research_batch"]["answered"] == ["answered"]
    assert result["research_batch"]["entities"][0]["name"] == "France"
    assert result["research_batch"]["entities"][0]["source_urls"] == ["https://www.fifa.com"]
    assert result["research_batch"]["source_refs"][0]["url"] == "https://www.fifa.com"
    assert result["research_batch"]["search_stats"]["source_count"] == 1
    assert not (runtime.workspace / "research" / "research-report.md").exists()


def test_run_search_agent_limits_web_search_calls(tmp_path: Path, monkeypatch) -> None:
    runtime = FishRuntime(workspace=tmp_path, max_search_tool_calls=2, max_search_tool_calls_per_turn=3)
    state: FishState = {"runtime": runtime, "task": "demo"}
    calls = [
        {"name": "WebSearchTool", "args": {"query": f"query {index}", "max_results": 1}, "id": f"search-{index}"}
        for index in range(6)
    ]
    search_model = _FakeModel([SimpleNamespace(content="", tool_calls=calls)])
    assessment_model = _FakeModel([_assessment_response()])
    executed_queries: list[str] = []

    def fake_execute(state_arg: FishState, call: dict) -> ToolMessage:
        executed_queries.append(call["args"]["query"])
        return ToolMessage(
            content=tool_result_json({"ok": True, "query": call["args"]["query"], "answer": "", "results": []}),
            name="WebSearchTool",
            tool_call_id=call.get("id") or "search-call",
        )

    _patch_models(monkeypatch, search_model, assessment_model)
    monkeypatch.setattr(search_agent, "_execute_search_tool", fake_execute)

    result = run_search_agent(state, "search only")

    assert result["search_calls"] == 2
    assert result["blocked_tool_calls"] == 4
    assert executed_queries == ["query 0", "query 1"]


def test_run_search_agent_continues_when_assessment_is_incomplete(tmp_path: Path, monkeypatch) -> None:
    runtime = FishRuntime(workspace=tmp_path, max_search_tool_calls=3, max_search_tool_calls_per_turn=3)
    state: FishState = {"runtime": runtime, "task": "demo"}
    search_model = _FakeModel(
        [
            SimpleNamespace(
                content="",
                tool_calls=[
                    {
                        "name": "WebSearchTool",
                        "args": {"query": "initial query", "max_results": 1},
                        "id": "search-1",
                    }
                ],
            ),
            SimpleNamespace(
                content="",
                tool_calls=[
                    {
                        "name": "WebSearchTool",
                        "args": {"query": "better query", "max_results": 1},
                        "id": "search-2",
                    }
                ],
            ),
        ]
    )
    assessment_model = _FakeModel(
        [
            _assessment_response(
                "incomplete",
                next_queries=["better query"],
                open_questions=["need better evidence"],
            ),
            _assessment_response("complete"),
        ]
    )
    executed_queries: list[str] = []

    def fake_execute(state_arg: FishState, call: dict) -> ToolMessage:
        executed_queries.append(call["args"]["query"])
        return ToolMessage(
            content=tool_result_json(
                {
                    "ok": True,
                    "query": call["args"]["query"],
                    "answer": f"answer for {call['args']['query']}",
                    "results": [{"title": call["args"]["query"], "url": f"https://example.com/{len(executed_queries)}"}],
                }
            ),
            name="WebSearchTool",
            tool_call_id=call.get("id") or "search-call",
        )

    _patch_models(monkeypatch, search_model, assessment_model)
    monkeypatch.setattr(search_agent, "_execute_search_tool", fake_execute)

    result = run_search_agent(state, "research task")

    assert result["ok"] is True
    assert result["research_status"] == "complete"
    assert result["search_calls"] == 2
    assert executed_queries == ["initial query", "better query"]


def test_run_search_agent_returns_incomplete_from_assessment(tmp_path: Path, monkeypatch) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {"runtime": runtime, "task": "demo"}
    search_model = _FakeModel(
        [
            SimpleNamespace(
                content="",
                tool_calls=[
                    {
                        "name": "WebSearchTool",
                        "args": {"query": "initial query", "max_results": 1},
                        "id": "search-1",
                    }
                ],
            )
        ]
    )
    assessment_model = _FakeModel(
        [
            _assessment_response(
                "incomplete",
                open_questions=["still missing official source"],
                evidence_gaps=["official source missing"],
            )
        ]
    )

    def fake_execute(state_arg: FishState, call: dict) -> ToolMessage:
        return ToolMessage(
            content=tool_result_json(
                {
                    "ok": True,
                    "query": call["args"]["query"],
                    "answer": "partial answer",
                    "results": [{"title": "Partial", "url": "https://example.com/partial"}],
                }
            ),
            name="WebSearchTool",
            tool_call_id=call.get("id") or "search-call",
        )

    _patch_models(monkeypatch, search_model, assessment_model)
    monkeypatch.setattr(search_agent, "_execute_search_tool", fake_execute)

    result = run_search_agent(state, "research task")

    assert result["ok"] is False
    assert result["research_status"] == "incomplete"
    assert result["open_questions"] == ["still missing official source"]
    assert result["evidence_gaps"] == ["official source missing"]
    assert result["proposed_tasks"][0]["agent"] == "searchAgent"
    assert "official source" in result["proposed_tasks"][0]["instruction"]
    assert result["research_batch"]["proposed_tasks"][0]["agent"] == "searchAgent"


def test_run_search_agent_downgrades_complete_with_unresolved_questions(tmp_path: Path, monkeypatch) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {"runtime": runtime, "task": "demo"}
    search_model = _FakeModel(
        [
            SimpleNamespace(
                content="",
                tool_calls=[
                    {
                        "name": "WebSearchTool",
                        "args": {"query": "initial query", "max_results": 1},
                        "id": "search-1",
                    }
                ],
            )
        ]
    )
    assessment_model = _FakeModel(
        [
            _assessment_response(
                "complete",
                open_questions=["still missing official roster"],
                evidence_gaps=["official source missing"],
            )
        ]
    )

    def fake_execute(state_arg: FishState, call: dict) -> ToolMessage:
        return ToolMessage(
            content=tool_result_json(
                {
                    "ok": True,
                    "query": call["args"]["query"],
                    "answer": "partial answer",
                    "results": [{"title": "Partial", "url": "https://example.com/partial"}],
                }
            ),
            name="WebSearchTool",
            tool_call_id=call.get("id") or "search-call",
        )

    _patch_models(monkeypatch, search_model, assessment_model)
    monkeypatch.setattr(search_agent, "_execute_search_tool", fake_execute)

    result = run_search_agent(state, "research task")

    assert result["ok"] is False
    assert result["research_status"] == "incomplete"
    assert result["open_questions"] == ["still missing official roster"]
    assert result["evidence_gaps"] == ["official source missing"]
    assert result["stop_reason"] == "unresolved_research_gaps"
    assert result["proposed_tasks"][0]["agent"] == "searchAgent"


def test_research_assessment_scopes_payload_to_current_task(tmp_path: Path) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "overall task",
        "task_plan": [
            {"id": "task-1", "agent": "searchAgent", "instruction": "other search"},
            {"id": "task-2", "agent": "searchAgent", "instruction": "current search"},
        ],
    }
    model = _RecordingAssessmentModel()

    result = search_agent._assess_research_progress(
        model,
        state,
        "current search",
        {"id": "task-2", "agent": "searchAgent", "instruction": "current search"},
        "summary",
        ["answer"],
        [{"title": "Source", "url": "https://example.com"}],
        1,
        3,
        0,
        False,
    )

    assert result["status"] == "complete"
    assert "task" not in model.payload
    assert "task_plan" not in model.payload
    assert model.payload["planner_instruction"] == "current search"
    assert model.payload["current_task"]["id"] == "task-2"
