from __future__ import annotations

from pathlib import Path
from typing import Any

from fishclaw.agents import planner_tools
from fishclaw.agents.planner_tools import _call_code_agent, _call_search_agent
from fishclaw.state import FishRuntime, FishState


def test_search_agent_tool_records_reducer_delta(tmp_path: Path, monkeypatch) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "demo",
        "search_notes": "old note",
        "sources": [{"url": "https://example.com/a", "title": "old"}],
        "handoffs": [{"to": "searchAgent", "instruction": "old", "summary": "old"}],
    }
    state_updates: FishState = {}
    events: list[dict[str, Any]] = []

    def fake_run_search_agent(state_arg: FishState, instruction: str, *, writer=None) -> dict[str, Any]:
        assert state_arg is state
        assert instruction == "lookup"
        return {
            "summary": "new note",
            "sources": [
                {"url": "https://example.com/a", "title": "duplicate"},
                {"url": "https://example.com/b", "title": "new"},
            ],
        }

    monkeypatch.setattr(planner_tools, "run_search_agent", fake_run_search_agent)

    result = _call_search_agent(state, events.append, "lookup", state_updates)

    assert result["ok"] is True
    assert state["search_notes"] == "old note\n\nnew note"
    assert [source["url"] for source in state["sources"]] == ["https://example.com/a", "https://example.com/b"]
    assert [handoff["instruction"] for handoff in state["handoffs"]] == ["old", "lookup"]
    assert state_updates["search_notes"] == "new note"
    assert [source["url"] for source in state_updates["sources"]] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert [handoff["instruction"] for handoff in state_updates["handoffs"]] == ["lookup"]
    assert [event["type"] for event in events] == ["handoff", "handoff_result"]


def test_code_agent_tool_records_reducer_delta(tmp_path: Path, monkeypatch) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "demo",
        "code_summary": "old summary",
        "handoffs": [],
    }
    state_updates: FishState = {}

    def fake_run_code_agent(state_arg: FishState, instruction: str, *, writer=None) -> dict[str, Any]:
        assert state_arg is state
        assert instruction == "implement"
        return {"summary": "new summary"}

    monkeypatch.setattr(planner_tools, "run_code_agent", fake_run_code_agent)

    result = _call_code_agent(state, lambda _: None, "implement", state_updates)

    assert result == {"ok": True, "summary": "new summary"}
    assert state["code_summary"] == "old summary\n\nnew summary"
    assert [handoff["instruction"] for handoff in state["handoffs"]] == ["implement"]
    assert state_updates["code_summary"] == "new summary"
    assert [handoff["instruction"] for handoff in state_updates["handoffs"]] == ["implement"]
