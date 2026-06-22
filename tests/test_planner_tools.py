from __future__ import annotations

from pathlib import Path
from typing import Any

from fishclaw.agents import planner_tools
from fishclaw.agents.planner_tools import _call_code_agent, _call_search_agent
from fishclaw.state import FishRuntime, FishState


def test_write_task_plan_records_reducer_delta(tmp_path: Path) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {"runtime": runtime, "task": "demo", "task_plan": []}
    state_updates: FishState = {}
    events: list[dict[str, Any]] = []

    result = planner_tools._write_task_plan(
        state,
        events.append,
        [
            {
                "id": "research-qualified-teams",
                "title": "查询已晋级球队",
                "agent": "searchAgent",
                "instruction": "查询 2026 World Cup 已确认出战国家并保存来源。",
            }
        ],
        state_updates=state_updates,
    )

    assert result["ok"] is True
    assert result["task_count"] == 1
    assert result["next_task_id"] == "research-qualified-teams"
    assert state["task_plan"][0]["status"] == "pending"
    assert state_updates["task_plan"][0]["id"] == "research-qualified-teams"
    assert state["plan_summary"] == "research-qualified-teams=pending"
    assert events == [
        {
            "type": "task_plan_updated",
            "task_count": 1,
            "plan_summary": "research-qualified-teams=pending",
        }
    ]


def test_write_task_plan_preserves_existing_progressed_status(tmp_path: Path) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "demo",
        "task_plan": [
            {
                "id": "task-1",
                "title": "Research",
                "status": "completed",
                "agent": "searchAgent",
                "instruction": "old research",
                "result": "done",
            }
        ],
    }

    result = planner_tools._write_task_plan(
        state,
        lambda _: None,
        [
            {
                "id": "task-1",
                "title": "Research",
                "agent": "searchAgent",
                "instruction": "updated research",
            }
        ],
    )

    assert result["ok"] is True
    assert state["task_plan"][0]["status"] == "completed"
    assert state["task_plan"][0]["result"] == "done"
    assert state["plan_summary"] == "task-1=completed"



def test_write_task_plan_deduplicates_equivalent_existing_task(tmp_path: Path) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "demo",
        "task_plan": [
            {
                "id": "task-1",
                "title": "补充初步名单来源",
                "status": "pending",
                "agent": "searchAgent",
                "instruction": "继续查询 2026 World Cup preliminary squads 的官方来源。",
            }
        ],
        "plan_summary": "task-1=pending",
    }
    state_updates: FishState = {}
    events: list[dict[str, Any]] = []

    result = planner_tools._write_task_plan(
        state,
        events.append,
        [
            {
                "id": "task-2",
                "title": "再次补充初步名单来源",
                "status": "pending",
                "agent": "SearchAgentTool",
                "instruction": "继续查询 2026 World Cup preliminary squads 的官方来源。",
            }
        ],
        state_updates=state_updates,
    )

    assert result["ok"] is True
    assert result["changed"] is False
    assert result["next_task_id"] == "task-1"
    assert [item["id"] for item in state["task_plan"]] == ["task-1"]
    assert state_updates == {}
    assert events[-1]["type"] == "task_plan_unchanged"


def test_write_task_plan_normalizes_tool_name_agents(tmp_path: Path) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {"runtime": runtime, "task": "demo", "task_plan": []}

    result = planner_tools._write_task_plan(
        state,
        lambda _: None,
        [
            {
                "id": "task-1",
                "title": "Search",
                "agent": "SearchAgentTool",
                "instruction": "search data",
            },
            {
                "id": "task-2",
                "title": "Code",
                "agent": "CodeAgentTool",
                "instruction": "write file",
            },
        ],
    )

    assert result["ok"] is True
    assert [item["agent"] for item in state["task_plan"]] == ["searchAgent", "codeAgent"]


def test_write_task_plan_preserves_active_task_when_replanned(tmp_path: Path) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "demo",
        "active_task_id": "task-1",
        "task_plan": [
            {
                "id": "task-1",
                "title": "Search",
                "status": "in_progress",
                "agent": "searchAgent",
                "instruction": "search data",
            }
        ],
    }

    result = planner_tools._write_task_plan(
        state,
        lambda _: None,
        [
            {
                "id": "task-1",
                "title": "Search",
                "status": "pending",
                "agent": "SearchAgentTool",
                "instruction": "search data",
            }
        ],
    )

    assert result["ok"] is True
    assert state["active_task_id"] == "task-1"
    assert state["task_plan"][0]["status"] == "in_progress"


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

    def fake_run_search_agent(
        state_arg: FishState,
        instruction: str,
        *,
        writer=None,
        active_task=None,
    ) -> dict[str, Any]:
        assert state_arg is state
        assert instruction == "lookup"
        return {
            "summary": "new note",
            "sources": [
                {"url": "https://example.com/a", "title": "duplicate"},
                {"url": "https://example.com/b", "title": "new"},
            ],
            "research_artifacts": [],
            "research_status": "incomplete",
            "confidence": 0.6,
            "assessment_summary": "need roster follow-up",
            "open_questions": ["which rosters are published"],
            "evidence_gaps": ["official rosters missing"],
            "next_queries": ["2026 World Cup preliminary squads"],
            "proposed_tasks": [
                {
                    "title": "补充初步名单来源",
                    "agent": "searchAgent",
                    "instruction": "继续查询 2026 World Cup preliminary squads 的官方来源。",
                    "reason": "当前研究缺少官方名单来源。",
                }
            ],
            "stop_reason": "incomplete_available",
        }

    monkeypatch.setattr(planner_tools, "run_search_agent", fake_run_search_agent)

    result = _call_search_agent(state, events.append, "lookup", state_updates)

    assert result["ok"] is True
    assert state["search_notes"] == "old note\n\nnew note"
    assert [source["url"] for source in state["sources"]] == ["https://example.com/a", "https://example.com/b"]
    assert state["research_artifacts"] == []
    assert state["latest_research_assessment"]["status"] == "incomplete"
    assert state["latest_research_assessment"]["task_id"] == ""
    assert state["latest_research_assessment"]["summary"] == "need roster follow-up"
    assert state["latest_research_assessment"]["artifact_paths"] == []
    assert state["latest_research_batch"]["status"] == "incomplete"
    assert state["latest_research_batch"]["summary"] == "need roster follow-up"
    assert state["latest_research_batch"]["open_questions"] == ["which rosters are published"]
    assert state["latest_research_batch"]["source_refs"][0]["url"] == "https://example.com/a"
    assert state["research_batches"][0]["batch_id"].endswith("-001")
    assert state["research_assessments"][0]["next_queries"] == ["2026 World Cup preliminary squads"]
    assert state["latest_research_assessment"]["proposed_tasks"][0]["agent"] == "searchAgent"
    assert state["latest_research_batch"]["proposed_tasks"][0]["instruction"].startswith("继续查询")
    assert result["proposed_tasks"][0]["title"] == "补充初步名单来源"
    assert [handoff["instruction"] for handoff in state["handoffs"]] == ["old", "lookup"]
    assert state_updates["search_notes"] == "new note"
    assert [source["url"] for source in state_updates["sources"]] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert state_updates["research_artifacts"] == []
    assert state_updates["latest_research_batch"]["status"] == "incomplete"
    assert state_updates["research_batches"][0]["summary"] == "need roster follow-up"
    assert state_updates["latest_research_assessment"]["status"] == "incomplete"
    assert state_updates["research_assessments"][0]["summary"] == "need roster follow-up"
    assert state_updates["latest_research_assessment"]["proposed_tasks"][0]["agent"] == "searchAgent"
    assert [handoff["instruction"] for handoff in state_updates["handoffs"]] == ["lookup"]
    assert [event["type"] for event in events] == ["handoff", "handoff_result"]
    assert "code_dirty" not in state_updates


def test_search_agent_tool_updates_task_plan_status(tmp_path: Path, monkeypatch) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "demo",
        "task_plan": [
            {
                "id": "research-qualified-teams",
                "title": "查询已晋级球队",
                "status": "pending",
                "agent": "searchAgent",
                "instruction": "查询已确认出战国家并保存来源。",
            }
        ],
    }
    state_updates: FishState = {}

    def fake_run_search_agent(
        state_arg: FishState,
        instruction: str,
        *,
        writer=None,
        active_task=None,
    ) -> dict[str, Any]:
        assert instruction == "lookup teams"
        return {"summary": "teams saved", "sources": [], "research_artifacts": []}

    monkeypatch.setattr(planner_tools, "run_search_agent", fake_run_search_agent)

    result = _call_search_agent(
        state,
        lambda _: None,
        "lookup teams",
        state_updates,
        task_id="research-qualified-teams",
    )

    assert result["ok"] is True
    assert state["active_task_id"] == ""
    assert state["task_plan"][0]["status"] == "completed"
    assert state["task_plan"][0]["result"] == "teams saved"
    assert state["latest_research_assessment"]["task_id"] == "research-qualified-teams"
    assert state_updates["task_plan"][0]["id"] == "research-qualified-teams"
    assert state_updates["task_plan"][0]["status"] == "completed"


def test_search_agent_tool_blocks_task_when_research_has_gaps(tmp_path: Path, monkeypatch) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "demo",
        "task_plan": [
            {
                "id": "task-1",
                "title": "Research",
                "status": "pending",
                "agent": "searchAgent",
                "instruction": "research data",
            }
        ],
    }
    state_updates: FishState = {}

    def fake_run_search_agent(
        state_arg: FishState,
        instruction: str,
        *,
        writer=None,
        active_task=None,
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "summary": "incomplete research",
            "sources": [],
            "research_artifacts": [],
            "research_status": "complete",
            "open_questions": ["missing official source"],
            "evidence_gaps": [],
        }

    monkeypatch.setattr(planner_tools, "run_search_agent", fake_run_search_agent)

    result = _call_search_agent(state, lambda _: None, "research data", state_updates, task_id="task-1")

    assert result["ok"] is True
    assert state["task_plan"][0]["status"] == "blocked"
    assert state["active_task_id"] == ""
    assert state_updates["task_plan"][0]["status"] == "blocked"



def test_write_task_plan_allows_one_blocked_split(tmp_path: Path) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "demo",
        "task_plan": [
            {
                "id": "task-1",
                "title": "Research",
                "status": "blocked",
                "agent": "searchAgent",
                "instruction": "research broad data",
                "result": "missing details",
            }
        ],
        "latest_research_assessment": {"task_id": "task-1", "status": "incomplete"},
    }

    result = planner_tools._write_task_plan(
        state,
        lambda _: None,
        [
            {
                "id": "task-1-a",
                "title": "Research detail",
                "agent": "searchAgent",
                "instruction": "research the missing detail",
            }
        ],
    )

    assert result["ok"] is True
    assert result["changed"] is True
    tasks = {item["id"]: item for item in state["task_plan"]}
    assert tasks["task-1"]["blocked_split_count"] == 1
    assert tasks["task-1"]["status"] == "blocked"
    assert tasks["task-1-a"]["parent_task_id"] == "task-1"
    assert tasks["task-1-a"]["blocked_split_depth"] == 1
    assert tasks["task-1-a"]["status"] == "pending"


def test_write_task_plan_rejects_second_split_for_same_blocked_task(tmp_path: Path) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "demo",
        "task_plan": [
            {
                "id": "task-1",
                "title": "Research",
                "status": "blocked",
                "agent": "searchAgent",
                "instruction": "research broad data",
                "blocked_split_count": 1,
            },
            {
                "id": "task-2",
                "title": "Next task",
                "status": "pending",
                "agent": "codeAgent",
                "instruction": "write the available notes",
            },
        ],
        "latest_research_assessment": {"task_id": "task-1", "status": "incomplete"},
    }
    state_updates: FishState = {}

    result = planner_tools._write_task_plan(
        state,
        lambda _: None,
        [
            {
                "id": "task-1-b",
                "title": "Another split",
                "agent": "searchAgent",
                "instruction": "research another detail",
            }
        ],
        state_updates=state_updates,
    )

    assert result["ok"] is True
    assert result["changed"] is False
    assert result["next_task_id"] == "task-2"
    assert "split limit" in result["message"]
    assert [item["id"] for item in state["task_plan"]] == ["task-1", "task-2"]
    assert state_updates == {}


def test_write_task_plan_rejects_split_for_blocked_child_task(tmp_path: Path) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "demo",
        "task_plan": [
            {
                "id": "task-1-a",
                "title": "Research detail",
                "status": "blocked",
                "agent": "searchAgent",
                "instruction": "research the missing detail",
                "parent_task_id": "task-1",
                "blocked_split_depth": 1,
            },
            {
                "id": "task-2",
                "title": "Next task",
                "status": "pending",
                "agent": "codeAgent",
                "instruction": "write the available notes",
            },
        ],
        "latest_research_assessment": {"task_id": "task-1-a", "status": "incomplete"},
    }

    result = planner_tools._write_task_plan(
        state,
        lambda _: None,
        [
            {
                "id": "task-1-a-1",
                "title": "Too deep",
                "agent": "searchAgent",
                "instruction": "split again",
            }
        ],
    )

    assert result["ok"] is True
    assert result["changed"] is False
    assert result["next_task_id"] == "task-2"
    assert [item["id"] for item in state["task_plan"]] == ["task-1-a", "task-2"]


def test_search_child_task_marks_manual_required_after_blocked_again(tmp_path: Path, monkeypatch) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "demo",
        "task_plan": [
            {
                "id": "task-1-a",
                "title": "Research detail",
                "status": "pending",
                "agent": "searchAgent",
                "instruction": "research detail",
                "parent_task_id": "task-1",
                "blocked_split_depth": 1,
            }
        ],
    }

    def fake_run_search_agent(
        state_arg: FishState,
        instruction: str,
        *,
        writer=None,
        active_task=None,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "summary": "still blocked",
            "sources": [],
            "research_artifacts": [],
            "research_status": "incomplete",
            "open_questions": ["still missing"],
            "evidence_gaps": [],
        }

    monkeypatch.setattr(planner_tools, "run_search_agent", fake_run_search_agent)

    result = _call_search_agent(state, lambda _: None, "", {}, task_id="task-1-a")

    assert result["ok"] is False
    task = state["task_plan"][0]
    assert task["status"] == "blocked"
    assert task["manual_required"] is True
    assert task["skip_reason"] == "blocked_after_split_limit"
    assert task["blocked_split_depth"] == 1
def test_update_task_status_rejects_manual_in_progress(tmp_path: Path) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "demo",
        "active_task_id": "",
        "task_plan": [
            {
                "id": "task-2",
                "title": "Research rosters",
                "status": "pending",
                "agent": "searchAgent",
                "instruction": "research rosters",
            }
        ],
    }

    result = planner_tools._update_task_status(state, lambda _: None, "task-2", "in_progress")

    assert result["ok"] is False
    assert "AgentTool" in result["error"]
    assert state["active_task_id"] == ""
    assert state["task_plan"][0]["status"] == "pending"


def test_code_agent_tool_records_reducer_delta(tmp_path: Path, monkeypatch) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "demo",
        "code_summary": "old summary",
        "handoffs": [],
    }
    state_updates: FishState = {}

    def fake_run_code_agent(
        state_arg: FishState,
        instruction: str,
        *,
        writer=None,
        active_task=None,
    ) -> dict[str, Any]:
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
    assert state["code_dirty"] is True
    assert state["verified"] is False
    assert state["verification_status"] == "needs_verification"
    assert state_updates["code_dirty"] is True
    assert state_updates["verified"] is False
    assert state_updates["verification_status"] == "needs_verification"


def test_code_agent_tool_document_output_does_not_mark_code_dirty(tmp_path: Path, monkeypatch) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "demo",
        "code_summary": "",
        "handoffs": [],
        "code_dirty": False,
        "verified": False,
        "verification_status": "not_started",
    }
    state_updates: FishState = {}

    def fake_run_code_agent(
        state_arg: FishState,
        instruction: str,
        *,
        writer=None,
        active_task=None,
    ) -> dict[str, Any]:
        return {
            "summary": "wrote research notes",
            "tool_events": [
                {
                    "name": "FileWriteTool",
                    "result": {"ok": True, "path": "research/world-cup.md", "created": True},
                }
            ],
        }

    monkeypatch.setattr(planner_tools, "run_code_agent", fake_run_code_agent)

    result = _call_code_agent(state, lambda _: None, "整理资料并写入 Markdown 报告", state_updates)

    assert result == {"ok": True, "summary": "wrote research notes"}
    assert state["code_summary"] == "wrote research notes"
    assert state["code_dirty"] is False
    assert state["verified"] is False
    assert state["verification_status"] == "not_started"
    assert "code_dirty" not in state_updates
    assert "verified" not in state_updates
    assert "verification_status" not in state_updates


def test_code_agent_tool_does_not_reuse_active_search_task(tmp_path: Path, monkeypatch) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "demo",
        "active_task_id": "task-2",
        "task_plan": [
            {
                "id": "task-2",
                "title": "Search data",
                "status": "in_progress",
                "agent": "searchAgent",
                "instruction": "search only",
            },
            {
                "id": "task-3",
                "title": "Write code",
                "status": "pending",
                "agent": "codeAgent",
                "instruction": "implement from research",
            },
        ],
        "code_summary": "",
        "handoffs": [],
    }

    def fake_run_code_agent(
        state_arg: FishState,
        instruction: str,
        *,
        writer=None,
        active_task=None,
    ) -> dict[str, Any]:
        assert active_task["id"] == "task-3"
        assert instruction == "implement from research"
        return {"summary": "implemented task-3"}

    monkeypatch.setattr(planner_tools, "run_code_agent", fake_run_code_agent)

    result = _call_code_agent(state, lambda _: None, "", {})

    assert result == {"ok": True, "summary": "implemented task-3"}
    statuses = {item["id"]: item["status"] for item in state["task_plan"]}
    assert statuses["task-2"] == "in_progress"
    assert statuses["task-3"] == "completed"


def test_code_agent_tool_rejects_explicit_search_task(tmp_path: Path, monkeypatch) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "demo",
        "task_plan": [
            {
                "id": "task-1",
                "title": "Search data",
                "status": "in_progress",
                "agent": "searchAgent",
                "instruction": "search only",
            }
        ],
        "code_summary": "",
        "handoffs": [],
    }
    events: list[dict[str, Any]] = []

    def fake_run_code_agent(*args, **kwargs) -> dict[str, Any]:
        raise AssertionError("CodeAgent should not run for a searchAgent task")

    monkeypatch.setattr(planner_tools, "run_code_agent", fake_run_code_agent)

    result = _call_code_agent(state, events.append, "read reports", {}, task_id="task-1")

    assert result["ok"] is False
    assert "No codeAgent task matches" in result["error"]
    assert events[-1]["type"] == "handoff_blocked"
    assert state["task_plan"][0]["status"] == "in_progress"
    assert state["code_summary"] == ""


def test_review_agent_tool_records_review_delta(tmp_path: Path, monkeypatch) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "demo",
        "review_notes": "old review",
        "handoffs": [],
    }
    state_updates: FishState = {}

    def fake_run_review_agent(
        state_arg: FishState,
        instruction: str,
        *,
        writer=None,
        active_task=None,
    ) -> dict[str, Any]:
        assert state_arg is state
        assert instruction == "review"
        return {"summary": "new review", "passed": True}

    monkeypatch.setattr(planner_tools, "run_review_agent", fake_run_review_agent)

    result = planner_tools._call_review_agent(state, lambda _: None, "review", state_updates)

    assert result["ok"] is True
    assert result["passed"] is True
    assert state["review_notes"] == "old review\n\nnew review"
    assert state["verification_status"] == "review_passed"
    assert state_updates["review_notes"] == "new review"
    assert state_updates["verification_status"] == "review_passed"

def test_test_agent_tool_marks_verified_when_passed(tmp_path: Path, monkeypatch) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "demo",
        "test_notes": "",
        "handoffs": [],
        "code_dirty": True,
        "verified": False,
    }
    state_updates: FishState = {}

    def fake_run_test_agent(
        state_arg: FishState,
        instruction: str,
        *,
        writer=None,
        active_task=None,
    ) -> dict[str, Any]:
        assert state_arg is state
        assert instruction == "test"
        return {"summary": "tests passed", "passed": True, "commands": ["python -m pytest"]}

    monkeypatch.setattr(planner_tools, "run_test_agent", fake_run_test_agent)

    result = planner_tools._call_test_agent(state, lambda _: None, "test", state_updates)

    assert result["passed"] is True
    assert state["test_notes"] == "tests passed"
    assert state["verified"] is True
    assert state["code_dirty"] is False
    assert state["verification_status"] == "passed"
    assert state_updates["verified"] is True
    assert state_updates["code_dirty"] is False

def test_test_agent_tool_keeps_dirty_when_failed(tmp_path: Path, monkeypatch) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "demo",
        "test_notes": "",
        "handoffs": [],
        "code_dirty": True,
        "verified": False,
    }
    state_updates: FishState = {}

    def fake_run_test_agent(
        state_arg: FishState,
        instruction: str,
        *,
        writer=None,
        active_task=None,
    ) -> dict[str, Any]:
        return {"summary": "tests failed", "passed": False, "commands": ["python -m pytest"]}

    monkeypatch.setattr(planner_tools, "run_test_agent", fake_run_test_agent)

    result = planner_tools._call_test_agent(state, lambda _: None, "test", state_updates)

    assert result["passed"] is False
    assert state["verified"] is False
    assert state["code_dirty"] is True
    assert state["verification_status"] == "failed"

def test_build_planner_tools_exposes_review_and_test(tmp_path: Path) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {"runtime": runtime, "task": "demo"}
    tools = planner_tools.build_planner_tools(state, lambda _: None)
    names = {tool.name for tool in tools}

    assert "WriteTaskPlanTool" in names
    assert "UpdateTaskStatusTool" in names
    assert "SearchAgentTool" in names
    assert "CodeAgentTool" in names
    assert "ReviewAgentTool" in names
    assert "TestAgentTool" in names
