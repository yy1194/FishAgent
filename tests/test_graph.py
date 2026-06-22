from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, message_to_dict
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from fishclaw.graph.nodes import context_compressor_node, context_route, final_node
from fishclaw.graph.state_io import _merge_update, _message_text, _restore_saved_state
from fishclaw.state import FishRuntime, FishState


def test_context_route_goes_final_when_done(tmp_path: Path) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {"runtime": runtime, "done": True}

    assert context_route(state) == "final"


def test_context_route_goes_final_when_max_rounds_reached(tmp_path: Path) -> None:
    runtime = FishRuntime(workspace=tmp_path, max_planner_rounds=3)
    state: FishState = {"runtime": runtime, "planner_rounds": 3}

    assert context_route(state) == "final"


def test_context_route_goes_compressor_when_requested(tmp_path: Path) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "done": False,
        "planner_rounds": 1,
        "should_compress": True,
    }

    assert context_route(state) == "context_compressor"


def test_context_route_goes_planner_by_default(tmp_path: Path) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "done": False,
        "planner_rounds": 1,
        "should_compress": False,
    }

    assert context_route(state) == "planner"


def test_merge_update_uses_reducers_and_overwrites_plain_fields() -> None:
    state: FishState = {
        "messages": [HumanMessage(content="old")],
        "planner_rounds": 1,
        "search_notes": "old note",
        "sources": [{"url": "https://example.com/a", "title": "old"}],
        "research_artifacts": [{"path": "research/old.md", "title": "old"}],
        "research_batches": [{"batch_id": "batch-a", "status": "incomplete"}],
        "latest_research_batch": {"batch_id": "batch-a", "status": "incomplete"},
        "research_assessments": [{"task_id": "research-a", "status": "incomplete"}],
        "latest_research_assessment": {"task_id": "research-a", "status": "incomplete"},
        "task_plan": [{"id": "research-a", "title": "Research A", "status": "pending"}],
        "handoffs": [{"to": "searchAgent", "instruction": "old", "summary": "old"}],
        "metadata": {"existing": True},
    }
    event = {
        "planner": {
            "messages": [AIMessage(content="new")],
            "planner_rounds": 2,
            "search_notes": "new note",
            "sources": [
                {"url": "https://example.com/a", "title": "duplicate"},
                {"url": "https://example.com/b", "title": "new"},
            ],
            "research_artifacts": [{"path": "research/new.md", "title": "new"}],
            "research_batches": [{"batch_id": "batch-b", "status": "complete"}],
            "latest_research_batch": {"batch_id": "batch-b", "status": "complete"},
            "research_assessments": [{"task_id": "research-b", "status": "complete"}],
            "latest_research_assessment": {"task_id": "research-b", "status": "complete"},
            "task_plan": [
                {"id": "research-a", "status": "completed", "result": "done"},
                {"id": "research-b", "title": "Research B", "status": "pending"},
            ],
            "handoffs": [{"to": "codeAgent", "instruction": "new", "summary": "new"}],
            "metadata": {"last_planner_response": "ok"},
        }
    }

    _merge_update(state, event)

    assert [message.content for message in state["messages"]] == ["old", "new"]
    assert state["planner_rounds"] == 2
    assert state["search_notes"] == "old note\n\nnew note"
    assert [source["url"] for source in state["sources"]] == ["https://example.com/a", "https://example.com/b"]
    assert [artifact["path"] for artifact in state["research_artifacts"]] == ["research/old.md", "research/new.md"]
    assert [batch["batch_id"] for batch in state["research_batches"]] == ["batch-a", "batch-b"]
    assert state["latest_research_batch"] == {"batch_id": "batch-b", "status": "complete"}
    assert [assessment["task_id"] for assessment in state["research_assessments"]] == ["research-a", "research-b"]
    assert state["latest_research_assessment"] == {"task_id": "research-b", "status": "complete"}
    assert [task["id"] for task in state["task_plan"]] == ["research-a", "research-b"]
    assert state["task_plan"][0]["status"] == "completed"
    assert state["task_plan"][0]["title"] == "Research A"
    assert state["task_plan"][0]["result"] == "done"
    assert [handoff["instruction"] for handoff in state["handoffs"]] == ["old", "new"]
    assert state["metadata"] == {"existing": True, "last_planner_response": "ok"}


def test_merge_update_can_remove_all_messages() -> None:
    state: FishState = {
        "messages": [HumanMessage(content="old"), AIMessage(content="older")],
    }
    event = {
        "context_compressor": {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                AIMessage(content="summary"),
            ],
        }
    }

    _merge_update(state, event)

    assert len(state["messages"]) == 1
    assert state["messages"][0].content == "summary"


def test_restore_saved_state_rehydrates_messages(tmp_path: Path) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    payload = {
        "task": "demo",
        "status": "running",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "messages": [message_to_dict(AIMessage(content="hello"))],
    }

    state = _restore_saved_state(payload, runtime)

    assert state["runtime"] is runtime
    assert state["task"] == "demo"
    assert state["messages"][0].content == "hello"
    assert "status" not in state
    assert "updated_at" not in state
    assert state["sources"] == []
    assert state["research_artifacts"] == []
    assert state["research_batches"] == []
    assert state["latest_research_batch"] == {}
    assert state["research_assessments"] == []
    assert state["latest_research_assessment"] == {}
    assert state["task_plan"] == []
    assert state["active_task_id"] == ""
    assert state["plan_summary"] == ""
    assert state["handoffs"] == []
    assert state["search_notes"] == ""
    assert state["code_summary"] == ""
    assert state["metadata"] == {}
    assert state["errors"] == []
    assert state["review_notes"] == ""
    assert state["test_notes"] == ""
    assert state["verification_status"] == "not_started"
    assert state["verified"] is False
    assert state["code_dirty"] is False


def test_final_node_fallback_includes_task_plan(tmp_path: Path) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "final_answer": "",
        "task_plan": [
            {
                "id": "research-qualified-teams",
                "title": "查询已确认出战国家",
                "status": "blocked",
                "instruction": "查询并保存来源。",
                "result": "还缺少部分来源。",
            }
        ],
        "code_summary": "",
        "search_notes": "已完成部分查询。",
    }

    result = final_node(state)

    assert result["done"] is True
    assert "research-qualified-teams [blocked]" in result["final_answer"]
    assert "还缺少部分来源" in result["final_answer"]


def test_message_text_serializes_non_string_content() -> None:
    message = AIMessage(content=[{"type": "text", "text": "hello"}])

    assert "hello" in _message_text(message)


def test_context_compressor_falls_back_when_model_fails(tmp_path: Path, monkeypatch) -> None:
    runtime = FishRuntime(workspace=tmp_path)

    def raise_model_error(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("fishclaw.graph.nodes.create_model", raise_model_error)
    state: FishState = {
        "runtime": runtime,
        "task": "demo",
        "messages": [HumanMessage(content="old context")],
        "sources": [],
        "handoffs": [],
    }

    result = context_compressor_node(state)

    assert result["since_compression"] == 0
    assert result["should_compress"] is False
    assert "压缩模型不可用" in result["context_summary"]
    assert (runtime.workspace / ".fishclaw" / "HISTORY.md").exists()
    assert isinstance(result["messages"][0], RemoveMessage)
    assert isinstance(result["messages"][1], AIMessage)
