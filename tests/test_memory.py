from __future__ import annotations

from pathlib import Path

from fishclaw.memory import build_agent_memory
from fishclaw.state import FishRuntime, FishState


def test_build_agent_memory_scopes_to_current_task(tmp_path: Path) -> None:
    runtime = FishRuntime(workspace=tmp_path)
    state: FishState = {
        "runtime": runtime,
        "task": "overall goal",
        "active_task_id": "task-2",
        "task_plan": [
            {
                "id": "task-1",
                "title": "Search",
                "status": "completed",
                "agent": "searchAgent",
                "instruction": "search data",
            },
            {
                "id": "task-2",
                "title": "Implement",
                "status": "in_progress",
                "agent": "codeAgent",
                "instruction": "write code",
            },
        ],
        "search_notes": "upstream research",
        "sources": [{"url": "https://example.com/source", "content": "large raw source text"}],
        "latest_research_batch": {
            "batch_id": "task-1-001",
            "task_id": "task-1",
            "status": "incomplete",
            "confidence": 0.7,
            "summary": "country list done; rosters missing",
            "answered": ["qualified country list"],
            "entities": [
                {
                    "name": "France",
                    "type": "team",
                    "status": "partial",
                    "facts": ["qualified for 2026 World Cup"],
                    "source_urls": ["https://example.com/source"],
                }
            ],
            "open_questions": ["which rosters are published"],
            "evidence_gaps": ["official roster source missing"],
            "next_queries": ["2026 World Cup squads"],
            "source_refs": [
                {
                    "title": "Source",
                    "url": "https://example.com/source",
                    "content_preview": "large raw source text",
                }
            ],
            "search_stats": {"search_calls": 1, "blocked_tool_calls": 0, "source_count": 1},
        },
        "research_batches": [
            {
                "batch_id": "task-1-001",
                "task_id": "task-1",
                "status": "incomplete",
                "summary": "country list done; rosters missing",
            }
        ],
        "latest_research_assessment": {
            "task_id": "task-1",
            "status": "incomplete",
            "summary": "country list done; rosters missing",
            "open_questions": ["which rosters are published"],
            "evidence_gaps": ["official roster source missing"],
            "next_queries": ["2026 World Cup squads"],
            "artifact_paths": ["research/world-cup.md"],
        },
        "research_assessments": [
            {
                "task_id": "task-1",
                "status": "incomplete",
                "summary": "country list done; rosters missing",
            }
        ],
    }

    memory = build_agent_memory(state, instruction="write code")

    assert "task_plan" not in memory
    assert memory["planner_instruction"] == "write code"
    assert memory["current_task"]["id"] == "task-2"
    assert memory["current_task"]["agent"] == "codeAgent"
    assert memory["search_notes"] == ""
    assert memory["sources"] == []
    assert "country list done; rosters missing" in memory["research_context"]
    assert memory["latest_research_batch"]["batch_id"] == "task-1-001"
    assert memory["latest_research_batch"]["answered"] == ["qualified country list"]
    assert memory["latest_research_batch"]["entities"][0]["name"] == "France"
    assert memory["research_batches"][0]["batch_id"] == "task-1-001"
    assert memory["latest_research_assessment"]["task_id"] == "task-1"
    assert memory["latest_research_assessment"]["next_queries"] == ["2026 World Cup squads"]
    assert memory["research_assessments"][0]["summary"] == "country list done; rosters missing"
