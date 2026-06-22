from __future__ import annotations

from fishclaw.agents.common import build_agent_summary
from fishclaw.agents.test_agent import _test_result_summary


def test_build_agent_summary_falls_back_to_tool_events() -> None:
    summary = build_agent_summary(
        "CodeAgent",
        "",
        [{"name": "PatchTool", "result": {"ok": True, "path": "src/example.py"}}],
    )

    assert "CodeAgent" not in summary
    assert "工具调用 1 次" in summary
    assert "PatchTool(ok" in summary
    assert "src/example.py" in summary


def test_test_result_summary_includes_commands_and_status() -> None:
    summary = _test_result_summary(
        "all good",
        True,
        ["python -m pytest"],
        True,
        False,
        [],
        False,
    )

    assert "验证通过" in summary
    assert "python -m pytest" in summary
    assert "模型总结：all good" in summary
