from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from fishclaw.state import FishRuntime
from fishclaw.tools.harness import (
    build_code_tools,
    file_read,
    file_write,
    grep,
    list_files,
    patch_file,
    resolve_path,
    run_bash,
)


@pytest.fixture()
def runtime(tmp_path: Path) -> FishRuntime:
    return FishRuntime(workspace=tmp_path)


def _entry_paths(result: dict) -> set[str]:
    return {str(entry["path"]).replace("\\", "/") for entry in result["entries"]}


def test_resolve_path_stays_inside_workspace(runtime: FishRuntime) -> None:
    path = resolve_path(runtime, "workspace/demo.txt")

    assert path == runtime.workspace / "demo.txt"


def test_resolve_path_rejects_workspace_escape(runtime: FishRuntime) -> None:
    with pytest.raises(ValueError):
        resolve_path(runtime, "../outside.txt")


def test_file_write_creates_new_file(runtime: FishRuntime) -> None:
    result = file_write(runtime, "demo.txt", "hello")

    assert result["ok"] is True
    assert result["created"] is True
    assert (runtime.workspace / "demo.txt").read_text(encoding="utf-8") == "hello"


def test_existing_file_must_be_read_before_write(runtime: FishRuntime) -> None:
    path = runtime.workspace / "demo.txt"
    path.write_text("old", encoding="utf-8")

    result = file_write(runtime, "demo.txt", "new")

    assert result["ok"] is False
    assert "must be read" in result["error"]


def test_file_write_after_read_succeeds(runtime: FishRuntime) -> None:
    path = runtime.workspace / "demo.txt"
    path.write_text("old", encoding="utf-8")

    read_result = file_read(runtime, "demo.txt")
    write_result = file_write(runtime, "demo.txt", "new")

    assert read_result["ok"] is True
    assert write_result["ok"] is True
    assert path.read_text(encoding="utf-8") == "new"


def test_file_write_rejects_external_change_after_read(runtime: FishRuntime) -> None:
    path = runtime.workspace / "demo.txt"
    path.write_text("old", encoding="utf-8")

    read_result = file_read(runtime, "demo.txt")
    assert read_result["ok"] is True

    time.sleep(0.02)
    path.write_text("changed elsewhere", encoding="utf-8")
    os.utime(path, None)

    result = file_write(runtime, "demo.txt", "new")

    assert result["ok"] is False
    assert "changed after read" in result["error"]


def test_grep_skips_internal_directories(runtime: FishRuntime) -> None:
    (runtime.workspace / "visible.txt").write_text("needle\n", encoding="utf-8")
    internal = runtime.workspace / ".fishclaw"
    internal.mkdir()
    (internal / "hidden.txt").write_text("needle\n", encoding="utf-8")

    result = grep(runtime, "needle")

    assert result["ok"] is True
    assert result["matches"] == [{"path": "visible.txt", "line": 1, "text": "needle"}]


def test_list_files_lists_top_level_entries_and_skips_internal_dirs(runtime: FishRuntime) -> None:
    (runtime.workspace / "a.txt").write_text("x", encoding="utf-8")
    (runtime.workspace / "sub").mkdir()
    internal = runtime.workspace / ".fishclaw"
    internal.mkdir()
    (internal / "state.json").write_text("{}", encoding="utf-8")

    result = list_files(runtime)

    assert result["ok"] is True
    assert result["truncated"] is False
    assert _entry_paths(result) == {"a.txt", "sub"}


def test_list_files_recursive_lists_nested_files(runtime: FishRuntime) -> None:
    sub = runtime.workspace / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("x", encoding="utf-8")

    result = list_files(runtime, recursive=True)

    assert result["ok"] is True
    assert {"sub", "sub/nested.txt"}.issubset(_entry_paths(result))


def test_list_files_honors_max_entries(runtime: FishRuntime) -> None:
    for index in range(3):
        (runtime.workspace / f"{index}.txt").write_text("x", encoding="utf-8")

    result = list_files(runtime, max_entries=2)

    assert result["ok"] is True
    assert len(result["entries"]) == 2
    assert result["truncated"] is True


def test_patch_file_requires_read(runtime: FishRuntime) -> None:
    path = runtime.workspace / "a.txt"
    path.write_text("hello world", encoding="utf-8")

    result = patch_file(runtime, "a.txt", "world", "Fishclaw")

    assert result["ok"] is False
    assert "must be read" in result["error"]


def test_patch_file_replaces_exact_text(runtime: FishRuntime) -> None:
    path = runtime.workspace / "a.txt"
    path.write_text("hello world", encoding="utf-8")
    file_read(runtime, "a.txt")

    result = patch_file(runtime, "a.txt", "world", "Fishclaw")

    assert result["ok"] is True
    assert result["replacements"] == 1
    assert path.read_text(encoding="utf-8") == "hello Fishclaw"


def test_patch_file_rejects_ambiguous_match(runtime: FishRuntime) -> None:
    path = runtime.workspace / "a.txt"
    path.write_text("x x", encoding="utf-8")
    file_read(runtime, "a.txt")

    result = patch_file(runtime, "a.txt", "x", "y")

    assert result["ok"] is False
    assert "matched 2 time" in result["error"]


def test_patch_file_can_replace_expected_multiple_matches(runtime: FishRuntime) -> None:
    path = runtime.workspace / "a.txt"
    path.write_text("x x", encoding="utf-8")
    file_read(runtime, "a.txt")

    result = patch_file(runtime, "a.txt", "x", "y", expected_replacements=2)

    assert result["ok"] is True
    assert path.read_text(encoding="utf-8") == "y y"


def test_patch_file_rejects_external_change_after_read(runtime: FishRuntime) -> None:
    path = runtime.workspace / "a.txt"
    path.write_text("hello world", encoding="utf-8")
    file_read(runtime, "a.txt")

    time.sleep(0.02)
    path.write_text("changed elsewhere", encoding="utf-8")
    os.utime(path, None)

    result = patch_file(runtime, "a.txt", "world", "Fishclaw")

    assert result["ok"] is False
    assert "changed after read" in result["error"]


def test_run_bash_blocks_dangerous_command(runtime: FishRuntime) -> None:
    result = run_bash(runtime, "rm -rf .")

    assert result["ok"] is False
    assert "blocked dangerous command" in result["error"]


def test_build_code_tools_exposes_list_files_and_patch(runtime: FishRuntime) -> None:
    tool_names = {tool.name for tool in build_code_tools(runtime)}

    assert {"FileReadTool", "ListFilesTool", "PatchTool", "FileWriteTool", "GrepTool", "BashTool"}.issubset(tool_names)
