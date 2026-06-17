"""Fishclaw 的 Harness Engineering：受控文件、搜索、命令工具。"""

from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool

from fishclaw.state import FishRuntime


DANGEROUS_COMMANDS = [
    r"\brm\s+-rf\b",
    r"\bRemove-Item\b.*\b-Recurse\b.*\b-Force\b",
    r"\bdel\s+/[sq]\b",
    r"\bformat\b",
    r"\bshutdown\b",
    r"\breboot\b",
]


def read_text_lossy(path: Path) -> str:
    """按常见编码读取文本，失败时用替换字符兜底。"""
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def resolve_path(runtime: FishRuntime, file_path: str) -> Path:
    """把模型传入的路径解析为 workspace 内路径。"""
    raw = file_path.replace("\\", "/").strip()
    if raw.startswith("workspace/"):
        raw = raw[len("workspace/") :]
    path = Path(raw)
    if not path.is_absolute():
        path = runtime.workspace / path
    return runtime.assert_workspace_path(path)


def display_path(runtime: FishRuntime, path: Path) -> str:
    """优先用 workspace 相对路径展示工具结果。"""
    try:
        return str(path.resolve().relative_to(runtime.workspace.resolve()))
    except ValueError:
        return str(path)


def file_read(runtime: FishRuntime, file_path: str, offset: int | str = 0, limit: int | str = 400) -> dict[str, Any]:
    """读取 workspace 文本文件，并返回带行号片段。"""
    try:
        path = resolve_path(runtime, file_path)
        offset_value = int(offset)
        limit_value = max(1, min(int(limit), 1200))
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}
    if not path.exists() or not path.is_file():
        return {"ok": False, "error": f"file does not exist: {display_path(runtime, path)}"}
    text = read_text_lossy(path)
    lines = text.splitlines()
    selected = lines[offset_value : offset_value + limit_value]
    runtime.record_read(path)
    content = "\n".join(f"{offset_value + index + 1}: {line}" for index, line in enumerate(selected))
    return {"ok": True, "path": display_path(runtime, path), "total_lines": len(lines), "content": content}


def file_write(runtime: FishRuntime, file_path: str, content: str) -> dict[str, Any]:
    """创建或重写文件；已有文件必须先读过且未被外部修改。"""
    try:
        path = resolve_path(runtime, file_path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    existed = path.exists()
    original = read_text_lossy(path) if existed else ""
    if existed:
        snapshot = runtime.read_snapshots.get(path.resolve())
        if snapshot is None:
            return {"ok": False, "error": "existing file must be read before writing"}
        if path.stat().st_mtime_ns != snapshot:
            return {"ok": False, "error": "file changed after read; read it again before writing"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    runtime.record_read(path)
    diff = "\n".join(
        difflib.unified_diff(
            original.splitlines(),
            content.splitlines(),
            fromfile=f"a/{display_path(runtime, path)}",
            tofile=f"b/{display_path(runtime, path)}",
            lineterm="",
        )
    )
    return {"ok": True, "path": display_path(runtime, path), "created": not existed, "diff": diff[:4000]}


def grep(runtime: FishRuntime, pattern: str, path: str = ".", head_limit: int | str = 50) -> dict[str, Any]:
    """在 workspace 内用正则搜索文本。"""
    if not pattern:
        return {"ok": False, "error": "pattern must not be empty"}
    try:
        root = resolve_path(runtime, path)
        limit = max(1, int(head_limit))
        regex = re.compile(pattern)
    except (TypeError, ValueError, re.error) as exc:
        return {"ok": False, "error": str(exc)}
    candidates = [root] if root.is_file() else [item for item in root.rglob("*") if item.is_file()]
    matches: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(part in {".git", ".fishclaw", ".venv", "__pycache__", "node_modules"} for part in candidate.parts):
            continue
        for line_no, line in enumerate(read_text_lossy(candidate).splitlines(), start=1):
            if regex.search(line):
                matches.append({"path": display_path(runtime, candidate), "line": line_no, "text": line})
                if len(matches) >= limit:
                    return {"ok": True, "matches": matches, "truncated": True}
    return {"ok": True, "matches": matches, "truncated": False}


def run_bash(runtime: FishRuntime, command: str, timeout_seconds: int | str | None = None) -> dict[str, Any]:
    """在 workspace 内运行非交互命令，并拦截明显危险操作。"""
    command = command.strip()
    if not command:
        return {"ok": False, "error": "command must not be empty"}
    for pattern in DANGEROUS_COMMANDS:
        if re.search(pattern, command, flags=re.IGNORECASE):
            return {"ok": False, "error": f"blocked dangerous command pattern: {pattern}"}
    timeout = runtime.bash_timeout_seconds if timeout_seconds is None else int(timeout_seconds)
    started = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        completed = subprocess.run(command, cwd=runtime.workspace, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "timed_out": True, "stdout": exc.stdout or "", "stderr": exc.stderr or "", "exit_code": None}
    return {
        "ok": completed.returncode == 0,
        "timed_out": False,
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout[: runtime.max_output_chars],
        "stderr": completed.stderr[: runtime.max_output_chars],
        "duration_ms": round((time.perf_counter() - started) * 1000),
    }


def web_search(query: str, max_results: int | str = 5) -> dict[str, Any]:
    """使用 Tavily 搜索网页，并返回精简来源列表。"""
    load_dotenv()
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"ok": False, "error": "missing required .env setting: TAVILY_API_KEY"}
    try:
        from tavily import TavilyClient
    except ImportError as exc:
        return {"ok": False, "error": f"tavily-python is not installed: {exc}"}
    try:
        limit = max(1, min(int(max_results), 10))
        response = TavilyClient(api_key=api_key).search(query=query, search_depth="basic", max_results=limit, include_answer=True)
    except Exception as exc:
        return {"ok": False, "query": query, "error": f"{type(exc).__name__}: {exc}"}
    results = [
        {
            "title": str(item.get("title", "")),
            "url": str(item.get("url", "")),
            "content": str(item.get("content", ""))[:1000],
            "score": item.get("score"),
        }
        for item in response.get("results", []) or []
    ]
    return {"ok": True, "query": query, "answer": response.get("answer") or "", "results": results}


def build_code_tools(runtime: FishRuntime) -> list[StructuredTool]:
    """构建 CodeAgent 可用的 workspace harness 工具。"""
    return [
        StructuredTool.from_function(
            name="FileReadTool",
            func=lambda file_path, offset=0, limit=400: file_read(runtime, file_path, offset, limit),
            description="读取 workspace 内文本文件。参数：file_path, offset, limit。",
        ),
        StructuredTool.from_function(
            name="FileWriteTool",
            func=lambda file_path, content: file_write(runtime, file_path, content),
            description="创建或重写 workspace 内文件；已有文件必须先读。参数：file_path, content。",
        ),
        StructuredTool.from_function(
            name="GrepTool",
            func=lambda pattern, path=".", head_limit=50: grep(runtime, pattern, path, head_limit),
            description="在 workspace 内正则搜索。参数：pattern, path, head_limit。",
        ),
        StructuredTool.from_function(
            name="BashTool",
            func=lambda command, timeout_seconds=None: run_bash(runtime, command, timeout_seconds),
            description="在 workspace 内运行非交互命令。参数：command, timeout_seconds。",
        ),
    ]


def build_search_tool() -> StructuredTool:
    """构建 SearchAgent 使用的 WebSearchTool。"""
    return StructuredTool.from_function(
        name="WebSearchTool",
        func=web_search,
        description="使用 Tavily 搜索网页。参数：query, max_results。",
    )


def tool_result_json(result: Any) -> str:
    """把工具结果编码成 ToolMessage 可用的 JSON 字符串。"""
    return json.dumps(result, ensure_ascii=False, default=str)

