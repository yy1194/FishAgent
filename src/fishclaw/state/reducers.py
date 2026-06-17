"""Reducer helpers for Fishclaw graph state."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any, Callable

from langgraph.graph import add_messages


StateReducer = Callable[[Any, Any], Any]


def merge_messages(left: Any, right: Any) -> list[Any]:
    """Merge LangChain messages with LangGraph's message reducer semantics."""
    return list(add_messages(left or [], right or []))


def append_text(left: Any, right: Any) -> str:
    """Append non-empty text fragments separated by a blank line."""
    parts: list[str] = []
    for value in (left, right):
        if value is None:
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def append_items(left: Any, right: Any) -> list[Any]:
    """Append list-like values while accepting a single item as an update."""
    left_items = left if isinstance(left, list) else ([] if left is None else [left])
    right_items = right if isinstance(right, list) else ([] if right is None else [right])
    return [*left_items, *right_items]


def merge_sources(left: Any, right: Any) -> list[dict[str, Any]]:
    """Merge source records, keeping the first item for each URL."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in (left or [], right or []):
        sources = group if isinstance(group, list) else [group]
        for source in sources:
            if not isinstance(source, dict):
                continue
            url = str(source.get("url", "")).strip()
            if not url or url in seen:
                continue
            seen.add(url)
            merged.append(source)
    return merged


def merge_dicts(left: Any, right: Any) -> dict[str, Any]:
    """Shallow-merge dictionaries, with update keys taking precedence."""
    merged = dict(left or {}) if isinstance(left, Mapping) else {}
    if isinstance(right, Mapping):
        merged.update(right)
    return merged


STATE_REDUCERS: dict[str, StateReducer] = {
    "messages": merge_messages,
    "search_notes": append_text,
    "sources": merge_sources,
    "code_summary": append_text,
    "handoffs": append_items,
    "metadata": merge_dicts,
    "errors": append_items,
}


def reduce_state_value(key: str, current_value: Any, update_value: Any) -> Any:
    """Apply the configured reducer for a state key, or overwrite by default."""
    reducer = STATE_REDUCERS.get(key)
    if reducer is None:
        return update_value
    return reducer(current_value, update_value)


def apply_state_update(state: MutableMapping[str, Any], update: Mapping[str, Any]) -> None:
    """Merge a state update into an existing state mapping."""
    for key, value in update.items():
        state[key] = reduce_state_value(key, state.get(key), value)
