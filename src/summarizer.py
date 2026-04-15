"""Summarize tool requests and responses for tree rendering.

Reuses proven logic from sample_query/build_stream_tree.py, adapted
for the orchestrator's needs (compact summaries + page_context handles).
"""

from __future__ import annotations

import json
from typing import Any


def truncate(text: str, limit: int = 180) -> str:
    """Collapse whitespace and truncate to *limit* characters."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


# ---------------------------------------------------------------------------
# Request summaries
# ---------------------------------------------------------------------------

def summarize_tool_request(tool_name: str, tool_input: dict[str, Any]) -> str:
    """One-line summary of the tool request (<=140 chars)."""
    if tool_name == "lumina_search":
        return truncate(str(tool_input.get("q", "")), 140)
    if tool_name == "lumina_open":
        url = tool_input.get("url")
        if url:
            return truncate(f"url={url}", 140)
        pc = tool_input.get("page_context")
        if pc:
            return truncate(f"page_context={json.dumps(pc, separators=(',', ':'))}", 140)
        return truncate(json.dumps(tool_input, separators=(",", ":")), 140)
    if tool_name == "lumina_find":
        pattern = tool_input.get("pattern", "")
        qtype = tool_input.get("query_type", "pattern")
        return truncate(f'find "{pattern}" ({qtype})', 140)
    if tool_name == "save_page":
        return truncate(f"ref={tool_input.get('ref_key')} -> {tool_input.get('filename')}", 140)
    if tool_name == "grep_file":
        return truncate(f"grep '{tool_input.get('pattern')}' in {tool_input.get('filename')}", 140)
    if tool_name == "read_file":
        return truncate(
            f"read {tool_input.get('filename')} [{tool_input.get('offset', 0)}:{tool_input.get('limit', 100)}]",
            140,
        )
    if tool_name == "submit_answer":
        return truncate(str(tool_input.get("answer", "")), 140)
    return truncate(json.dumps(tool_input, separators=(",", ":")), 140)


# ---------------------------------------------------------------------------
# Response summaries
# ---------------------------------------------------------------------------

def _try_parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def _summarize_search_response(parsed: dict[str, Any]) -> str:
    results = parsed.get("results") or []
    if not isinstance(results, list):
        return truncate(json.dumps(parsed), 160)
    titles: list[str] = []
    for item in results[:2]:
        if isinstance(item, dict):
            title = item.get("title") or item.get("url") or "untitled"
            titles.append(str(title))
    parts = [f"{len(results)} result(s)"]
    if titles:
        parts.append("top: " + "; ".join(truncate(t, 60) for t in titles))
    errors = parsed.get("errors")
    if errors:
        parts.append("errors: " + truncate(json.dumps(errors), 80))
    return " | ".join(parts)


def _summarize_open_response(parsed: dict[str, Any]) -> str:
    pages = parsed.get("pages") or []
    if not isinstance(pages, list) or not pages:
        return truncate(json.dumps(parsed), 160)
    first = pages[0]
    if not isinstance(first, dict):
        return truncate(json.dumps(first), 160)
    title = str(first.get("title") or first.get("url") or "opened page")
    content = str(first.get("content") or "")
    total_lines = first.get("totalLines")
    summary = truncate(content, 110)
    line_text = f", {total_lines} lines" if total_lines is not None else ""
    return f"{truncate(title, 70)}{line_text} | {summary}"


def _summarize_find_response(parsed: dict[str, Any]) -> str:
    matches = parsed.get("matches") or []
    if isinstance(matches, list):
        return f"{len(matches)} match(es)"
    return truncate(json.dumps(parsed), 160)


def summarize_tool_response(tool_name: str, response_text: str) -> str:
    """One-line summary of the tool response (<=160 chars)."""
    if not response_text.strip():
        return "no response captured"
    parsed = _try_parse_json(response_text)
    if isinstance(parsed, dict):
        if tool_name == "lumina_search":
            return _summarize_search_response(parsed)
        if tool_name == "lumina_open":
            return _summarize_open_response(parsed)
        if tool_name == "lumina_find":
            return _summarize_find_response(parsed)
        if parsed.get("error") or parsed.get("errors"):
            return truncate(json.dumps(parsed), 160)
    return truncate(response_text, 160)


# ---------------------------------------------------------------------------
# Page reference summary (compact page_context handles for tree outline)
# ---------------------------------------------------------------------------

def build_refs_summary(
    indexed_refs: list[tuple[str, Any]],
    max_refs: int = 3,
) -> str | None:
    """Build compact page_context handles for the tree outline.

    Each ref shows: key pc={page_context_json} "title"
    Claude can copy the page_context value into lumina_open/lumina_find calls.
    """
    if not indexed_refs:
        return None
    parts: list[str] = []
    for key, ref in indexed_refs[:max_refs]:
        pc_json = json.dumps(ref.page_context or {}, separators=(",", ":"))
        title = truncate(ref.title or "", 50)
        parts.append(f'{key} pc={pc_json} "{title}"')
    return " | ".join(parts)
