#!/usr/bin/env python3
"""Build a rooted tree view from a Claude stream.jsonl trace."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def truncate_text(value: str, limit: int = 180) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return ""


def try_parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def summarize_search_response(parsed: dict[str, Any]) -> str:
    results = parsed.get("results") or []
    if not isinstance(results, list):
        return truncate_text(json.dumps(parsed), 160)
    titles: list[str] = []
    for item in results[:2]:
        if isinstance(item, dict):
            title = item.get("title") or item.get("url") or "untitled result"
            titles.append(str(title))
    errors = parsed.get("errors") or []
    parts = [f"{len(results)} result(s)"]
    if titles:
        parts.append("top: " + "; ".join(truncate_text(title, 60) for title in titles))
    if errors:
        parts.append("errors: " + truncate_text(json.dumps(errors), 80))
    return " | ".join(parts)


def summarize_open_response(parsed: dict[str, Any]) -> str:
    pages = parsed.get("pages") or []
    if not isinstance(pages, list) or not pages:
        return truncate_text(json.dumps(parsed), 160)
    first = pages[0]
    if not isinstance(first, dict):
        return truncate_text(json.dumps(first), 160)
    title = str(first.get("title") or first.get("url") or "opened page")
    content = str(first.get("content") or "")
    total_lines = first.get("totalLines")
    summary = truncate_text(content, 110)
    line_text = f", {total_lines} lines" if total_lines is not None else ""
    return f"{truncate_text(title, 70)}{line_text} | {summary}"


def summarize_tool_request(tool_name: str, tool_input: Any) -> str:
    if not isinstance(tool_input, dict):
        return truncate_text(json.dumps(tool_input), 140)
    if tool_name.endswith("lumina_search"):
        return truncate_text(str(tool_input.get("q", "")), 140)
    if tool_name.endswith("lumina_open"):
        return truncate_text(str(tool_input.get("url", "")), 140)
    if tool_name.endswith("lumina_find"):
        url = str(tool_input.get("url", ""))
        text = str(tool_input.get("text", ""))
        return truncate_text(f"url={url} | text={text}", 140)
    return truncate_text(json.dumps(tool_input, ensure_ascii=True), 140)


def summarize_tool_response(tool_name: str, response_text: str) -> str:
    if not response_text.strip():
        return "no response captured"

    parsed = try_parse_json(response_text)
    if isinstance(parsed, dict):
        if tool_name.endswith("lumina_search"):
            return summarize_search_response(parsed)
        if tool_name.endswith("lumina_open"):
            return summarize_open_response(parsed)
        if tool_name.endswith("lumina_find"):
            matches = parsed.get("matches") or []
            return f"{len(matches)} match(es)" if isinstance(matches, list) else truncate_text(response_text, 140)
        if parsed.get("error") or parsed.get("errors"):
            return truncate_text(json.dumps(parsed), 160)

    return truncate_text(response_text, 160)


def sanitize_mermaid_label(text: str) -> str:
    cleaned = text.replace("\r", " ").replace("\n", "<br/>")
    cleaned = cleaned.replace('"', "'")
    cleaned = cleaned.replace("[", "(").replace("]", ")")
    cleaned = cleaned.replace("{", "(").replace("}", ")")
    cleaned = cleaned.replace("|", "/")
    return cleaned


def add_node(nodes: list[dict[str, Any]], node_map: dict[str, dict[str, Any]], kind: str, title: str, details: dict[str, Any]) -> dict[str, Any]:
    node_id = f"n{len(nodes) + 1}"
    node = {
        "id": node_id,
        "kind": kind,
        "title": title,
        "children": [],
        **details,
    }
    nodes.append(node)
    node_map[node_id] = node
    return node


_RESTART_PATTERNS = [
    "different approach",
    "completely different",
    "none of these",
    "step back",
    "didn't return",
    "didn't find",
    "aren't finding",
    "initial searches",
    "search results are mostly",
    "keep pointing",
    "no direct answer",
    "not useful",
    "let me reconsider",
    "let me think differently",
    "try a completely",
    "more targeted search",
    "try more specific",
    "not finding",
    "none of the",
    "let me try a different",
    "let me take a step",
]


def _is_restart_text(text: str) -> bool:
    """Check if assistant text indicates a restart (dismissing prior tool results)."""
    prefix = text[:250].lower()
    return any(p in prefix for p in _RESTART_PATTERNS)


def build_tree(stream_path: Path) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    node_map: dict[str, dict[str, Any]] = {}
    pending_tools: dict[str, dict[str, Any]] = {}
    latest_completed_tool: dict[str, Any] | None = None

    root = add_node(
        nodes,
        node_map,
        "root",
        stream_path.parent.name,
        {
            "summary": f"Trace for {stream_path.name}",
        },
    )
    current_model_parent = root

    with stream_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            event = json.loads(line)
            event_type = event.get("type")

            if event_type == "assistant":
                message = event.get("message") or {}
                content = message.get("content") or []
                text_blocks: list[str] = []
                tool_blocks: list[dict[str, Any]] = []

                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "text":
                        text = str(block.get("text") or "").strip()
                        if text:
                            text_blocks.append(text)
                    elif block_type == "tool_use":
                        tool_blocks.append(block)

                if text_blocks:
                    assistant_text = "\n\n".join(text_blocks)
                    if latest_completed_tool and not _is_restart_text(assistant_text):
                        parent_node = latest_completed_tool
                    else:
                        parent_node = current_model_parent
                    model_node = add_node(
                        nodes,
                        node_map,
                        "model",
                        f"Model {len(nodes)}",
                        {
                            "summary": truncate_text(assistant_text, 180),
                            "assistant_text": assistant_text,
                        },
                    )
                    parent_node["children"].append(model_node["id"])
                    current_model_parent = model_node
                    latest_completed_tool = None

                for tool_block in tool_blocks:
                    tool_name = str(tool_block.get("name") or "tool")
                    tool_input = tool_block.get("input")
                    tool_use_id = str(tool_block.get("id") or "")
                    request_summary = summarize_tool_request(tool_name, tool_input)
                    tool_node = add_node(
                        nodes,
                        node_map,
                        "tool",
                        tool_name,
                        {
                            "tool_use_id": tool_use_id,
                            "request": tool_input,
                            "request_summary": request_summary,
                            "response_text": "",
                            "response_summary": "pending",
                        },
                    )
                    current_model_parent["children"].append(tool_node["id"])
                    pending_tools[tool_use_id] = tool_node

            elif event_type == "user":
                message = event.get("message") or {}
                content = message.get("content") or []
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    tool_use_id = str(block.get("tool_use_id") or "")
                    tool_node = pending_tools.get(tool_use_id)
                    if tool_node is None:
                        continue
                    response_text = extract_text_content(block.get("content"))
                    tool_node["response_text"] = response_text
                    tool_node["response_summary"] = summarize_tool_response(tool_node["title"], response_text)
                    latest_completed_tool = tool_node

            elif event_type == "result":
                final_text = str(event.get("result") or "")
                final_node = add_node(
                    nodes,
                    node_map,
                    "final",
                    "Final Answer",
                    {
                        "summary": truncate_text(final_text, 220),
                        "assistant_text": final_text,
                        "duration_ms": event.get("duration_ms"),
                        "num_turns": event.get("num_turns"),
                        "total_cost_usd": event.get("total_cost_usd"),
                    },
                )
                current_model_parent["children"].append(final_node["id"])

    return {
        "root": root,
        "nodes": nodes,
        "node_map": node_map,
    }


def build_mermaid(tree: dict[str, Any]) -> str:
    nodes = tree["nodes"]
    lines = ["flowchart TD"]

    for node in nodes:
        node_id = node["id"]
        kind = node["kind"]
        if kind == "root":
            label = sanitize_mermaid_label(f"{node['title']}<br/>{node['summary']}")
        elif kind == "model":
            label = sanitize_mermaid_label(f"Model<br/>{node['summary']}")
        elif kind == "tool":
            label = sanitize_mermaid_label(
                f"{node['title']}<br/>req: {node['request_summary']}<br/>resp: {node['response_summary']}"
            )
        else:
            label = sanitize_mermaid_label(f"Final Answer<br/>{node['summary']}")
        lines.append(f'    {node_id}["{label}"]')

    for node in nodes:
        for child_id in node["children"]:
            lines.append(f"    {node['id']} --> {child_id}")

    lines.extend(
        [
            "    classDef root fill:#12343b,color:#f5f3ea,stroke:#0b1d21,stroke-width:2px;",
            "    classDef model fill:#f7d9c4,color:#2f1b12,stroke:#8b5e3c;",
            "    classDef tool fill:#d9efe8,color:#14332b,stroke:#2b7564;",
            "    classDef final fill:#f2e4ff,color:#2b1742,stroke:#6e47a1;",
        ]
    )

    for node in nodes:
        lines.append(f"    class {node['id']} {node['kind']};")

    return "\n".join(lines)


def render_outline(tree: dict[str, Any], node_id: str, depth: int = 0) -> list[str]:
    node = tree["node_map"][node_id]
    prefix = "  " * depth + "- "
    lines: list[str] = []

    if node["kind"] == "root":
        lines.append(f"{prefix}{node['title']}: {node['summary']}")
    elif node["kind"] == "model":
        lines.append(f"{prefix}Model: {node['summary']}")
        lines.append(f"{'  ' * (depth + 1)}assistant: {truncate_text(node['assistant_text'], 320)}")
    elif node["kind"] == "tool":
        lines.append(f"{prefix}Tool: {node['title']}")
        lines.append(f"{'  ' * (depth + 1)}request: {node['request_summary']}")
        lines.append(f"{'  ' * (depth + 1)}response: {node['response_summary']}")
    else:
        lines.append(f"{prefix}Final: {node['summary']}")

    for child_id in node["children"]:
        lines.extend(render_outline(tree, child_id, depth + 1))
    return lines


def build_markdown(tree: dict[str, Any], mermaid: str, stream_path: Path) -> str:
    root = tree["root"]
    outline = "\n".join(render_outline(tree, root["id"]))
    return "\n".join(
        [
            f"# Stream Tree for {stream_path.parent.name}",
            "",
            f"Source: `{stream_path}`",
            "",
            "## Mermaid",
            "",
            "```mermaid",
            mermaid,
            "```",
            "",
            "## Outline",
            "",
            outline,
            "",
        ]
    )


def format_pretty_json(value: Any) -> str:
        return json.dumps(value, indent=2, ensure_ascii=True)


def format_node_body(node: dict[str, Any]) -> str:
        if node["kind"] in {"model", "final"}:
                return html.escape(node.get("assistant_text") or "")
        if node["kind"] == "tool":
                request = format_pretty_json(node.get("request"))
                sections = [
                        '<div class="section-label">Request</div>',
                        f"<pre>{html.escape(request)}</pre>",
                ]
                return "".join(sections)
        return html.escape(node.get("summary") or "")


def build_html_node(tree: dict[str, Any], node_id: str) -> str:
    node = tree["node_map"][node_id]
    kind = node["kind"]
    child_html = "".join(build_html_node(tree, child_id) for child_id in node["children"])
    has_children = bool(node["children"])
    summary = html.escape(node.get("summary") or "")
    title = html.escape(node["title"])
    expander = '<span class="expander" aria-hidden="true"></span>' if has_children else '<span class="expander-spacer" aria-hidden="true"></span>'

    if kind == "root":
        return (
            f'<li class="tree-item root{" has-children" if has_children else ""}" data-node-id="{node_id}" data-kind="{kind}">'
            f'<details open><summary>{expander}<span class="badge root">ROOT</span><span class="node-title">{title}</span>'
            f'<span class="node-summary">{summary}</span></summary>'
            f'<div class="node-body">{html.escape(node.get("summary") or "")}</div>'
            f'<ul>{child_html}</ul></details></li>'
        )

    if kind == "tool":
        subtitle = html.escape(node.get("request_summary") or "")
    else:
        subtitle = summary

    body = format_node_body(node)
    return (
        f'<li class="tree-item {kind}{" has-children" if has_children else ""}" data-node-id="{node_id}" data-kind="{kind}">'
        f'<details><summary>{expander}<span class="badge {kind}">{kind.upper()}</span>'
        f'<span class="node-title">{title}</span><span class="node-summary">{subtitle}</span></summary>'
        f'<div class="node-body">{body}</div>'
        f'<ul>{child_html}</ul></details></li>'
    )


def build_html(tree: dict[str, Any], stream_path: Path, mermaid_path: Path) -> str:
        root = tree["root"]
        nodes = tree["nodes"]
        html_tree = build_html_node(tree, root["id"])
        payload = json.dumps(nodes, ensure_ascii=True)
        title = f"Interactive Stream Tree: {stream_path.parent.name}"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{html.escape(title)}</title>
    <style>
        :root {{
            --bg: #f3efe7;
            --panel: #fffdf8;
            --ink: #1d1d1b;
            --muted: #645f57;
            --line: #d8d0c3;
            --root: #12343b;
            --model: #f7d9c4;
            --tool: #d9efe8;
            --final: #f2e4ff;
            --accent: #9d4edd;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: Georgia, 'Palatino Linotype', serif;
            background: radial-gradient(circle at top left, #fff7ea 0, var(--bg) 45%, #ebe5d8 100%);
            color: var(--ink);
        }}
        .shell {{
            display: grid;
            grid-template-columns: minmax(380px, 1.2fr) minmax(320px, 0.8fr);
            min-height: 100vh;
        }}
        .pane {{ padding: 24px; }}
        .tree-pane {{ border-right: 1px solid var(--line); overflow: auto; }}
        .detail-pane {{ background: rgba(255, 253, 248, 0.86); backdrop-filter: blur(8px); }}
        h1 {{ margin: 0 0 8px; font-size: 28px; line-height: 1.15; }}
        .subhead {{ color: var(--muted); margin-bottom: 18px; }}
        .toolbar {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; }}
        button {{
            border: 1px solid var(--line);
            background: var(--panel);
            color: var(--ink);
            padding: 10px 14px;
            border-radius: 999px;
            cursor: pointer;
            font: inherit;
        }}
        button:hover {{ border-color: #b8ac97; }}
        .counter {{
            display: inline-flex;
            align-items: center;
            padding: 10px 14px;
            border-radius: 999px;
            background: #efe8da;
            color: var(--muted);
            border: 1px solid var(--line);
        }}
        ul {{ list-style: none; margin: 0; padding-left: 18px; }}
        .tree-item {{ margin: 8px 0; }}
        details > summary {{
            list-style: none;
            display: grid;
            grid-template-columns: 20px auto auto 1fr;
            gap: 10px;
            align-items: start;
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 12px 14px;
            cursor: pointer;
            box-shadow: 0 10px 25px rgba(31, 23, 11, 0.04);
        }}
        details > summary::-webkit-details-marker {{ display: none; }}
        .expander,
        .expander-spacer {{
            width: 20px;
            height: 20px;
            margin-top: 2px;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex: 0 0 20px;
        }}
        .expander {{
            border: 1px solid #c9baa4;
            background: linear-gradient(180deg, #fffdf8, #f2e9db);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
        }}
        .expander::before {{
            content: '';
            width: 7px;
            height: 7px;
            border-right: 2px solid #7a6854;
            border-bottom: 2px solid #7a6854;
            transform: rotate(-45deg);
            transition: transform 140ms ease;
            margin-right: 2px;
        }}
        details[open] > summary .expander::before {{
            transform: rotate(45deg);
            margin-right: 0;
            margin-top: -2px;
        }}
        .expander-spacer {{
            opacity: 0;
            pointer-events: none;
        }}
        .tree-item.active > details > summary {{
            outline: 3px solid var(--accent);
            outline-offset: 3px;
            box-shadow: 0 0 0 6px rgba(157, 78, 221, 0.14), 0 18px 32px rgba(31, 23, 11, 0.1);
            transform: translateX(4px);
            border-color: #8b43d1;
        }}
        .tree-item.active > details > summary .expander {{
            border-color: #8b43d1;
            background: linear-gradient(180deg, #faf3ff, #ead9ff);
        }}
        .tree-item.active > details > summary .expander::before {{
            border-color: #6b21a8;
        }}
        .tree-item.active > details > .node-body {{
            border-left-color: var(--accent);
            background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(246, 236, 255, 0.92));
            box-shadow: inset 0 0 0 1px rgba(157, 78, 221, 0.14);
        }}
        .tree-item.active > details > summary .node-title {{
            color: #5f1898;
        }}
        .tree-item.active > details > summary .node-summary {{
            color: #4b3b5e;
        }}
        .badge {{
            display: inline-block;
            min-width: 66px;
            text-align: center;
            font-size: 12px;
            letter-spacing: 0.08em;
            padding: 5px 8px;
            border-radius: 999px;
            color: #1a1714;
        }}
        .badge.root {{ background: var(--root); color: #f5f3ea; }}
        .badge.model {{ background: var(--model); }}
        .badge.tool {{ background: var(--tool); }}
        .badge.final {{ background: var(--final); }}
        .node-title {{ font-weight: 700; }}
        .node-summary {{ color: var(--muted); min-width: 0; }}
        .node-body {{
            margin: 10px 12px 8px 12px;
            padding: 12px 14px;
            background: rgba(255,255,255,0.7);
            border-left: 3px solid #d3c1ab;
            border-radius: 0 12px 12px 0;
        }}
        .node-body pre {{
            white-space: pre-wrap;
            word-break: break-word;
            margin: 0 0 14px;
            font-family: Consolas, 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.45;
        }}
        .section-label {{
            font-size: 12px;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 6px;
            text-transform: uppercase;
        }}
        .detail-card {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 20px 35px rgba(31, 23, 11, 0.05);
            position: sticky;
            top: 24px;
        }}
        .detail-meta {{ color: var(--muted); margin-bottom: 12px; }}
        .detail-card pre {{
            background: #f8f3ea;
            border: 1px solid #eadfce;
            border-radius: 14px;
            padding: 14px;
            white-space: pre-wrap;
            word-break: break-word;
            font-family: Consolas, 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.45;
            max-height: 58vh;
            overflow: auto;
        }}
        .link-row {{ margin-top: 16px; color: var(--muted); }}
        .link-row a {{ color: #0a5c63; }}
        @media (max-width: 980px) {{
            .shell {{ grid-template-columns: 1fr; }}
            .tree-pane {{ border-right: 0; border-bottom: 1px solid var(--line); }}
            .detail-card {{ position: static; }}
        }}
    </style>
</head>
<body>
    <div class="shell">
        <section class="pane tree-pane">
            <h1>{html.escape(title)}</h1>
            <div class="subhead">Step through the reasoning chain, inspect each tool request and response, and expand nested branches inline.</div>
            <div class="toolbar">
                <button type="button" id="prevBtn">Previous</button>
                <button type="button" id="nextBtn">Next</button>
                <button type="button" id="expandBtn">Expand All</button>
                <button type="button" id="collapseBtn">Collapse All</button>
                <span class="counter" id="counter"></span>
            </div>
            <ul class="tree-root">{html_tree}</ul>
        </section>
        <aside class="pane detail-pane">
            <div class="detail-card">
                <div class="detail-meta" id="detailMeta">Select a node</div>
                <h2 id="detailTitle">Node details</h2>
                <p id="detailSummary"></p>
                <pre id="detailBody"></pre>
                <div class="link-row">Source trace: <code>{html.escape(str(stream_path))}</code></div>
                <div class="link-row">Mermaid source: <code>{html.escape(str(mermaid_path))}</code></div>
            </div>
        </aside>
    </div>
    <script>
        const nodes = {payload};
        const nodeMap = Object.fromEntries(nodes.map(node => [node.id, node]));
        const focusable = nodes.slice();
        let currentIndex = 0;

        const detailMeta = document.getElementById('detailMeta');
        const detailTitle = document.getElementById('detailTitle');
        const detailSummary = document.getElementById('detailSummary');
        const detailBody = document.getElementById('detailBody');
        const counter = document.getElementById('counter');

        function nodeBody(node) {{
            if (node.kind === 'tool') {{
                const request = JSON.stringify(node.request, null, 2);
                return `Request\n${{request}}\n\nResponse\n${{node.response_text || ''}}`;
            }}
            return node.assistant_text || node.summary || '';
        }}

        function expandParents(element) {{
            let parent = element.parentElement;
            while (parent) {{
                if (parent.tagName === 'DETAILS') {{
                    parent.open = true;
                }}
                parent = parent.parentElement;
            }}
        }}

        function renderActiveNode(index, forceOpen) {{
            currentIndex = (index + focusable.length) % focusable.length;
            const node = focusable[currentIndex];
            document.querySelectorAll('.tree-item').forEach(item => item.classList.remove('active'));
            const element = document.querySelector(`.tree-item[data-node-id="${{node.id}}"]`);
            if (element) {{
                element.classList.add('active');
                if (forceOpen) {{
                    const details = element.querySelector(':scope > details');
                    if (details) {{
                        details.open = true;
                    }}
                }}
                expandParents(element);
                element.scrollIntoView({{ block: 'nearest', behavior: 'smooth' }});
            }}
            detailMeta.textContent = `Step ${{currentIndex + 1}} of ${{focusable.length}} · ${{node.kind}}`;
            detailTitle.textContent = node.title;
            detailSummary.textContent = node.summary || '';
            detailBody.textContent = nodeBody(node);
            counter.textContent = `${{currentIndex + 1}} / ${{focusable.length}} nodes`;
        }}

        document.getElementById('prevBtn').addEventListener('click', () => renderActiveNode(currentIndex - 1, true));
        document.getElementById('nextBtn').addEventListener('click', () => renderActiveNode(currentIndex + 1, true));
        document.getElementById('expandBtn').addEventListener('click', () => {{
            document.querySelectorAll('details').forEach(item => item.open = true);
        }});
        document.getElementById('collapseBtn').addEventListener('click', () => {{
            document.querySelectorAll('.tree-item details').forEach((item, idx) => item.open = idx === 0);
        }});

        document.querySelector('.tree-root').addEventListener('click', event => {{
            const interactiveTarget = event.target.closest('summary, .node-body');
            if (!interactiveTarget) {{
                return;
            }}
            const container = interactiveTarget.closest('.tree-item');
            if (!container) {{
                return;
            }}
            const nodeId = container.getAttribute('data-node-id');
            const index = focusable.findIndex(node => node.id === nodeId);
            if (index >= 0) {{
                renderActiveNode(index, false);
            }}
        }});

        renderActiveNode(0, true);
    </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stream_path", type=Path, help="Path to stream.jsonl")
    parser.add_argument("--output-dir", type=Path, help="Directory for generated artifacts")
    args = parser.parse_args()

    stream_path = args.stream_path.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else stream_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    tree = build_tree(stream_path)
    mermaid = build_mermaid(tree)
    markdown = build_markdown(tree, mermaid, stream_path)

    serializable_tree = {
        "root": tree["root"],
        "nodes": tree["nodes"],
    }

    stem = stream_path.stem
    json_path = output_dir / f"{stem}_tree.json"
    mermaid_path = output_dir / f"{stem}_tree.mmd"
    markdown_path = output_dir / f"{stem}_tree.md"
    html_path = output_dir / f"{stem}_tree.html"
    html_page = build_html(tree, stream_path, mermaid_path)

    json_path.write_text(json.dumps(serializable_tree, indent=2, ensure_ascii=True), encoding="utf-8")
    mermaid_path.write_text(mermaid, encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html_page, encoding="utf-8")

    print(json.dumps({
        "json": str(json_path),
        "mermaid": str(mermaid_path),
        "markdown": str(markdown_path),
        "html": str(html_path),
        "node_count": len(tree["nodes"]),
    }, indent=2))


if __name__ == "__main__":
    main()