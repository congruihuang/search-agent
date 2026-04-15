"""Main orchestrator: drives the iteration loop using the Anthropic API directly.

Each iteration:
  1. Render tree to markdown outline
  2. Build prompt (system + tree outline + instructions)
  3. Call Claude via anthropic.messages.create (with tool schemas)
  4. Parse response: extract reasoning + tool_use blocks + final_answer
  5. Add ModelNode to tree
  6. If final answer: add FinalNode, done
  7. If tool_use: add ToolNodes, execute in parallel, update tree
  8. Append to stream.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic

from .config import AgentConfig
from .lumina_interceptor import LuminaInterceptor
from .prompt_builder import build_iteration_prompt
from .snapshot_log import SnapshotLog
from .stream_log import StreamLog
from .tree_manager import TreeManager


# ---------------------------------------------------------------------------
# Tool schemas for custom tools (in Anthropic API format)
# ---------------------------------------------------------------------------

CUSTOM_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "save_page",
        "description": (
            "Save a page's full content to a local markdown file. "
            "Pass the ref key from the tree outline (e.g. 'p0', 'r3')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ref_key": {"type": "string", "description": "Reference key from tree outline refs line"},
                "filename": {"type": "string", "description": "Filename (without extension) to save as"},
            },
            "required": ["ref_key", "filename"],
        },
    },
    {
        "name": "grep_file",
        "description": (
            "Search a saved file for lines matching a regex pattern. "
            "Returns matching lines with line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Filename (without extension) to search"},
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
            },
            "required": ["filename", "pattern"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a range of lines from a saved file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Filename (without extension) to read"},
                "offset": {"type": "integer", "description": "Line offset to start reading from", "default": 0},
                "limit": {"type": "integer", "description": "Number of lines to read", "default": 100},
            },
            "required": ["filename"],
        },
    },
    {
        "name": "submit_answer",
        "description": "Submit the final answer to the user's question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The answer to the question"},
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Confidence level in the answer",
                },
            },
            "required": ["answer", "confidence"],
        },
    },
]


# ---------------------------------------------------------------------------
# Custom tool execution
# ---------------------------------------------------------------------------

async def execute_custom_tool(
    tool_name: str,
    args: dict[str, Any],
    interceptor: LuminaInterceptor,
    config: AgentConfig,
) -> dict[str, Any]:
    """Execute a custom (non-Lumina) tool and return an MCP-style result dict."""

    if tool_name == "save_page":
        ref = interceptor.page_refs.get(args.get("ref_key", ""))
        if not ref:
            return _text_result(f"Error: unknown ref key '{args.get('ref_key')}'")
        if not ref.raw_content:
            return _text_result(f"Error: ref '{args.get('ref_key')}' has no saved content (open the page first)")
        path = config.output_dir / "pages" / f"{args['filename']}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ref.raw_content, encoding="utf-8")
        line_count = ref.raw_content.count("\n") + 1
        return _text_result(f"Saved {line_count} lines to {path}")

    if tool_name == "grep_file":
        path = config.output_dir / "pages" / f"{args['filename']}.md"
        if not path.exists():
            return _text_result(f"Error: file not found: {path}")
        matches = []
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(args["pattern"], line, re.IGNORECASE):
                matches.append(f"{i}: {line}")
        result = "\n".join(matches[:50]) if matches else "No matches found."
        return _text_result(result)

    if tool_name == "read_file":
        path = config.output_dir / "pages" / f"{args['filename']}.md"
        if not path.exists():
            return _text_result(f"Error: file not found: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        start = max(0, args.get("offset", 0))
        end = start + args.get("limit", 100)
        chunk = lines[start:end]
        result = "\n".join(f"{i + start + 1}: {l}" for i, l in enumerate(chunk))
        return _text_result(result)

    if tool_name == "submit_answer":
        return _text_result(f"ANSWER: {args.get('answer', '')}")

    return _text_result(f"Error: unknown tool '{tool_name}'")


def _text_result(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


# ---------------------------------------------------------------------------
# Parsed response
# ---------------------------------------------------------------------------

@dataclass
class ParsedResponse:
    reasoning: str
    tool_calls: list[dict[str, Any]]  # [{id, name, input}, ...]
    final_answer: str | None
    final_confidence: str | None


def parse_response(response: anthropic.types.Message) -> ParsedResponse:
    """Extract reasoning text, tool_use blocks, and final answer from API response."""
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    final_answer: str | None = None
    final_confidence: str | None = None

    for block in response.content:
        if block.type == "text":
            reasoning_parts.append(block.text)
        elif block.type == "tool_use":
            if block.name == "submit_answer":
                final_answer = block.input.get("answer")
                final_confidence = block.input.get("confidence")
            else:
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

    return ParsedResponse(
        reasoning="\n".join(reasoning_parts),
        tool_calls=tool_calls,
        final_answer=final_answer,
        final_confidence=final_confidence,
    )


# ---------------------------------------------------------------------------
# Strip tool_state and searchResultContentType from schemas sent to Claude
# ---------------------------------------------------------------------------

def _strip_orchestrator_fields(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove tool_state and searchResultContentType from Lumina tool schemas.

    Claude should never set these - the orchestrator injects them.
    Removing them from the schema prevents Claude from wasting tokens on them.
    """
    stripped: list[dict[str, Any]] = []
    for schema in schemas:
        s = dict(schema)
        input_schema = s.get("input_schema")
        if isinstance(input_schema, dict):
            props = input_schema.get("properties")
            if isinstance(props, dict):
                new_props = {
                    k: v for k, v in props.items()
                    if k not in ("tool_state", "searchResultContentType")
                }
                new_required = [
                    r for r in input_schema.get("required", [])
                    if r not in ("tool_state", "searchResultContentType")
                ]
                s["input_schema"] = {
                    **input_schema,
                    "properties": new_props,
                    "required": new_required,
                }
        stripped.append(s)
    return stripped


# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    answer: str | None
    confidence: str | None
    tree: TreeManager
    iterations: int
    total_input_tokens: int
    total_output_tokens: int
    duration_ms: int


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_search_agent(query: str, config: AgentConfig) -> AgentResult:
    """Run the search agent loop and return the result."""
    t0 = time.monotonic()

    # Load MCP config
    mcp_raw = json.loads(config.mcp_config_path.read_text(encoding="utf-8"))
    lumina_cfg = mcp_raw["mcpServers"]["lumina-web"]

    # Connect to Lumina MCP server and get tool schemas
    interceptor = LuminaInterceptor(lumina_cfg)
    lumina_schemas = await interceptor.connect()
    lumina_schemas_clean = _strip_orchestrator_fields(lumina_schemas)

    # All tool schemas for Claude
    all_tool_schemas = lumina_schemas_clean + CUSTOM_TOOL_SCHEMAS

    # Init tree, stream log, Anthropic client
    tree = TreeManager()
    tree.add_root(query)
    stream_log = StreamLog(config.output_dir / "stream.jsonl")

    # Debug snapshots
    snapshot_log: SnapshotLog | None = None
    if config.enable_debug:
        snapshot_log = SnapshotLog(config.output_dir / "debug_snapshots.jsonl")
        snapshot_log.write_header(
            query=query,
            config_dict={
                "model": config.model,
                "max_iterations": config.max_iterations,
                "max_budget_usd": config.max_budget_usd,
            },
            tool_schema_names=[s["name"] for s in all_tool_schemas],
        )

    # Support ANTHROPIC_AUTH_TOKEN (proxy) fallback when ANTHROPIC_API_KEY is unset
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Track state
    pending_tool_node_ids: list[str] = []
    total_input_tokens = 0
    total_output_tokens = 0
    final_parsed: ParsedResponse | None = None
    stall_count = 0
    iteration = 0

    try:
        for iteration in range(config.max_iterations):
            # 1. Render tree
            tree_outline = tree.render_outline()

            # 2. Build prompt
            system_prompt, user_message = build_iteration_prompt(
                query=query,
                tree_outline=tree_outline,
                iteration=iteration,
                max_iterations=config.max_iterations,
            )

            # Debug: capture iteration input
            if snapshot_log:
                snapshot_log.begin_iteration(
                    iteration=iteration,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    tree_outline=tree_outline,
                    tree_node_count=len(tree.nodes),
                )

            # 3. Call Claude
            api_t0 = time.monotonic()
            response = await client.messages.create(
                model=config.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                tools=all_tool_schemas,
            )
            api_latency_ms = int((time.monotonic() - api_t0) * 1000)

            # Track tokens
            iter_input_tokens = 0
            iter_output_tokens = 0
            if response.usage:
                iter_input_tokens = response.usage.input_tokens
                iter_output_tokens = response.usage.output_tokens
                total_input_tokens += iter_input_tokens
                total_output_tokens += iter_output_tokens

            # 4. Parse response
            parsed = parse_response(response)
            final_parsed = parsed

            # Debug: capture API response + parsed output
            if snapshot_log:
                snapshot_log.record_api_response(
                    model=config.model,
                    max_tokens=4096,
                    input_tokens=iter_input_tokens,
                    output_tokens=iter_output_tokens,
                    stop_reason=response.stop_reason,
                    latency_ms=api_latency_ms,
                )
                snapshot_log.record_output(
                    reasoning=parsed.reasoning,
                    tool_calls=parsed.tool_calls,
                    final_answer=parsed.final_answer,
                    final_confidence=parsed.final_confidence,
                )

            # Log to stream
            stream_log.write_assistant(
                reasoning=parsed.reasoning,
                tool_calls=parsed.tool_calls,
            )

            # 5. Add model node
            if pending_tool_node_ids:
                model_node = tree.add_model_node_after_tools(
                    parsed.reasoning, pending_tool_node_ids,
                )
                pending_tool_node_ids = []
            else:
                model_node = tree.add_model_node(
                    parsed.reasoning, parent=tree.current_leaf_id,
                )

            # 6. Check for final answer
            if parsed.final_answer:
                final_node = tree.add_final_node(
                    parsed.final_answer,
                    parent=model_node.id,
                    confidence=parsed.final_confidence,
                )
                if snapshot_log:
                    snapshot_log.end_iteration(
                        new_node_ids=[model_node.id, final_node.id],
                        model_node_id=model_node.id,
                    )
                break

            # 7. Execute tools
            # Lumina tools share toolState and must run sequentially.
            # Non-Lumina tools can run in parallel with each other.
            if parsed.tool_calls:
                stall_count = 0
                tool_nodes: list[tuple[Any, dict[str, Any]]] = []
                for tc in parsed.tool_calls:
                    tool_node = tree.add_tool_node(
                        parent=model_node.id,
                        tool_name=tc["name"],
                        request=tc["input"],
                    )
                    tool_nodes.append((tool_node, tc))

                async def execute_one(
                    tool_node: Any, tc: dict[str, Any]
                ) -> str:
                    tool_name = tc["name"]
                    tool_t0 = time.monotonic()
                    result: dict[str, Any] = {}
                    status = "success"
                    try:
                        if interceptor.is_lumina_tool(tool_name):
                            result, indexed_refs = await interceptor.intercept(
                                tool_name, tc["input"],
                            )
                        else:
                            result = await execute_custom_tool(
                                tool_name, tc["input"], interceptor, config,
                            )
                            indexed_refs = []

                        tree.complete_tool_node(
                            tool_node.id,
                            result=result,
                            indexed_refs=indexed_refs,
                            status="success",
                            tool_state=interceptor.tool_state,
                        )
                    except Exception as exc:
                        result = _text_result(f"Error: {exc}")
                        status = "error"
                        tree.complete_tool_node(
                            tool_node.id,
                            result=result,
                            status="error",
                        )

                    tool_latency_ms = int((time.monotonic() - tool_t0) * 1000)

                    # Debug: capture tool result
                    if snapshot_log:
                        snapshot_log.record_tool_result(
                            tool_call_id=tc.get("id", ""),
                            tool_name=tool_name,
                            status=status,
                            latency_ms=tool_latency_ms,
                            result_full=result,
                            result_summary=tree.nodes[tool_node.id].response_summary,  # type: ignore[union-attr]
                            tree_node_id=tool_node.id,
                        )

                    # Log tool result
                    stream_log.write_tool_result(
                        tc.get("id", ""),
                        tree.nodes[tool_node.id].response_text,  # type: ignore[union-attr]
                    )
                    return tool_node.id

                # Separate Lumina and non-Lumina tools
                lumina_tasks = [(tn, tc) for tn, tc in tool_nodes if interceptor.is_lumina_tool(tc["name"])]
                other_tasks = [(tn, tc) for tn, tc in tool_nodes if not interceptor.is_lumina_tool(tc["name"])]

                results: list[str] = []

                # Run Lumina tools sequentially (shared toolState session)
                for tn, tc in lumina_tasks:
                    results.append(await execute_one(tn, tc))

                # Run non-Lumina tools in parallel
                if other_tasks:
                    results.extend(
                        await asyncio.gather(
                            *(execute_one(tn, tc) for tn, tc in other_tasks)
                        )
                    )

                pending_tool_node_ids = results

                # Debug: end iteration after tool execution
                if snapshot_log:
                    snapshot_log.end_iteration(
                        new_node_ids=[model_node.id] + [tn.id for tn, _ in tool_nodes],
                        model_node_id=model_node.id,
                    )
                continue

            # No tools, no answer -> stall detection
            stall_count += 1
            tree.current_leaf_id = model_node.id
            if snapshot_log:
                snapshot_log.end_iteration(
                    new_node_ids=[model_node.id],
                    model_node_id=model_node.id,
                )
            if stall_count >= 3:
                break

    finally:
        await interceptor.disconnect()

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    answer = final_parsed.final_answer if final_parsed else None
    confidence = final_parsed.final_confidence if final_parsed else None

    # Write final result to stream log
    stream_log.write_result(
        result=answer or "(no answer)",
        duration_ms=elapsed_ms,
        num_turns=iteration + 1,
    )

    # Save tree
    tree.save(config.output_dir / "stream_tree.json")

    # Debug: write footer and generate HTML view
    if snapshot_log:
        snapshot_log.write_footer(
            total_iterations=iteration + 1,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_duration_ms=elapsed_ms,
            answer=answer,
            confidence=confidence,
        )
        from .build_debug_view import build_debug_html
        build_debug_html(
            config.output_dir / "debug_snapshots.jsonl",
            config.output_dir / "debug_view.html",
        )

    return AgentResult(
        answer=answer,
        confidence=confidence,
        tree=tree,
        iterations=iteration + 1,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        duration_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous web-research agent",
    )
    parser.add_argument("query", help="The question to research")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Claude model to use")
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--max-budget", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("--mcp-config", type=Path, default=Path(".mcp.json"))
    parser.add_argument("--debug", action="store_true", default=True,
                        help="Enable debug snapshots and HTML view (default: on)")
    parser.add_argument("--no-debug", dest="debug", action="store_false",
                        help="Disable debug snapshots")
    args = parser.parse_args()

    config = AgentConfig(
        model=args.model,
        max_iterations=args.max_iterations,
        max_budget_usd=args.max_budget,
        output_dir=args.output_dir,
        mcp_config_path=args.mcp_config,
        enable_debug=args.debug,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    result = asyncio.run(run_search_agent(args.query, config))

    print("\n" + "=" * 60)
    if result.answer:
        print(f"ANSWER ({result.confidence}): {result.answer}")
    else:
        print("No answer found.")
    print(f"Iterations: {result.iterations}")
    print(f"Tokens: {result.total_input_tokens:,} in / {result.total_output_tokens:,} out")
    print(f"Duration: {result.duration_ms / 1000:.1f}s")
    print(f"Output: {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
