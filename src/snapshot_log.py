"""Per-iteration debug snapshot writer.

Captures the full input→output→tree-state cycle for each orchestrator
iteration into debug_snapshots.jsonl.  Complements stream.jsonl (which
is kept for backward compatibility with build_stream_tree.py).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class SnapshotLog:
    """Append-only JSONL writer for per-iteration debug snapshots."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("", encoding="utf-8")
        self._pending: dict[str, Any] | None = None

    def _append(self, record: dict[str, Any]) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    # -- Header / Footer --

    def write_header(
        self,
        query: str,
        config_dict: dict[str, Any],
        tool_schema_names: list[str],
    ) -> None:
        self._append({
            "type": "header",
            "query": query,
            "config": config_dict,
            "tool_schema_names": tool_schema_names,
            "start_time_ms": int(time.time() * 1000),
        })

    def write_footer(
        self,
        total_iterations: int,
        total_input_tokens: int,
        total_output_tokens: int,
        total_duration_ms: int,
        answer: str | None,
        confidence: str | None,
    ) -> None:
        self._append({
            "type": "footer",
            "total_iterations": total_iterations,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_duration_ms": total_duration_ms,
            "answer": answer,
            "confidence": confidence,
        })

    # -- Iteration lifecycle --

    def begin_iteration(
        self,
        iteration: int,
        system_prompt: str,
        user_message: str,
        tree_outline: str,
        tree_node_count: int,
    ) -> None:
        self._pending = {
            "type": "iteration",
            "iteration": iteration,
            "timestamp_ms": int(time.time() * 1000),
            "input": {
                "system_prompt": system_prompt,
                "user_message": user_message,
                "tree_outline": tree_outline,
                "tree_node_count": tree_node_count,
            },
            "api_call": {},
            "output": {},
            "tool_results": [],
            "tree_after": {},
        }

    def record_api_response(
        self,
        model: str,
        max_tokens: int,
        input_tokens: int,
        output_tokens: int,
        stop_reason: str | None,
        latency_ms: int,
    ) -> None:
        if self._pending is None:
            return
        self._pending["api_call"] = {
            "model": model,
            "max_tokens": max_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "stop_reason": stop_reason,
            "latency_ms": latency_ms,
        }

    def record_output(
        self,
        reasoning: str,
        tool_calls: list[dict[str, Any]],
        final_answer: str | None,
        final_confidence: str | None,
    ) -> None:
        if self._pending is None:
            return
        self._pending["output"] = {
            "reasoning": reasoning,
            "tool_calls": tool_calls,
            "final_answer": final_answer,
            "final_confidence": final_confidence,
        }

    def record_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        status: str,
        latency_ms: int,
        result_full: dict[str, Any],
        result_summary: str,
        tree_node_id: str,
    ) -> None:
        if self._pending is None:
            return
        # Truncate very large result payloads (lumina_open can be huge)
        result_str = json.dumps(result_full, ensure_ascii=True)
        if len(result_str) > 50_000:
            result_full = {"_truncated": True, "_preview": result_str[:50_000]}
        self._pending["tool_results"].append({
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "status": status,
            "latency_ms": latency_ms,
            "result_full": result_full,
            "result_summary": result_summary,
            "tree_node_id": tree_node_id,
        })

    def end_iteration(
        self,
        new_node_ids: list[str],
        model_node_id: str | None,
        dead_marked: list[str] | None = None,
    ) -> None:
        if self._pending is None:
            return
        self._pending["tree_after"] = {
            "new_node_ids": new_node_ids,
            "model_node_id": model_node_id,
            "dead_marked": dead_marked or [],
        }
        self._append(self._pending)
        self._pending = None
