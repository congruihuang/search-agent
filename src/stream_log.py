"""Writes stream.jsonl for post-hoc visualization with build_stream_tree.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class StreamLog:
    """Append-only JSONL writer compatible with build_stream_tree.py."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate on start
        self._path.write_text("", encoding="utf-8")

    def _append(self, record: dict[str, Any]) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def write_assistant(
        self,
        reasoning: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """Write an assistant message (reasoning + optional tool_use blocks)."""
        content: list[dict[str, Any]] = []
        if reasoning:
            content.append({"type": "text", "text": reasoning})
        for tc in tool_calls or []:
            content.append({
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": tc.get("name", ""),
                "input": tc.get("input", {}),
            })
        self._append({
            "type": "assistant",
            "message": {"role": "assistant", "content": content},
        })

    def write_tool_result(
        self,
        tool_use_id: str,
        result_text: str,
    ) -> None:
        """Write a tool result message."""
        self._append({
            "type": "user",
            "message": {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_text,
                }],
            },
        })

    def write_result(
        self,
        result: str,
        duration_ms: int,
        num_turns: int,
        total_cost_usd: float | None = None,
    ) -> None:
        """Write the final result record."""
        self._append({
            "type": "result",
            "result": result,
            "duration_ms": duration_ms,
            "num_turns": num_turns,
            "total_cost_usd": total_cost_usd,
        })
