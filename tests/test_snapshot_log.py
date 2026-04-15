"""Tests for snapshot_log module."""

from __future__ import annotations

import json
from pathlib import Path

from src.snapshot_log import SnapshotLog


class TestSnapshotLogHeader:
    def test_write_header(self, tmp_path: Path) -> None:
        log = SnapshotLog(tmp_path / "debug.jsonl")
        log.write_header(
            query="test question",
            config_dict={"model": "claude-sonnet-4-6", "max_iterations": 10},
            tool_schema_names=["lumina_search", "submit_answer"],
        )
        lines = (tmp_path / "debug.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        header = json.loads(lines[0])
        assert header["type"] == "header"
        assert header["query"] == "test question"
        assert header["config"]["model"] == "claude-sonnet-4-6"
        assert "lumina_search" in header["tool_schema_names"]
        assert "start_time_ms" in header


class TestSnapshotLogIteration:
    def test_full_iteration_lifecycle(self, tmp_path: Path) -> None:
        log = SnapshotLog(tmp_path / "debug.jsonl")
        log.write_header("q", {}, [])

        # Begin iteration
        log.begin_iteration(
            iteration=0,
            system_prompt="You are a research agent...",
            user_message="Question: test?",
            tree_outline="- ROOT: test",
            tree_node_count=1,
        )

        # Record API response
        log.record_api_response(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            input_tokens=5000,
            output_tokens=300,
            stop_reason="tool_use",
            latency_ms=2100,
        )

        # Record parsed output
        log.record_output(
            reasoning="I should search for this",
            tool_calls=[{"id": "t1", "name": "lumina_search", "input": {"q": "test"}}],
            final_answer=None,
            final_confidence=None,
        )

        # Record tool result
        log.record_tool_result(
            tool_call_id="t1",
            tool_name="lumina_search",
            status="success",
            latency_ms=450,
            result_full={"results": [{"title": "Test Result"}]},
            result_summary="1 result(s) | top: Test Result",
            tree_node_id="n3",
        )

        # End iteration
        log.end_iteration(
            new_node_ids=["n2", "n3"],
            model_node_id="n2",
            dead_marked=[],
        )

        lines = (tmp_path / "debug.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 iteration
        snap = json.loads(lines[1])
        assert snap["type"] == "iteration"
        assert snap["iteration"] == 0
        assert snap["input"]["system_prompt"] == "You are a research agent..."
        assert snap["input"]["tree_outline"] == "- ROOT: test"
        assert snap["api_call"]["input_tokens"] == 5000
        assert snap["api_call"]["latency_ms"] == 2100
        assert snap["output"]["reasoning"] == "I should search for this"
        assert len(snap["output"]["tool_calls"]) == 1
        assert len(snap["tool_results"]) == 1
        assert snap["tool_results"][0]["latency_ms"] == 450
        assert snap["tree_after"]["new_node_ids"] == ["n2", "n3"]

    def test_pending_cleared_after_end(self, tmp_path: Path) -> None:
        log = SnapshotLog(tmp_path / "debug.jsonl")
        log.begin_iteration(0, "sys", "usr", "tree", 1)
        log.end_iteration(["n1"], "n1")
        assert log._pending is None

    def test_no_crash_when_no_pending(self, tmp_path: Path) -> None:
        """Calling record methods without begin_iteration is a no-op."""
        log = SnapshotLog(tmp_path / "debug.jsonl")
        log.record_api_response("m", 4096, 0, 0, None, 0)
        log.record_output("text", [], None, None)
        log.record_tool_result("id", "name", "ok", 0, {}, "", "n1")
        log.end_iteration([], "n1")
        # File should be empty (header not written, no iteration)
        assert (tmp_path / "debug.jsonl").read_text() == ""


class TestSnapshotLogFooter:
    def test_write_footer(self, tmp_path: Path) -> None:
        log = SnapshotLog(tmp_path / "debug.jsonl")
        log.write_footer(
            total_iterations=5,
            total_input_tokens=50000,
            total_output_tokens=2000,
            total_duration_ms=30000,
            answer="42",
            confidence="high",
        )
        lines = (tmp_path / "debug.jsonl").read_text().strip().split("\n")
        footer = json.loads(lines[0])
        assert footer["type"] == "footer"
        assert footer["total_iterations"] == 5
        assert footer["answer"] == "42"


class TestSnapshotLogTruncation:
    def test_large_result_truncated(self, tmp_path: Path) -> None:
        log = SnapshotLog(tmp_path / "debug.jsonl")
        log.begin_iteration(0, "sys", "usr", "tree", 1)
        # Create a result with >50K of JSON content
        big_result = {"data": "x" * 60_000}
        log.record_tool_result("t1", "lumina_open", "success", 100, big_result, "summary", "n1")
        log.end_iteration(["n1"], "n1")

        lines = (tmp_path / "debug.jsonl").read_text().strip().split("\n")
        snap = json.loads(lines[0])
        tr = snap["tool_results"][0]
        assert tr["result_full"]["_truncated"] is True
        assert len(tr["result_full"]["_preview"]) == 50_000
