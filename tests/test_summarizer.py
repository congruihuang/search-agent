"""Tests for summarizer module."""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.summarizer import (
    truncate,
    summarize_tool_request,
    summarize_tool_response,
    build_refs_summary,
)


class TestTruncate:
    def test_short_text_unchanged(self) -> None:
        assert truncate("hello", 100) == "hello"

    def test_long_text_truncated(self) -> None:
        result = truncate("a" * 200, 50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_whitespace_collapsed(self) -> None:
        result = truncate("hello   world\n\tfoo", 100)
        assert result == "hello world foo"


class TestSummarizeToolRequest:
    def test_lumina_search(self) -> None:
        result = summarize_tool_request("lumina_search", {"q": "Python tutorial"})
        assert result == "Python tutorial"

    def test_lumina_open_url(self) -> None:
        result = summarize_tool_request("lumina_open", {"url": "https://python.org"})
        assert "url=https://python.org" in result

    def test_lumina_open_page_context(self) -> None:
        result = summarize_tool_request("lumina_open", {"page_context": {"action": "search", "id": 0}})
        assert "page_context=" in result

    def test_lumina_find(self) -> None:
        result = summarize_tool_request("lumina_find", {"pattern": "install", "query_type": "semantic"})
        assert 'find "install"' in result
        assert "semantic" in result

    def test_save_page(self) -> None:
        result = summarize_tool_request("save_page", {"ref_key": "p0", "filename": "mypage"})
        assert "ref=p0" in result
        assert "mypage" in result

    def test_grep_file(self) -> None:
        result = summarize_tool_request("grep_file", {"filename": "doc", "pattern": "error"})
        assert "grep" in result
        assert "error" in result

    def test_submit_answer(self) -> None:
        result = summarize_tool_request("submit_answer", {"answer": "Paris"})
        assert "Paris" in result


class TestSummarizeToolResponse:
    def test_search_response(self) -> None:
        data = {"results": [{"title": "Result 1"}, {"title": "Result 2"}]}
        result = summarize_tool_response("lumina_search", json.dumps(data))
        assert "2 result(s)" in result
        assert "Result 1" in result

    def test_open_response(self) -> None:
        data = {"pages": [{"title": "My Page", "content": "Hello world", "totalLines": 42}]}
        result = summarize_tool_response("lumina_open", json.dumps(data))
        assert "My Page" in result
        assert "42 lines" in result

    def test_find_response(self) -> None:
        data = {"matches": [{"line": 1}, {"line": 5}]}
        result = summarize_tool_response("lumina_find", json.dumps(data))
        assert "2 match(es)" in result

    def test_empty_response(self) -> None:
        result = summarize_tool_response("lumina_search", "  ")
        assert result == "no response captured"

    def test_plain_text_response(self) -> None:
        result = summarize_tool_response("save_page", "Saved 100 lines to output/page.md")
        assert "Saved 100 lines" in result


class TestBuildRefsSummary:
    def test_empty_refs(self) -> None:
        assert build_refs_summary([]) is None

    def test_single_ref(self) -> None:
        @dataclass
        class FakeRef:
            page_context: dict = None
            title: str = ""

        refs = [("r0", FakeRef(page_context={"action": "search", "id": 0}, title="Test Result"))]
        result = build_refs_summary(refs)
        assert result is not None
        assert "r0" in result
        assert '"action":"search"' in result
        assert "Test Result" in result

    def test_max_refs(self) -> None:
        @dataclass
        class FakeRef:
            page_context: dict = None
            title: str = ""

        refs = [(f"r{i}", FakeRef(page_context={"id": i}, title=f"Title {i}")) for i in range(5)]
        result = build_refs_summary(refs, max_refs=2)
        assert result is not None
        assert "r0" in result
        assert "r1" in result
        assert "r2" not in result
