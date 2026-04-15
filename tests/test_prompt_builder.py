"""Tests for prompt_builder."""

from __future__ import annotations

from src.prompt_builder import build_iteration_prompt, SYSTEM_PROMPT


class TestBuildIterationPrompt:
    def test_returns_system_and_user(self) -> None:
        system, user = build_iteration_prompt(
            query="What is Python?",
            tree_outline="- ROOT: What is Python?",
            iteration=0,
            max_iterations=30,
        )
        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_system_contains_tree_outline(self) -> None:
        outline = "- ROOT: test question\n  - MODEL [n2]: reasoning"
        system, _ = build_iteration_prompt(
            query="test question",
            tree_outline=outline,
            iteration=5,
            max_iterations=30,
        )
        assert outline in system

    def test_system_contains_iteration_info(self) -> None:
        system, _ = build_iteration_prompt(
            query="q", tree_outline="", iteration=4, max_iterations=30,
        )
        assert "5/30" in system

    def test_user_contains_query(self) -> None:
        _, user = build_iteration_prompt(
            query="What is the meaning of life?",
            tree_outline="",
            iteration=0,
            max_iterations=10,
        )
        assert "What is the meaning of life?" in user

    def test_system_prompt_has_tool_guidance(self) -> None:
        assert "lumina_search" in SYSTEM_PROMPT
        assert "lumina_open" in SYSTEM_PROMPT
        assert "lumina_find" in SYSTEM_PROMPT
        assert "save_page" in SYSTEM_PROMPT
        assert "submit_answer" in SYSTEM_PROMPT
        assert "tool_state" in SYSTEM_PROMPT

    def test_branch_directive_in_full_prompt(self) -> None:
        system, _ = build_iteration_prompt(
            query="q", tree_outline="", iteration=0, max_iterations=10,
        )
        assert "branch:" in system
