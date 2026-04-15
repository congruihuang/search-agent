"""Tests for TreeManager: node creation, dead-end logic, and rendering."""

from __future__ import annotations

import json

from src.tree_manager import TreeManager, RootNode, ModelNode, ToolNode, FinalNode


class TestTreeBasics:
    def test_add_root(self) -> None:
        tree = TreeManager()
        root = tree.add_root("What is the capital of France?")
        assert root.kind == "root"
        assert root.id == "n1"
        assert tree.root_id == "n1"
        assert tree.current_leaf_id == "n1"

    def test_add_model_node(self) -> None:
        tree = TreeManager()
        tree.add_root("question")
        model = tree.add_model_node("I'll search for this.", parent="n1")
        assert model.kind == "model"
        assert model.parent_id == "n1"
        assert model.id in tree.nodes["n1"].children

    def test_add_tool_node(self) -> None:
        tree = TreeManager()
        tree.add_root("question")
        model = tree.add_model_node("searching", parent="n1")
        tool = tree.add_tool_node(
            parent=model.id,
            tool_name="lumina_search",
            request={"q": "capital of France"},
        )
        assert tool.kind == "tool"
        assert tool.tool_name == "lumina_search"
        assert tool.status == "pending"
        assert tool.id in tree.nodes[model.id].children

    def test_complete_tool_node(self) -> None:
        tree = TreeManager()
        tree.add_root("question")
        model = tree.add_model_node("searching", parent="n1")
        tool = tree.add_tool_node(parent=model.id, tool_name="lumina_search", request={"q": "test"})
        tree.complete_tool_node(
            tool.id,
            result={"results": [{"title": "Result 1", "url": "http://example.com"}]},
            status="success",
        )
        node = tree.nodes[tool.id]
        assert isinstance(node, ToolNode)
        assert node.status == "success"
        assert "1 result" in node.response_summary

    def test_add_final_node(self) -> None:
        tree = TreeManager()
        tree.add_root("question")
        model = tree.add_model_node("I know the answer", parent="n1")
        final = tree.add_final_node("Paris", parent=model.id, confidence="high")
        assert final.kind == "final"
        assert isinstance(final, FinalNode)
        assert final.confidence == "high"


class TestDeadEndLogic:
    def _setup_parallel_tools(self) -> tuple[TreeManager, str, list[str]]:
        """Create a tree with a model node and 3 parallel tool children."""
        tree = TreeManager()
        tree.add_root("question")
        model = tree.add_model_node("Try multiple searches", parent="n1")
        ids = []
        for i in range(3):
            tool = tree.add_tool_node(
                parent=model.id,
                tool_name="lumina_search",
                request={"q": f"search {i}"},
            )
            tree.complete_tool_node(tool.id, result={"results": []}, status="success")
            ids.append(tool.id)
        return tree, model.id, ids

    def test_single_tool_no_dead_marking(self) -> None:
        tree = TreeManager()
        tree.add_root("question")
        model = tree.add_model_node("searching", parent="n1")
        tool = tree.add_tool_node(parent=model.id, tool_name="lumina_search", request={"q": "test"})
        tree.complete_tool_node(tool.id, result={"results": []}, status="success")

        new_model = tree.add_model_node_after_tools("Analyzing results", [tool.id])
        assert new_model.parent_id == tool.id
        assert not tree.nodes[tool.id].dead

    def test_branch_directive_selects_one(self) -> None:
        tree, model_id, tool_ids = self._setup_parallel_tools()
        chosen = tool_ids[1]  # e.g. n4
        reasoning = f"branch: {chosen}\nThis result is most promising."

        new_model = tree.add_model_node_after_tools(reasoning, tool_ids)
        assert new_model.parent_id == chosen
        for tid in tool_ids:
            if tid == chosen:
                assert not tree.nodes[tid].dead
            else:
                assert tree.nodes[tid].dead

    def test_branch_none_marks_all_dead(self) -> None:
        tree, model_id, tool_ids = self._setup_parallel_tools()
        reasoning = "branch: none\nNone of these results are useful."

        new_model = tree.add_model_node_after_tools(reasoning, tool_ids)
        assert new_model.parent_id == model_id
        for tid in tool_ids:
            assert tree.nodes[tid].dead

    def test_no_branch_directive_defaults_to_first(self) -> None:
        tree, model_id, tool_ids = self._setup_parallel_tools()
        reasoning = "Let me analyze these results."

        new_model = tree.add_model_node_after_tools(reasoning, tool_ids)
        assert new_model.parent_id == tool_ids[0]
        for tid in tool_ids:
            assert not tree.nodes[tid].dead  # conservative: no marking


class TestTreeRendering:
    def test_basic_render(self) -> None:
        tree = TreeManager()
        tree.add_root("What is Python?")
        model = tree.add_model_node("I'll search for this.", parent="n1")
        tool = tree.add_tool_node(parent=model.id, tool_name="lumina_search", request={"q": "Python"})
        tree.complete_tool_node(tool.id, result={"results": [{"title": "Python.org"}]}, status="success")

        outline = tree.render_outline()
        assert "ROOT:" in outline
        assert "MODEL [n2]:" in outline
        assert "TOOL [lumina_search] [n3]:" in outline
        assert "->" in outline  # response summary line

    def test_dead_nodes_collapsed(self) -> None:
        tree = TreeManager()
        tree.add_root("question")
        model = tree.add_model_node("Try two searches", parent="n1")

        tool1 = tree.add_tool_node(parent=model.id, tool_name="lumina_search", request={"q": "first"})
        tree.complete_tool_node(tool1.id, result={"results": []}, status="success")
        tool2 = tree.add_tool_node(parent=model.id, tool_name="lumina_search", request={"q": "second"})
        tree.complete_tool_node(tool2.id, result={"results": [{"title": "Found it"}]}, status="success")

        # Mark tool1 as dead
        tree._mark_subtree_dead(tool1.id)

        outline = tree.render_outline()
        assert "DEAD [lumina_search]:" in outline
        assert "first" in outline  # request summary still visible
        # Dead node should NOT have response summary
        lines = outline.split("\n")
        dead_lines = [l for l in lines if "DEAD" in l]
        assert len(dead_lines) == 1
        assert "->" not in dead_lines[0]

    def test_render_with_refs(self) -> None:
        tree = TreeManager()
        tree.add_root("question")
        model = tree.add_model_node("searching", parent="n1")
        tool = tree.add_tool_node(parent=model.id, tool_name="lumina_search", request={"q": "test"})

        from dataclasses import dataclass as dc

        @dc
        class FakeRef:
            page_context: dict = None
            title: str = ""
            url: str | None = None
            source: str = "search"
            links: list | None = None
            raw_content: str | None = None

        refs = [
            ("r0", FakeRef(page_context={"action": "search", "id": 0}, title="Result 1")),
        ]
        tree.complete_tool_node(tool.id, result={"results": [{"title": "Result 1"}]}, indexed_refs=refs)

        outline = tree.render_outline()
        assert "refs:" in outline
        assert "r0" in outline
        assert '"action":"search"' in outline


class TestSerialization:
    def test_save_and_load(self, tmp_path) -> None:
        tree = TreeManager()
        tree.add_root("test question")
        model = tree.add_model_node("reasoning", parent="n1")
        tree.add_final_node("42", parent=model.id, confidence="high")

        path = tmp_path / "tree.json"
        tree.save(path)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert "root" in data
        assert "nodes" in data
        assert len(data["nodes"]) == 3  # root, model, final
