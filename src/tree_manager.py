"""Execution tree data structure and rendering.

Manages the in-memory tree of model reasoning steps and tool invocations.
Renders the tree as a compact markdown outline for prompt injection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from .summarizer import truncate, summarize_tool_request, summarize_tool_response, build_refs_summary


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    id: str
    kind: Literal["root", "model", "tool", "final"]
    title: str
    parent_id: str | None
    children: list[str] = field(default_factory=list)
    summary: str = ""
    dead: bool = False


@dataclass
class RootNode(TreeNode):
    """kind='root'. One per execution."""
    query: str = ""


@dataclass
class ModelNode(TreeNode):
    """kind='model'. Represents a model reasoning step."""
    assistant_text: str = ""
    action_decided: str | None = None


@dataclass
class ToolNode(TreeNode):
    """kind='tool'. Represents a tool invocation and its result."""
    tool_name: str = ""
    request: dict[str, Any] = field(default_factory=dict)
    request_summary: str = ""
    response_text: str = ""
    response_summary: str = ""
    page_refs_summary: str | None = None
    status: str = "pending"
    tool_state: dict | None = None


@dataclass
class FinalNode(TreeNode):
    """kind='final'. The answer."""
    assistant_text: str = ""
    confidence: str | None = None


# ---------------------------------------------------------------------------
# TreeManager
# ---------------------------------------------------------------------------

class TreeManager:
    """In-memory execution tree with rendering and dead-end logic."""

    def __init__(self) -> None:
        self.nodes: dict[str, TreeNode] = {}
        self.root_id: str = "n1"
        self.current_leaf_id: str = "n1"
        self._counter: int = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"n{self._counter}"

    # -- Build --

    def add_root(self, query: str) -> RootNode:
        node = RootNode(
            id=self._next_id(),
            kind="root",
            title="ROOT",
            parent_id=None,
            summary=truncate(query, 180),
            query=query,
        )
        self.nodes[node.id] = node
        self.root_id = node.id
        self.current_leaf_id = node.id
        return node

    def add_model_node(self, reasoning_text: str, parent: str) -> ModelNode:
        node = ModelNode(
            id=self._next_id(),
            kind="model",
            title="model",
            parent_id=parent,
            summary=truncate(reasoning_text, 180),
            assistant_text=reasoning_text,
            action_decided=None,
        )
        self.nodes[node.id] = node
        self.nodes[parent].children.append(node.id)
        return node

    def add_model_node_after_tools(
        self,
        reasoning_text: str,
        tool_node_ids: list[str],
    ) -> ModelNode:
        """Add a model node after tool(s) complete.

        For a single tool: always parent under it.
        For parallel tools: parse ``branch: <id>`` from reasoning text.
        """
        if len(tool_node_ids) == 1:
            parent_id = tool_node_ids[0]
        else:
            branch_id = self._parse_branch_directive(reasoning_text)
            if branch_id and branch_id in tool_node_ids:
                parent_id = branch_id
                for tid in tool_node_ids:
                    if tid != branch_id:
                        self._mark_subtree_dead(tid)
            elif branch_id == "none":
                parent_id = self.nodes[tool_node_ids[0]].parent_id
                assert parent_id is not None
                for tid in tool_node_ids:
                    self._mark_subtree_dead(tid)
            else:
                parent_id = tool_node_ids[0]

        node = ModelNode(
            id=self._next_id(),
            kind="model",
            title="model",
            parent_id=parent_id,
            summary=truncate(reasoning_text, 180),
            assistant_text=reasoning_text,
            action_decided=None,
        )
        self.nodes[node.id] = node
        self.nodes[parent_id].children.append(node.id)
        return node

    def add_tool_node(
        self,
        parent: str,
        tool_name: str,
        request: dict[str, Any],
    ) -> ToolNode:
        node = ToolNode(
            id=self._next_id(),
            kind="tool",
            title=tool_name,
            parent_id=parent,
            tool_name=tool_name,
            request=request,
            request_summary=summarize_tool_request(tool_name, request),
            status="pending",
        )
        self.nodes[node.id] = node
        self.nodes[parent].children.append(node.id)
        return node

    def complete_tool_node(
        self,
        node_id: str,
        result: dict[str, Any],
        indexed_refs: list[tuple[str, Any]] | None = None,
        status: str = "success",
        tool_state: dict | None = None,
    ) -> None:
        node = self.nodes[node_id]
        assert isinstance(node, ToolNode)
        response_text = self._extract_text_from_result(result)
        node.response_text = response_text
        node.response_summary = summarize_tool_response(node.tool_name, response_text)
        node.page_refs_summary = build_refs_summary(indexed_refs or [])
        node.status = status
        node.tool_state = tool_state

    def add_final_node(
        self,
        answer_text: str,
        parent: str,
        confidence: str | None = None,
    ) -> FinalNode:
        node = FinalNode(
            id=self._next_id(),
            kind="final",
            title="ANSWER",
            parent_id=parent,
            summary=truncate(answer_text, 220),
            assistant_text=answer_text,
            confidence=confidence,
        )
        self.nodes[node.id] = node
        self.nodes[parent].children.append(node.id)
        return node

    # -- Dead-end logic --

    def _parse_branch_directive(self, text: str) -> str | None:
        first_line = text.strip().split("\n")[0].strip().lower()
        if first_line.startswith("branch:"):
            return first_line.split(":", 1)[1].strip()
        return None

    def _mark_subtree_dead(self, node_id: str) -> None:
        node = self.nodes[node_id]
        node.dead = True
        for child_id in node.children:
            self._mark_subtree_dead(child_id)

    # -- Rendering --

    def render_outline(self, node_id: str | None = None, depth: int = 0) -> str:
        """Render the tree as an indented markdown outline."""
        if node_id is None:
            node_id = self.root_id
        lines = self._render_outline_lines(node_id, depth)
        return "\n".join(lines)

    def _render_outline_lines(self, node_id: str, depth: int) -> list[str]:
        node = self.nodes[node_id]
        indent = "  " * depth
        lines: list[str] = []

        # Dead nodes: collapse to a single line
        if node.dead:
            if node.kind == "tool":
                assert isinstance(node, ToolNode)
                lines.append(f"{indent}- DEAD [{node.tool_name}]: {node.request_summary}")
            return lines

        if node.kind == "root":
            assert isinstance(node, RootNode)
            lines.append(f"{indent}- ROOT: {node.summary}")
        elif node.kind == "model":
            assert isinstance(node, ModelNode)
            lines.append(f"{indent}- MODEL [{node.id}]: {truncate(node.summary, 180)}")
        elif node.kind == "tool":
            assert isinstance(node, ToolNode)
            lines.append(f"{indent}- TOOL [{node.tool_name}] [{node.id}]: {node.request_summary}")
            lines.append(f"{indent}  -> {node.response_summary}")
            if node.page_refs_summary:
                lines.append(f"{indent}  refs: {node.page_refs_summary}")
        elif node.kind == "final":
            assert isinstance(node, FinalNode)
            lines.append(f"{indent}- ANSWER: {truncate(node.summary, 220)}")

        for child_id in node.children:
            lines.extend(self._render_outline_lines(child_id, depth + 1))

        return lines

    # -- Serialization --

    def save(self, path: str | Any) -> None:
        """Save the tree as JSON."""
        from pathlib import Path as P
        p = P(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self._to_serializable()
        p.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")

    def _to_serializable(self) -> dict[str, Any]:
        nodes_list = []
        for node in self.nodes.values():
            d: dict[str, Any] = {
                "id": node.id,
                "kind": node.kind,
                "title": node.title,
                "parent_id": node.parent_id,
                "children": node.children,
                "summary": node.summary,
                "dead": node.dead,
            }
            if isinstance(node, RootNode):
                d["query"] = node.query
            elif isinstance(node, ModelNode):
                d["assistant_text"] = node.assistant_text
                d["action_decided"] = node.action_decided
            elif isinstance(node, ToolNode):
                d["tool_name"] = node.tool_name
                d["request"] = node.request
                d["request_summary"] = node.request_summary
                d["response_text"] = node.response_text
                d["response_summary"] = node.response_summary
                d["page_refs_summary"] = node.page_refs_summary
                d["status"] = node.status
            elif isinstance(node, FinalNode):
                d["assistant_text"] = node.assistant_text
                d["confidence"] = node.confidence
            nodes_list.append(d)

        root_data = self._node_to_dict(self.nodes[self.root_id])
        return {"root": root_data, "nodes": nodes_list}

    def _node_to_dict(self, node: TreeNode) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": node.id,
            "kind": node.kind,
            "title": node.title,
            "summary": node.summary,
        }
        if isinstance(node, ToolNode):
            d["request_summary"] = node.request_summary
            d["response_summary"] = node.response_summary
        return d

    # -- Helpers --

    @staticmethod
    def _extract_text_from_result(result: dict[str, Any]) -> str:
        """Extract text content from an MCP tool result dict or raw API response."""
        # MCP CallToolResult-style: {"content": [{"type": "text", "text": "..."}]}
        content = result.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            if parts:
                return "\n".join(parts)
        # Raw Lumina response: just serialize the whole thing
        return json.dumps(result, ensure_ascii=True)
