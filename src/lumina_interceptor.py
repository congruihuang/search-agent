"""Intercepts Lumina MCP tool calls to manage toolState and page_context.

Sits between the orchestrator's tool execution and the actual Lumina MCP server.
For each call:
  1. Injects the latest tool_state
  2. For lumina_search: forces searchResultContentType=2
  3. Forwards to the Lumina MCP server via stdio transport
  4. Captures updated tool_state from response
  5. Indexes page_context references for later calls
  6. Returns (result_dict, indexed_refs)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class PageRef:
    """Tracks a page_context and its source for later open/find/save calls."""
    page_context: dict | None = None
    url: str | None = None
    title: str | None = None
    source: str = "search"          # "search" | "open"
    links: list[dict] | None = None
    raw_content: str | None = None


class LuminaInterceptor:
    """Intercepts Lumina tool calls: injects toolState, indexes page_context."""

    LUMINA_TOOLS = {"lumina_search", "lumina_open", "lumina_find"}

    def __init__(self, mcp_config: dict[str, Any]) -> None:
        self._mcp_config = mcp_config
        self._session: ClientSession | None = None
        self._stdio_ctx: Any = None
        self._session_ctx: Any = None
        self.tool_state: dict | None = None
        self.page_refs: dict[str, PageRef] = {}
        self._next_ref_counter: int = 0

    def is_lumina_tool(self, tool_name: str) -> bool:
        return tool_name in self.LUMINA_TOOLS

    async def connect(self) -> list[dict[str, Any]]:
        """Connect to the Lumina MCP server and return tool schemas."""
        cfg = self._mcp_config
        params = StdioServerParameters(
            command=cfg["command"],
            args=cfg.get("args", []),
            env=cfg.get("env"),
        )
        self._stdio_ctx = stdio_client(params)
        read, write = await self._stdio_ctx.__aenter__()
        self._session_ctx = ClientSession(read, write)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()

        # Return tool schemas in Anthropic API format
        tools_response = await self._session.list_tools()
        schemas: list[dict[str, Any]] = []
        for tool in tools_response.tools:
            schema = {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            schemas.append(schema)
        return schemas

    async def disconnect(self) -> None:
        if self._session_ctx:
            await self._session_ctx.__aexit__(None, None, None)
        if self._stdio_ctx:
            await self._stdio_ctx.__aexit__(None, None, None)

    async def intercept(
        self, tool_name: str, args: dict[str, Any]
    ) -> tuple[dict[str, Any], list[tuple[str, PageRef]]]:
        """Intercept a Lumina tool call: inject state, forward, capture state."""
        assert self._session is not None, "Not connected"

        # 1. Inject query-scoped tool_state (omit if None — first call has no state)
        if self.tool_state is not None:
            args["tool_state"] = self.tool_state
        else:
            args.pop("tool_state", None)

        # 2. Tool-specific injection
        if tool_name == "lumina_search":
            args["searchResultContentType"] = 2
        elif tool_name == "lumina_open":
            if args.get("page_context") and args.get("url"):
                raise ValueError("lumina_open: provide page_context or url, not both")

        # 3. Forward to Lumina MCP server
        result = await self._session.call_tool(tool_name, arguments=args)

        # 4. Parse the response - MCP returns CallToolResult with content list
        result_dict = self._parse_call_result(result)

        # 5. Capture updated tool_state
        new_ts = result_dict.get("toolState")
        if new_ts is not None:
            self.tool_state = new_ts

        # 6. Index page references
        indexed_refs: list[tuple[str, PageRef]] = []
        if tool_name == "lumina_search":
            indexed_refs = self._index_search_results(result_dict)
        elif tool_name == "lumina_open":
            indexed_refs = self._index_opened_pages(result_dict)

        return result_dict, indexed_refs

    def _next_ref_key(self, prefix: str) -> str:
        key = f"{prefix}{self._next_ref_counter}"
        self._next_ref_counter += 1
        return key

    def _index_search_results(self, result: dict[str, Any]) -> list[tuple[str, PageRef]]:
        refs: list[tuple[str, PageRef]] = []
        for r in result.get("results", []):
            if not isinstance(r, dict):
                continue
            key = self._next_ref_key("r")
            ref = PageRef(
                page_context=r.get("pageContext"),
                url=r.get("url"),
                title=r.get("title"),
                source="search",
            )
            self.page_refs[key] = ref
            refs.append((key, ref))
        return refs

    def _index_opened_pages(self, result: dict[str, Any]) -> list[tuple[str, PageRef]]:
        refs: list[tuple[str, PageRef]] = []
        for page in result.get("pages", []):
            if not isinstance(page, dict):
                continue
            key = self._next_ref_key("p")
            structured = page.get("structuredDocument") or {}
            ref = PageRef(
                page_context=page.get("pageContext"),
                url=page.get("url"),
                title=page.get("title"),
                source="open",
                links=structured.get("links"),
                raw_content=page.get("content"),
            )
            self.page_refs[key] = ref
            refs.append((key, ref))
        return refs

    @staticmethod
    def _parse_call_result(result: Any) -> dict[str, Any]:
        """Parse MCP CallToolResult into a dict.

        The MCP SDK returns a CallToolResult object. Its content is a list of
        content blocks. For Lumina tools, the response is typically a single
        TextContent block containing JSON.
        """
        # Handle the structured content first (if available)
        if hasattr(result, "structuredContent") and result.structuredContent:
            return result.structuredContent

        # Fall back to parsing text content
        if hasattr(result, "content"):
            for block in result.content:
                if hasattr(block, "text"):
                    try:
                        return json.loads(block.text)
                    except json.JSONDecodeError:
                        return {"content": [{"type": "text", "text": block.text}]}
        return {"content": [{"type": "text", "text": str(result)}]}
