# Search Agent - Design Specification

## 1. Overview

Build an autonomous web-research agent that answers complex multi-hop questions by iteratively searching, opening, and extracting information from web pages. The agent uses the **Claude Agent SDK (Python)** to drive Claude, the **Lumina Web Skill** ([`SKILL.md`](./SKILL.md)) for web access, and a **self-managed execution tree** to give the model full visibility into its own reasoning history at every step.

> **Lumina Web Skill**: See [`SKILL.md`](./SKILL.md) for the complete MCP tool API (`lumina_search`, `lumina_open`, `lumina_find`) - parameters, `page_context`/`id` semantics, `toolState` threading rules, orchestration patterns, and decision logic. Claude calls these tools directly; the orchestrator intercepts calls to manage `toolState` transparently.

### Key Design Principles

1. **Tree-structured context management** - Rather than relying on the SDK's default conversation history (which grows linearly and quickly exhausts the context window), we manage a compact Tree representation of all prior execution steps. The model sees a rendered tree each turn, not a raw message log.
2. **Model-driven iteration** - The agent loop gives Claude the current tree and asks it to decide the next action (search/open/find/answer). Claude never sees raw tool-result payloads in its context more than once; they are summarized into tree nodes.
3. **Observability by default** - Every execution produces `stream_tree.json`, `stream_tree.html`, and `stream_tree.md` artifacts using the existing `build_stream_tree.py`, so runs are fully inspectable after the fact.

---

## 2. Architecture

```
+---------------------------------------------------------+
|                    search_agent.py                       |
|                   (Orchestrator)                         |
|                                                         |
|  +------------+   +--------------+   +---------------+ |
|  | Tree       |   | Claude Agent  |   | Lumina Web    | |
|  | Manager    |<--| SDK Client   |-->| Skill         | |
|  |            |   | (query loop) |   | (search/open/ | |
|  | build/     |   |              |   |  find)        | |
|  | render/    |   +--------------+   +---------------+ |
|  | serialize  |                                         |
|  +------------+                                         |
|        |                                                |
|        v                                                |
|  +----------------------------------------------------+ |
|  |            Execution Tree (in-memory)               | |
|  |                                                     | |
|  |  Root                                               | |
|  |   +- Model Node (reasoning)                         | |
|  |       +- Tool Node (lumina_search: q="...")          | |
|  |       |   +- Model Node (analysis)                  | |
|  |       |       +- Tool Node (lumina_open: ...)        | |
|  |       |       +- Tool Node (lumina_find: ...)        | |
|  |       +- Tool Node (lumina_search: q="...")          | |
|  |           +- Model Node (pivot reasoning)           | |
|  |               +- ...                                | |
|  +----------------------------------------------------+ |
+---------------------------------------------------------+
```

### Components

| Component | Responsibility |
|-----------|---------------|
| **Orchestrator** (`search_agent.py`) | Drives the iteration loop. Each iteration: (1) render tree -> (2) call Claude -> (3) execute tool if requested -> (4) update tree -> (5) check termination. |
| **Tree Manager** (`tree_manager.py`) | In-memory tree data structure. Add model/tool nodes, render to markdown for prompt injection, serialize to JSON. |
| **Claude Agent SDK** | Makes API calls to Claude. We use `query()` (stateless) per iteration - **not** `ClaudeSDKClient` - because we manage context ourselves. |
| **Lumina Web Skill** | External process providing `lumina_search`, `lumina_open`, `lumina_find`. Connected via stdio MCP config. See [`SKILL.md`](./SKILL.md). |
| **Tree Builder** (`build_stream_tree.py`) | Post-hoc visualization. Unchanged from current code. Runs on `stream.jsonl` after execution completes. |

---

## 3. Execution Tree - Data Model

### 3.1 Node Types

```python
@dataclass
class TreeNode:
    id: str                      # e.g. "n1", "n2", ...
    kind: Literal["root", "model", "tool", "final"]
    title: str                   # Display name
    parent_id: str | None        # Parent node ID (None for root)
    children: list[str]          # Child node IDs
    summary: str                 # Short text for tree rendering (<=180 chars)
    dead: bool = False           # True if this node is part of a dead-end branch

@dataclass
class RootNode(TreeNode):
    """kind='root'. One per execution."""
    query: str                   # The original user question

@dataclass
class ModelNode(TreeNode):
    """kind='model'. Represents a model reasoning step."""
    assistant_text: str          # Full model output text
    action_decided: str | None   # "lumina_search" | "lumina_open" | "lumina_find" | "submit_answer" | None

@dataclass
class ToolNode(TreeNode):
    """kind='tool'. Represents a tool invocation and its result."""
    tool_name: str               # "lumina_search" | "lumina_open" | "lumina_find" | "save_page" | "grep_file" | etc.
    request: dict[str, Any]      # Tool input parameters
    request_summary: str         # e.g. query string, URL
    response_text: str           # Raw response (stored, not sent to model)
    response_summary: str        # Compact summary (<=180 chars)
    page_refs_summary: str | None  # Compact Claude-visible `page_context` handles for later open/find calls
    status: str                  # "success" | "error" | "pending"
    tool_state: dict | None      # Query-scoped Lumina toolState snapshot after this tool completes

@dataclass
class FinalNode(TreeNode):
    """kind='final'. The answer."""
    assistant_text: str
    confidence: str | None       # "high" | "medium" | "low"
```

**TreeManager state**: In addition to the `nodes` dict, the `TreeManager` maintains:
- `root_id: str` - always `"n1"`.
- `current_leaf_id: str` - the ID of the most recently added model node that had no tool calls. Used as the default parent for the next model node when there are no pending tool results. Initialized to `root_id` in `add_root()`, updated in the main loop when a model node produces no tools (Section 4.2 line "tree.current_leaf_id = model_node.id"). When tools complete, `add_model_node_after_tools()` handles parent selection instead.

### 3.2 Dead-End Detection

When the model decides a branch was not useful ("None of these fit", "Let me try a different approach"), the tool nodes from that branch are **dead ends**. They produced no useful information, but we still need to remember they were tried to avoid repeating them.

**Dead-end detection is determined by the orchestrator's parent selection logic after each tool batch.** The rules depend on whether one or many tools completed:

**Single tool completed** (most common):
- The model node always becomes a **child** of the tool node. The tool is always **alive** - if the model wants to pivot, it does so by issuing a different search from within that child. We don't mark a single tool dead immediately because even "no useful results" informs the next action.
- Exception: if the model calls `submit_answer`, the branch is complete and no marking is needed.

**Multiple parallel tools completed** (siblings under the same model parent):
- The model must indicate which result it finds most useful. It does this by including a structured **`branch: <node_id>`** prefix in its reasoning text (e.g. `branch: n13`).
- `branch: n13` -> model node becomes child of n13, other siblings are marked dead.
- `branch: none` -> model node becomes sibling of all tools, all are marked dead (full pivot).
- If the model omits the prefix, default to the first tool node (no siblings marked dead - conservative fallback).

**Design choice: single surviving sibling.** The `branch:` directive selects exactly one tool node to continue from. We intentionally do not support "keep multiple siblings alive" because: (a) multi-hop research questions almost always narrow to one lead per branch point, and (b) supporting partial dead-marking (e.g. "keep 2 of 3") adds parsing complexity for marginal benefit. If the model genuinely needs information from multiple siblings, it can omit the `branch:` directive entirely - the conservative fallback keeps all siblings alive with their `refs:` lines visible.

This keeps `page_context` (a Lumina protocol concept for "which page to interact with") completely separate from `branch` (an orchestrator concept for "which tree branch to pursue"). The model can continue a branch by doing a fresh search or by opening a page - those are independent decisions.

When a node is marked dead:
1. The node and all its descendants are flagged `dead=True`.
2. On the next render, dead nodes collapse to a single line showing only the request summary - no response details, no children.
3. This frees token budget for active branches while still preventing the model from repeating tried queries.

```python
def add_model_node_after_tools(self, reasoning_text: str,
                                tool_node_ids: list[str]) -> ModelNode:
    """Add a model node after tool(s) complete. Parent selection determines
    whether tools are dead ends.

    For a single tool: always parent under it (no dead-end marking).
    For parallel tools: parse `branch: <id>` from reasoning text.
    """
    if len(tool_node_ids) == 1:
        # Single tool: always continue from it
        parent_id = tool_node_ids[0]
    else:
        # Multiple siblings: parse branch directive
        branch_id = self._parse_branch_directive(reasoning_text)
        if branch_id and branch_id in tool_node_ids:
            parent_id = branch_id
            for tool_id in tool_node_ids:
                if tool_id != branch_id:
                    self._mark_subtree_dead(tool_id)
        elif branch_id == "none":
            # Full pivot: parent is the tools' shared parent
            parent_id = self.nodes[tool_node_ids[0]].parent_id
            for tool_id in tool_node_ids:
                self._mark_subtree_dead(tool_id)
        else:
            # Fallback: parent under first tool, no marking
            parent_id = tool_node_ids[0]

    model_node = ModelNode(
        id=self._next_id(),
        kind="model",
        title="model",
        parent_id=parent_id,
        children=[],
        summary=truncate(reasoning_text, 180),
        assistant_text=reasoning_text,
        action_decided=None,
    )
    self.nodes[model_node.id] = model_node
    self.nodes[parent_id].children.append(model_node.id)

    return model_node

def _parse_branch_directive(self, text: str) -> str | None:
    """Parse 'branch: <node_id>' or 'branch: none' from first line of reasoning."""
    first_line = text.strip().split("\n")[0].strip().lower()
    if first_line.startswith("branch:"):
        value = first_line.split(":", 1)[1].strip()
        return value  # e.g. "n13" or "none"
    return None

def _mark_subtree_dead(self, node_id: str) -> None:
    """Recursively mark a node and all its descendants as dead."""
    node = self.nodes[node_id]
    node.dead = True
    for child_id in node.children:
        self._mark_subtree_dead(child_id)
```

### 3.3 Tree Structure Rules

1. The **root node** is always `n1`. It has the original query.
2. Model nodes are children of the previous tool node (if the model is reacting to a tool result) or of the root/previous model node (if the model is pivoting to a new strategy).
3. Tool nodes are children of the model node that requested them.
4. **Parallel tool calls are sibling nodes** under the same model parent. When the model requests multiple tools in one turn (e.g. 3 searches), all 3 tool nodes are children of that model node and are executed concurrently.
5. The tree is append-only during execution. Nodes are never deleted or modified (except `ToolNode.response_*` and `status` fields, which update when a tool completes).

### 3.4 Parent Selection Logic

When adding a new model node after tool(s) complete, the parent determines
whether tools are alive or dead:

```
After tool(s) complete, call model. Then:

Case 1: Single tool completed
  -> model node always becomes child of the tool (no dead-end marking)
  -> if model wants to pivot, it does so from within this child

Case 2: Multiple parallel tools completed (siblings under same model parent)
  Example: model issued 3 searches -> tool_A [n5], tool_B [n6], tool_C [n7]

  Model's reasoning starts with "branch: n6"
    -> parent = n6
    -> model becomes child of n6
    -> n5 and n7 are marked dead

  Model's reasoning starts with "branch: none"
    -> parent = tools' shared model parent
    -> model becomes sibling of n5/n6/n7
    -> all three tools are marked dead

  Model omits branch directive
    -> parent = n5 (first tool, conservative fallback)
    -> no siblings marked dead
```

**Why `branch:` is only needed for parallel tools**: With a single tool, there's
no ambiguity - the model always continues from it. Dead-end marking only matters
when the orchestrator needs to know which of several siblings to keep alive.

**Why `branch:` is separate from `page_context`**: `page_context` is a Lumina
protocol concept - it tells Lumina *which page* to open or search within.
`branch:` is an orchestrator concept - it tells the tree *which investigation
line* to pursue. The model might continue a branch by doing a fresh search
(no `page_context` from any sibling) or pivot while reusing a `page_context`
from an earlier, non-sibling node. These are independent decisions.

---

## 4. Iteration Loop

### 4.1 High-Level Flow

```
initialize(query)
while not done and iteration < max_iterations:
    1. Render tree to markdown outline (dead branches collapsed)
    2. Build prompt = system_prompt + tree_outline + action_request
    3. Call Claude via SDK query()
    4. Parse response: extract reasoning text + tool_use blocks + final_answer
    5. Add ModelNode to tree (parent depends on context - see Section 3.4)
    6. If final answer: add FinalNode, done = True
    7. If tool_use blocks:
         a. Add all ToolNodes as siblings under the model node
         b. Execute all tools in parallel (asyncio.gather)
         c. Complete each ToolNode with its result
         d. Go to step 1 (next iteration calls Claude with updated tree)
    8. Append to stream.jsonl for post-hoc visualization
```

Note: there is **no separate "post-tool analysis" call**. After tools execute,
the loop simply iterates - step 1 renders the updated tree (now showing tool
results), step 3 calls Claude, and step 5 uses `add_model_node_after_tools`
(Section 3.2) to place the model node under the correct parent based on the `branch:`
directive. This avoids the double-call bug where a second Claude response's
tool_calls would be dropped.

### 4.2 Detailed Pseudocode

```python
async def run_search_agent(query: str, config: AgentConfig) -> AgentResult:
    tree = TreeManager()
    tree.add_root(query)
    lumina = LuminaInterceptor(config.mcp_config)  # manages toolState
    stream_log = StreamLog(config.output_dir / "stream.jsonl")

    # Track which tool nodes just completed (empty on first iteration).
    # Used by add_model_node_after_tools for parent selection / dead-end marking.
    pending_tool_node_ids: list[str] = []

    for iteration in range(config.max_iterations):
        # 1. Render current tree as compact markdown
        tree_outline = tree.render_outline()

        # 2. Build the prompt for this iteration
        prompt = build_iteration_prompt(
            query=query,
            tree_outline=tree_outline,
            iteration=iteration,
            max_iterations=config.max_iterations,
        )

        # 3. Call Claude (single-shot, not conversational)
        response = await call_claude(prompt, config)
        stream_log.write_assistant(response)

        # 4. Parse response
        parsed = parse_response(response)
        # Returns: ParsedResponse(reasoning, tool_calls, final_answer, final_confidence)

        # 5. Add model node (parent depends on whether tools just completed)
        if pending_tool_node_ids:
            # Post-tool iteration: parent selection uses branch directive (Section 3.2)
            model_node = tree.add_model_node_after_tools(
                parsed.reasoning, pending_tool_node_ids
            )
            pending_tool_node_ids = []
        else:
            # First iteration or no prior tools: parent is last model node or root
            model_node = tree.add_model_node(
                parsed.reasoning, parent=tree.current_leaf_id
            )

        # 6. Check for final answer
        if parsed.final_answer:
            tree.add_final_node(
                parsed.final_answer,
                parent=model_node.id,
                confidence=parsed.final_confidence,
            )
            break

        # 7. Execute tools in parallel (all are siblings under model_node)
        if parsed.tool_calls:
            tool_nodes = []
            for tc in parsed.tool_calls:
                tool_node = tree.add_tool_node(
                    parent=model_node.id,
                    tool_name=tc.name,
                    request=tc.input,
                )
                tool_nodes.append((tool_node, tc))

            async def execute_one(tool_node, tc):
                tool_name = tc.name.split("__")[-1]
                if lumina.is_lumina_tool(tool_name):
                    result, indexed_refs = await lumina.intercept(tool_name, tc.input)
                else:
                    result = await custom_mcp_server.call_tool(tool_name, tc.input)
                    indexed_refs = []
                tree.complete_tool_node(tool_node.id, result, indexed_refs)
                stream_log.write_tool_result(tool_node.id, result)
                return tool_node.id

            pending_tool_node_ids = list(await asyncio.gather(
                *(execute_one(tn, tc) for tn, tc in tool_nodes)
            ))
            # Loop continues: next iteration renders updated tree, calls Claude,
            # and add_model_node_after_tools places the new model node using
            # branch directive for parent selection.
            continue

        # No tools and no answer: model produced only reasoning.
        # Update leaf pointer and continue.
        tree.current_leaf_id = model_node.id

    # Post-processing
    tree.save(config.output_dir / "stream_tree.json")
    build_stream_tree(config.output_dir / "stream.jsonl")
    return AgentResult(answer=parsed.final_answer, tree=tree, stats=...)
```

### 4.3 Tool Schema & MCP Integration

#### How Claude learns tool schemas

Claude discovers tool schemas automatically through the MCP protocol. The SDK connects to each MCP server, calls `tools/list`, and includes the returned JSON Schema definitions in the Claude API `tools` parameter. Claude never needs to see tool schemas in the system prompt.

Our setup registers **two MCP servers**:

| MCP Server | Tools | Schema source |
|-----------|-------|---------------|
| **`lumina-web`** (external process) | `lumina_search`, `lumina_open`, `lumina_find` | Defined in Lumina's `schemas.ts` (Zod -> JSON Schema). Schemas match [`SKILL.md`](./SKILL.md). |
| **`search_tools`** (in-process) | `save_page`, `grep_file`, `read_file`, `submit_answer` | Defined by us via SDK `@tool` decorators. |

Claude sees both sets of tools in its `tools` parameter and can call any of them.

#### `tool_state` / `page_context` management

The Lumina schemas (from `schemas.ts`) include both `tool_state` and `page_context`, but they are **not owned the same way**:

- **`tool_state` is fully orchestrator-owned opaque session state**. Claude should never manage it directly.
- **`page_context` is a small model-visible page handle**. Claude uses it to indicate *which prior result/page* it wants to open or search within.

The orchestrator intercepts every Lumina tool call and:

1. **Strips** `tool_state` from Claude-visible results.
2. **Preserves** exact but compact `page_context` snippets in Claude-visible summaries so it can choose the right result/page in later calls.
3. **Injects** the latest `tool_state` before forwarding the call to the Lumina MCP server.
4. **Captures** the updated `tool_state` and indexes returned `page_context` objects for later reference and rendering.

This means Claude calls the real Lumina tools with their real schemas, but only has to reason about the small, meaningful `page_context` values. The orchestrator owns the opaque `tool_state` lifecycle transparently.

```
Claude sees tool schema:          Orchestrator injects before forwarding:
+-----------------------+         +-----------------------------------+
| lumina_search          |         | lumina_search                     |
|   q: "climate change"  |   ->    |   q: "climate change"             |
|   topN: 10             |         |   topN: 10                        |
|                        |         |   searchResultContentType: 2      |
|                        |         |   tool_state: { ... latest ... }  |
+-----------------------+         +-----------------------------------+
```

#### Lumina tool schemas (from `schemas.ts`)

The Lumina MCP server defines these schemas using Zod. They are converted to JSON Schema by the MCP protocol and sent to Claude automatically:

**`lumina_search`** - Search the web using Bing.
```
Parameters:
  q: string (required)        - Search query text
  topN: int (optional)        - Maximum results to return
  recency: int (optional)     - Days for recency filter (0 = no filter)
  domains: string[] (optional) - Restrict to these domains
  language: string (optional)  - ISO language code (e.g. "en")
  countryCode: string (opt)    - ISO country code (e.g. "us")
  market: string (optional)    - Market locale (e.g. "en-US")
  searchResultContentType: int - Always 2 (snippets only). Default 2 in schema; orchestrator also forces it.
  tool_state: ToolState (opt)  - Opaque. Injected by orchestrator.
```

**`lumina_open`** - Open a page (from search result, URL, or followed link). Provide either `page_context` or `url`, not both.
```
Parameters:
  page_context: PageContextInfo (optional) - From prior search/open result
  id: int (optional)           - External link index (when page_context.action="view")
  url: string (optional)       - Direct URL (alternative to page_context; mutually exclusive)
  line_no: int (optional)      - Start line (default 0)
  num_lines: int (optional)    - Lines to return
  tool_state: ToolState (opt)  - Opaque. Injected by orchestrator.
```

**`lumina_find`** - Find text within an already-opened/searched page.
```
Parameters:
  page_context: PageContextInfo (required) - Page to search within
  id: int (optional)           - External link index
  pattern: string (required)   - Text pattern to find
  query_type: "pattern"|"semantic" (optional, default "pattern")
  tool_state: ToolState (required) - Opaque. Injected by orchestrator.
```

> **Key**: `tool_state` appears in JSON Schemas sent to Claude but is always injected by the orchestrator - Claude should omit it. `page_context` also appears in the schemas, but Claude must **provide it** for `lumina_open` (when opening a search result) and `lumina_find` (always required). The orchestrator does not override `page_context` - it's Claude's responsibility to pass the correct handle from the tree outline's `refs:` lines.

#### Custom tools (in-process MCP server)

These are tools we define ourselves, not part of Lumina. Registered via the SDK's `@tool` decorator and served from an in-process MCP server:

```python
from claude_agent_sdk import tool, create_sdk_mcp_server

@tool("save_page",
      "Save a page's full content to a local markdown file. "
      "Use when a page is too large to process inline. "
      "Pass the ref key from the tree outline (e.g. 'p0', 'r3').",
      {"ref_key": str, "filename": str})
async def save_page_tool(args):
    ref = _interceptor.page_refs.get(args['ref_key'])
    if not ref:
        return {"content": [{"type": "text", "text": f"Error: unknown ref key '{args['ref_key']}'"}]}
    if not ref.raw_content:
        return {"content": [{"type": "text", "text": f"Error: ref '{args['ref_key']}' has no saved content (open the page first)"}]}
    path = _config.output_dir / "pages" / f"{args['filename']}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ref.raw_content, encoding="utf-8")
    line_count = ref.raw_content.count("\n") + 1
    return {"content": [{"type": "text", "text": f"Saved {line_count} lines to {path}"}]}

@tool("grep_file",
      "Search a saved file for lines matching a regex pattern. "
      "Returns matching lines with line numbers.",
      {"filename": str, "pattern": str})
async def grep_file_tool(args):
    import re
    path = _config.output_dir / "pages" / f"{args['filename']}.md"
    if not path.exists():
        return {"content": [{"type": "text", "text": f"Error: file not found: {path}"}]}
    matches = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if re.search(args["pattern"], line, re.IGNORECASE):
            matches.append(f"{i}: {line}")
    result = "\n".join(matches[:50]) if matches else "No matches found."
    return {"content": [{"type": "text", "text": result}]}

@tool("read_file",
      "Read a range of lines from a saved file.",
      {"filename": str, "offset": int, "limit": int})
async def read_file_tool(args):
    path = _config.output_dir / "pages" / f"{args['filename']}.md"
    if not path.exists():
        return {"content": [{"type": "text", "text": f"Error: file not found: {path}"}]}
    lines = path.read_text(encoding="utf-8").splitlines()
    start = max(0, args.get("offset", 0))
    end = start + args.get("limit", 100)
    chunk = lines[start:end]
    result = "\n".join(f"{i+start+1}: {l}" for i, l in enumerate(chunk))
    return {"content": [{"type": "text", "text": result}]}

@tool("submit_answer", "Submit the final answer to the user's question.",
      {"answer": str, "confidence": str})
async def answer_tool(args):
    return {"content": [{"type": "text", "text": f"ANSWER: {args['answer']}"}]}

custom_server = create_sdk_mcp_server("search_tools", tools=[
    save_page_tool, grep_file_tool, read_file_tool, answer_tool,
])
```

### 4.4 Prompt Structure

Each iteration, Claude receives a single prompt (not a conversation):

```
<system>
You are a web research agent. Your task is to answer the user's question
by searching the web and analyzing pages.

## Tool Usage Guide

You have access to these tools (schemas provided via MCP protocol):

**Web search tools** (Lumina - schemas auto-discovered):
- `lumina_search`: Search the web. You provide `q` and optional `topN`.
  Do NOT set `tool_state` or `searchResultContentType` - these are
  managed automatically.
- `lumina_open`: Open a page. Pass `page_context` from a prior search
  result to open that result, OR pass `url` to open a page directly.
  **Provide either `page_context` or `url`, not both.**
  To follow an external link on an opened page, pass that page's
  `page_context` plus `id` (the link index).
  Do NOT set `tool_state` - managed automatically.
- `lumina_find`: Find text in a page. Pass `page_context` from a prior
  search or open result, plus `pattern` (exact) or set
  `query_type: "semantic"` for meaning-based search.
  Do NOT set `tool_state` - managed automatically.

**File tools** (for large pages):
- `save_page(ref_key, filename)`: Save a page to a local markdown file.
  Pass the ref key shown in the tree outline's `refs:` line (e.g. `p0`,
  `r3`). The page must have been opened first via `lumina_open`.
- `grep_file(filename, pattern)`: Search a saved file for regex matches.
  Returns matching lines with line numbers.
- `read_file(filename, offset, limit)`: Read a range of lines from a
  saved file.

**Answer tool**:
- `submit_answer(answer, confidence)`: Submit your final answer.

> **Important**: `tool_state` and `searchResultContentType` are injected
> automatically by the orchestrator. Never set these yourself.

## Execution Tree

Your research history is tracked as a **tree**, not a flat list.
This is different from a linear conversation - here is how to read it:

### How the tree works
- **Indentation = parent-child**. A child node was caused by its parent.
  A TOOL node is always a child of the MODEL node that requested it.
  A MODEL node after a TOOL is a child of that TOOL (reacting to its result).
- **Branching = parallel or sequential attempts** from the same reasoning step.
  If a MODEL node has multiple TOOL children, those were issued together.
- **Depth = chain of reasoning**. A deep path means a sustained line of
  investigation: search -> analyze -> open -> find -> conclude.
- **Sibling branches at the same depth = pivots**. When you tried one
  approach, it didn't work, and you pivoted to another.

### How to use the tree
- **Avoid redundancy**: scan all TOOL nodes to see what queries, URLs, and
  patterns have already been tried. Do NOT repeat them.
- **Follow promising branches deeper**: if a branch found a partial clue
  (e.g. a candidate show name), continue from that branch - open pages,
  search for more details, or find specific text.
- **Prune dead ends**: if a branch yielded no useful results (indicated by
  response summaries like "0 results" or irrelevant titles), do not
  continue down that path. Pivot to a new approach at a higher level.
- **Identify what is still missing**: compare the clues in the question
  against the evidence gathered across all branches. Target the gaps.

### Node types
- `ROOT`: The original question.
- `MODEL`: Your reasoning at that point - what you concluded and why.
- `TOOL [name]`: A tool call. Shows the request summary and a `->` line
  with the response summary (result count, top titles, errors, etc.).
  Active Lumina tool nodes also show a `refs:` line with reusable handles:
  - Each ref has a **key** (e.g. `r0`, `p3`), a **`page_context`** object
    (copy into `lumina_open`/`lumina_find`), and a **title**.
  - Use the key with `save_page(ref_key, filename)` to save the page.
  - Use the `page_context` with `lumina_open` or `lumina_find` to interact
    with that specific search result or page.
- `DEAD [name]`: A tool call from a branch you previously abandoned.
  Shows only the request summary - no response, no children.
  **Do NOT repeat these queries** - they have already been tried.
- `ANSWER`: Your final submitted answer.

### Current tree

{tree_outline_markdown}

## Instructions

- Study the tree above. Do NOT repeat searches you have already tried.
- Decide your next action. You may call one or more tools, or submit_answer.
- Explain your reasoning briefly before calling a tool.
- If you have enough evidence, call submit_answer.
- **After multiple parallel tools complete**, you will be called again to
  analyze results. Start your response with a branch directive:
  - `branch: <node_id>` - the most promising tool result you want to pursue.
    Other sibling results become dead ends.
  - `branch: none` - none of the results were useful; pivot to a new approach.
    All sibling results become dead ends.
  (When only one tool completed, no branch directive is needed.)
- Iteration {current}/{max}. Budget remaining: ~{tokens_remaining} tokens.
</system>

<user>
Question: {original_query}

Based on the execution tree above, what is your next action?
</user>
```

---

## 5. Tree Rendering for Prompt Injection

### 5.1 Compact Outline Format

The tree is rendered as an indented outline. Tool nodes include request/response summaries plus compact Claude-visible `page_context` handles for reusable search results or opened pages. Model nodes include truncated reasoning. **Dead-end branches are collapsed** to a single line showing only the request summary, saving tokens while still preventing the model from repeating them.

```
- ROOT: Question about undercover martial arts show...
  - MODEL [n2]: I'll research this step by step.
    - DEAD [lumina_search]: q="undercover martial arts technology show 2005-2012"
    - DEAD [lumina_search]: q="animated show undercover martial arts 2005-2012 web game 2016"
    - MODEL [n7]: Results are mostly echoes. Try specific show names.
      - DEAD [lumina_search]: q="Shuriken School TV show 2005 theme song"
      - DEAD [lumina_search]: q="Dragon Booster show 2005 web game"
      - MODEL [n12]: Reconsider: production company with resource shortage...
        - DEAD [lumina_search] [n13]: q="Aardman Animations TV shows 2005-2012"
        - TOOL [lumina_search] [n14]: q="Chop Socky Chooks web game theme song"
          -> 10 results | top: Chop Socky Chooks Wikipedia
          refs: r0 pc={"action":"search","id":"sr_0"} "Chop Socky Chooks Wikipedia"
          - MODEL [n15]: branch: n14. Found the show! Now find theme song lyrics.
            - TOOL [lumina_open] [n16]: Chop Socky Chooks Hero Songs Wiki
              -> 60 lines | Lyrics: "Ba-Da, Ba-Da..."
              refs: p0 pc={"action":"view","id":"pg_0"} "Chop Socky Chooks Hero Songs Wiki"
              - MODEL [n17]: Third line is "Kung-Fu chickens get there on the Double"
```

Compare with the non-pruned version: dead branches take ~1 line each instead of ~2-3 lines with full response summaries. For a 25-iteration run with ~40% dead branches, this saves ~30% of tree outline tokens.

### 5.2 Rendering Logic

```python
def render_outline(self, node_id: str, depth: int = 0) -> list[str]:
    node = self.nodes[node_id]
    indent = "  " * depth
    lines = []

    # Dead nodes: collapse to a single line (request only, no children)
    if node.dead:
        if node.kind == "tool":
            lines.append(f"{indent}- DEAD [{node.tool_name}]: {node.request_summary}")
        # Skip children of dead nodes - they are also dead
        return lines

    if node.kind == "root":
        lines.append(f"{indent}- ROOT: {node.summary}")
    elif node.kind == "model":
        lines.append(f"{indent}- MODEL [{node.id}]: {truncate(node.summary, 180)}")
    elif node.kind == "tool":
        lines.append(f"{indent}- TOOL [{node.tool_name}] [{node.id}]: {node.request_summary}")
        lines.append(f"{indent}  -> {node.response_summary}")
        if node.page_refs_summary:
            lines.append(f"{indent}  refs: {node.page_refs_summary}")
    elif node.kind == "final":
        lines.append(f"{indent}- ANSWER: {truncate(node.summary, 220)}")

    for child_id in node.children:
        lines.extend(self.render_outline(child_id, depth + 1))

    return lines
```

### 5.3 Summarization Budget

| Node type | Summary max length | Notes |
|-----------|-------------------|-------|
| Root | 180 chars | Original query, truncated |
| Model | 180 chars | First sentence of reasoning |
| Tool request | 140 chars | Query string or URL |
| Tool response | 160 chars | Result count + top titles |
| Page refs | 240 chars | Exact compact `page_context` handles for top 1-3 reusable results/pages |
| Final | 220 chars | Answer text |

Total tree outline size for a 25-iteration run with ~65 nodes plus compact page refs on active Lumina nodes ~= 12-18 KB of text ~= 3,500-5,000 tokens. This is still well within a single iteration budget even for a 200K context window model.

---

## 6. Lumina Tool State Management

> Full API reference: [`SKILL.md`](./SKILL.md)

The three Lumina MCP tools share a **`toolState`** session object that must be threaded between every call (SKILL.md Section ToolState - Session Continuity). The orchestrator owns this state and passes it through.

### 6.1 Key Concepts from SKILL.md

| Concept | Rule |
|---------|------|
| **`toolState` threading** | `toolState` is **query-scoped shared session state**. Across sequential calls, pass the latest `toolState`. For a parallel batch of sibling tools in the same query, pass the same current `toolState` snapshot to all of them. |
| **`searchResultContentType`** | **Must always be `2`** on every `lumina_search` call (snippets only). |
| **`lumina_open` exclusivity** | Provide either `page_context` or `url`, not both. |
| **`page_context.id` vs outer `id`** | `page_context.id` always identifies the page/result set itself. Outer `id` only applies when `page_context.action == "view"` - it specifies which external link on that page to follow (the `[[[link_N]]]` index). When `action == "search"`, outer `id` is not needed. |
| **`lumina_find` constraint** | Cannot be the first call - requires a `pageContext` from a prior `lumina_search` or `lumina_open`. |
| **Orchestration patterns** | 4 patterns from SKILL.md: search-first, open-first (direct URL), search->find (skip open), follow external links. Plus a decision logic flowchart (see SKILL.md Section Decision Logic). |

### 6.2 Tool Call Interceptor

Since Claude calls the real Lumina MCP tools directly, but the orchestrator owns `tool_state` and indexes returned `page_context` values for rendering/validation, we need a middleware layer that intercepts every tool call.

```python
@dataclass
class PageRef:
    """Tracks a page_context and its source for later open/find/save calls."""
    page_context: dict        # The pageContext object from a Lumina response
    url: str | None           # URL if known
    title: str | None         # Title if known
    source: str               # "search" | "open"
    links: list[dict] | None  # External links from structuredDocument (for follow-link)
    raw_content: str | None   # Full page content (for save_page)

class LuminaInterceptor:
    """Intercepts Lumina MCP tool calls to manage tool_state and page_context.

    Sits between Claude's tool_use requests and the actual Lumina MCP server.
    For each call:
      1. Injects the latest tool_state
      2. For lumina_search: forces searchResultContentType=2
      3. Forwards to Lumina MCP server
      4. Captures updated tool_state from response
      5. Indexes page_context references for later calls
      6. Returns summarized result (not raw response)
    """

    LUMINA_TOOLS = {"lumina_search", "lumina_open", "lumina_find"}

    def __init__(self, mcp_transport):
        self.mcp = mcp_transport
        self.tool_state: dict | None = None  # shared per top-level query/run
        self.page_refs: dict[str, PageRef] = {}  # stable key -> PageRef (monotonic)
        self._next_ref_counter = 0                # monotonically increasing

    def is_lumina_tool(self, tool_name: str) -> bool:
        """Check if a tool call should be intercepted."""
        return tool_name in self.LUMINA_TOOLS

    async def intercept(self, tool_name: str, args: dict) -> dict:
        """Intercept a Lumina tool call: inject state, forward, capture state."""

        # 1. Inject query-scoped tool_state (override whatever Claude sent, if anything)
        args["tool_state"] = self.tool_state

        # 2. Tool-specific injection
        if tool_name == "lumina_search":
            args["searchResultContentType"] = 2  # always snippets
        elif tool_name == "lumina_open":
            # Enforce mutual exclusion: page_context or url, not both
            if args.get("page_context") and args.get("url"):
                raise ValueError("lumina_open: provide page_context or url, not both")
        # For lumina_open/lumina_find: page_context is passed by Claude
        # from the schema - Claude learns which page_context to use from
        # the tool result summaries in the tree outline.

        # 3. Forward to Lumina MCP server
        result = await self.mcp.call_tool(tool_name, args)

        # 4. Capture updated tool_state
        self.tool_state = result.get("toolState")

        # 5. Index references for later use
        indexed_refs = []
        if tool_name == "lumina_search":
            indexed_refs = self._index_search_results(result)
        elif tool_name == "lumina_open":
            indexed_refs = self._index_opened_pages(result)

        return result, indexed_refs  # caller passes indexed_refs to tree node

    def _next_ref_key(self, prefix: str) -> str:
        """Generate a stable, monotonically-increasing reference key."""
        key = f"{prefix}{self._next_ref_counter}"
        self._next_ref_counter += 1
        return key

    def _index_search_results(self, result: dict) -> list[tuple[str, PageRef]]:
        """Index search results with stable keys. Returns list of (key, PageRef)."""
        refs = []
        for r in result.get("results", []):
            key = self._next_ref_key("r")
            ref = PageRef(
                page_context=r.get("pageContext"),
                url=r.get("url"),
                title=r.get("title"),
                source="search",
                links=None,
                raw_content=None,
            )
            self.page_refs[key] = ref
            refs.append((key, ref))
        return refs

    def _index_opened_pages(self, result: dict) -> list[tuple[str, PageRef]]:
        """Index opened pages with stable keys. Returns list of (key, PageRef)."""
        refs = []
        for page in result.get("pages", []):
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
```

**Parallel batch semantics**:

- `toolState` is shared for the entire top-level query/run.
- If Claude issues multiple Lumina calls in one turn, all sibling calls receive the same current `toolState` snapshot.
- After the batch completes, the orchestrator keeps the last non-null returned `toolState` as the query's current session state.
- This relies on the Lumina contract that `toolState` values returned within one query represent the same logical shared session, not independent branches.

#### How `page_context` flows

Unlike `tool_state` (which is fully opaque), `page_context` is **semi-visible** to Claude. Claude sees `page_context` objects in tool results and must pass the correct one back when calling `lumina_open` or `lumina_find`. This is how the model selects *which* search result to open or *which* page to search within.

The flow:
1. Claude calls `lumina_search(q="...")` -> result includes `pageContext` per search result
2. The orchestrator renders exact compact `page_context` snippets into the tree outline (for example `pc={"action":"search","id":"sr_0"}`)
3. Claude copies one of those `page_context` objects into `lumina_open(...)` or `lumina_find(...)`
4. The orchestrator intercepts, injects `tool_state`, and forwards

> **Important**: Because each iteration is stateless, the tree outline must preserve exact reusable `page_context` handles for active branches. `response_summary` alone is not sufficient.

> **Note**: `page_context` is small (~100 bytes) and meaningful to the model's choice of which page to interact with. Unlike `tool_state` (which is large and opaque), `page_context` is intentionally visible to Claude.

### 6.3 toolState on Tree Nodes

The latest `toolState` snapshot is stored on each `ToolNode` so the tree can be replayed or resumed from any point:

```python
tool_node = tree.add_tool_node(
    parent=model_node.id,
    tool_name="lumina_search",
    request={"q": "Aardman Animations TV shows"},
)
# After execution:
result, indexed_refs = await lumina.intercept("lumina_search", {...})
tree.complete_tool_node(
    tool_node.id,
    result=result,
    indexed_refs=indexed_refs,  # e.g. [("r5", PageRef(...)), ("r6", PageRef(...))]
    status="success",
    tool_state=lumina.tool_state,  # snapshot for replay
)
# complete_tool_node generates:
#   response_summary = summarize_tool_result(result)  # "10 results | top: ..."
#   page_refs_summary = build_refs_summary(indexed_refs)
#     e.g. 'r5 pc={"action":"search","id":0} "List of Aardman TV series"'
```

The `build_refs_summary` function formats the top 1-3 refs into a compact
string for the tree outline. Each ref shows its stable key, the `page_context`
object (which Claude can copy into `lumina_open`/`lumina_find` calls), and a
truncated title:

```python
import json

def build_refs_summary(indexed_refs: list[tuple[str, PageRef]],
                       max_refs: int = 3) -> str | None:
    """Build compact page_context handles for the tree outline.

    Uses json.dumps for page_context so Claude can copy the value directly
    into lumina_open/lumina_find tool calls (valid JSON, not Python repr).
    """
    if not indexed_refs:
        return None
    parts = []
    for key, ref in indexed_refs[:max_refs]:
        pc_json = json.dumps(ref.page_context or {}, separators=(",", ":"))
        title = truncate(ref.title or "", 50)
        parts.append(f'{key} pc={pc_json} "{title}"')
    return " | ".join(parts)
```

---

## 7. Configuration

```python
@dataclass
class AgentConfig:
    # Model
    model: str = "claude-sonnet-4-6"
    max_thinking_tokens: int | None = None

    # Iteration limits
    max_iterations: int = 30
    max_budget_usd: float = 5.0
    max_total_tokens: int = 500_000

    # Lumina MCP
    mcp_config_path: Path = Path(".mcp.json")

    # Output
    output_dir: Path = Path("./output")

    # Prompt
    system_prompt_path: Path | None = None  # Override default system prompt

    # Tree rendering
    tree_summary_max_length: int = 180
    tool_response_summary_max_length: int = 160
```

---

## 8. File Structure

```
search-agent/
+-- DESIGN.md                    # This document
+-- SKILL.md                     # Lumina Web MCP tool reference (canonical)
+-- .mcp.json                    # Lumina MCP server config (stdio)
+-- src/
|   +-- __init__.py
|   +-- search_agent.py          # Main orchestrator + CLI entry point
|   +-- tree_manager.py          # ExecutionTree data structure + rendering
|   +-- lumina_interceptor.py    # Intercepts Lumina tool calls, manages toolState (implements SKILL.md patterns)
|   +-- prompt_builder.py        # Builds iteration prompts (tree outline + behavioral guidance)
|   +-- stream_log.py            # Writes stream.jsonl for post-hoc viz
|   +-- summarizer.py            # Summarize tool responses (from build_stream_tree.py)
|   +-- config.py                # AgentConfig dataclass
+-- sample_query/                # Existing sample data
|   +-- build_stream_tree.py
|   +-- stream.jsonl
|   +-- stream_tree.html
|   +-- ...
+-- pyproject.toml               # Dependencies: claude-agent-sdk, etc.
+-- tests/
    +-- test_tree_manager.py
    +-- test_prompt_builder.py
    +-- test_summarizer.py
```

---

## 9. Claude Agent SDK Integration Details

### 9.1 Per-Iteration Call (Stateless `query()`)

Each iteration is a standalone `query()` call. We do **not** use multi-turn conversation because:
- We control exactly what context Claude sees (the tree outline, not raw history).
- Token usage is bounded per iteration (~4K-8K input tokens for tree + prompt, not accumulating).
- We avoid the 200K context ceiling that the sample query hit ($30 for one question).

```python
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ToolUseBlock, TextBlock

async def call_claude(prompt: str, config: AgentConfig) -> ParsedResponse:
    options = ClaudeAgentOptions(
        model=config.model,
        permission_mode="bypassPermissions",
        max_turns=1,  # Single turn - we handle the loop ourselves
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=[
            # Lumina tools (from external MCP server, schemas from schemas.ts)
            "mcp__lumina-web__lumina_search",
            "mcp__lumina-web__lumina_open",
            "mcp__lumina-web__lumina_find",
            # Custom tools (from in-process MCP server)
            "mcp__search_tools__save_page",
            "mcp__search_tools__grep_file",
            "mcp__search_tools__read_file",
            "mcp__search_tools__submit_answer",
        ],
        mcp_servers={
            "lumina-web": lumina_mcp_server,    # external process (from .mcp.json)
            "search_tools": custom_mcp_server,   # in-process (our @tool defs)
        },
    )

    reasoning_parts = []
    tool_calls = []
    final_answer = None
    final_confidence = None

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    reasoning_parts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    if block.name.endswith("submit_answer"):
                        final_answer = block.input.get("answer")
                        final_confidence = block.input.get("confidence")
                    else:
                        tool_calls.append(block)

    reasoning = "\n".join(reasoning_parts)

    return ParsedResponse(
        reasoning=reasoning,
        tool_calls=tool_calls,
        final_answer=final_answer,
        final_confidence=final_confidence,
    )
```

### 9.2 Tool Execution with Interception

When `max_turns=1`, the SDK returns `ToolUseBlock`s without executing them. The orchestrator then executes tools itself, applying the intercept layer for Lumina tools:

```python
async def execute_tool(interceptor: LuminaInterceptor,
                       tc: ToolUseBlock) -> tuple[dict, list]:
    """Execute a single tool call, intercepting Lumina tools for state injection.
    Returns (result, indexed_refs) where indexed_refs is [] for non-Lumina tools.
    """
    tool_name = tc.name.split("__")[-1]  # e.g. "mcp__lumina-web__lumina_search" -> "lumina_search"

    if interceptor.is_lumina_tool(tool_name):
        # Intercept: inject tool_state, forward to Lumina, capture updated state
        result, indexed_refs = await interceptor.intercept(tool_name, tc.input)
    else:
        # Custom tools (save_page, grep_file, etc.): execute directly
        result = await custom_mcp_server.call_tool(tool_name, tc.input)
        indexed_refs = []

    return result, indexed_refs
```

This keeps Claude calling the real Lumina tool names (`lumina_search`, `lumina_open`, `lumina_find`) with their real schemas, while the orchestrator transparently manages `tool_state` threading.

### 9.3 Schema Flow Summary

```
1. SDK startup:
   SDK connects to lumina-web MCP server -> calls tools/list
   SDK connects to search_tools MCP server -> calls tools/list
   Both return JSON Schema tool definitions

2. Per-iteration Claude API call:
   SDK includes all tool schemas in the API "tools" parameter
   Claude sees: lumina_search, lumina_open, lumina_find,
                save_page, grep_file, read_file, submit_answer

3. Claude responds with ToolUseBlock:
   e.g. lumina_search({ q: "climate change", topN: 5 })
   Note: Claude may omit tool_state (optional in schema)

4. Orchestrator intercepts:
   Injects tool_state, searchResultContentType=2
   Forwards to Lumina MCP server
   Captures updated tool_state from response
   Summarizes result -> updates tree
```

---

## 10. Cost Optimization

The sample query cost $30.78 over 25 turns with ~1.1M input tokens. The primary issue: **raw tool responses were accumulating in context**.

Our design addresses this:

| Optimization | Mechanism | Estimated savings |
|---|---|---|
| Tree outline instead of raw history | Model sees ~4K tokens of tree outline vs. ~40K+ tokens of accumulated raw responses | ~90% input token reduction |
| Summarized tool responses | Tool handlers return 1-3 sentence summaries, not full JSON | ~95% reduction per tool result |
| Stateless iterations | Each iteration is a fresh `query()` - no context accumulation | Constant per-iteration cost |
| Configurable model | Use `claude-sonnet-4-6` instead of `claude-opus-4-6` for most iterations | ~5x cost reduction |
| Budget cap | `max_budget_usd` enforced by tracking `ResultMessage.total_cost_usd` | Hard cost ceiling |

**Estimated cost per query**: 25 iterations x ~5K input tokens x $3/M tokens (Sonnet) = **~$0.38** (vs. $30.78 in the sample).

---

## 11. Termination Conditions

The agent stops when any of these is true:

1. **Answer submitted** - Claude calls `submit_answer`.
2. **Max iterations reached** - `config.max_iterations` (default: 30).
3. **Budget exhausted** - Cumulative cost exceeds `config.max_budget_usd`.
4. **No progress** - The model's last 3 iterations produced no new tool calls and no answer (stalled).
5. **Explicit failure** - The model says it cannot find the answer.

---

## 12. Stream Logging & Post-Hoc Visualization

### 12.1 Stream Log Format

We write a `stream.jsonl` file compatible with `build_stream_tree.py`:

```jsonl
{"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "..."}, {"type": "tool_use", "id": "...", "name": "...", "input": {...}}]}}
{"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "...", "content": "..."}]}}
{"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "..."}]}}
{"type": "result", "result": "...", "duration_ms": ..., "num_turns": ..., "total_cost_usd": ...}
```

### 12.2 Post-Run Visualization

After the agent completes, run:

```bash
python build_stream_tree.py output/stream.jsonl --output-dir output/
```

This produces `stream_tree.json`, `stream_tree.html`, `stream_tree.md`, and `stream_tree.mmd`.

---

## 13. Error Handling

| Error | Handling |
|-------|----------|
| Lumina MCP server disconnected | Retry connection once. If still down, mark tool node as `status="error"` and let model decide next action. |
| Claude API rate limit | `RateLimitEvent` from SDK. Exponential backoff up to 60s, then fail the iteration. |
| Tool response too large | Truncate at 10K chars before summarizing. Store full response on disk, not in tree. |
| Malformed model output (no tool call, no answer) | Treat as a "reasoning only" model node. Prompt again next iteration with a nudge. |
| Max iterations reached without answer | Return partial findings with `confidence="low"`. |

---

## 14. Dependencies

```toml
[project]
name = "search-agent"
requires-python = ">=3.11"
dependencies = [
    "claude-agent-sdk",
]
```

The Lumina MCP server is launched as an external process via the `.mcp.json` config - no Python dependency needed.

---

## 15. Sequence Diagram - Single Iteration

```
Orchestrator        TreeManager        Claude SDK          Interceptor         Lumina MCP
    |                    |                  |                    |                  |
    |-render_outline()-->|                  |                    |                  |
    |<--tree_markdown----|                  |                    |                  |
    |                    |                  |                    |                  |
    |-build_prompt(tree_md)                 |                    |                  |
    |                    |                  |                    |                  |
    |-query(prompt)------------------------->                    |                  |
    |                    |                  |                    |                  |
    |<--ToolUseBlock(lumina_search)---------|                    |                  |
    |                    |                  |                    |                  |
    |-intercept(lumina_search, args)---------------------------->|                  |
    |                    |                  |    inject tool_state|                  |
    |                    |                  |                    |--lumina_search-->|
    |                    |                  |                    |<--results--------|
    |                    |                  |    capture tool_state                 |
    |<--result---------------------------------------------------|                  |
    |                    |                  |                    |                  |
    |-add_tool_node()--->|                  |                    |                  |
    |-add_model_node()-->|                  |                    |                  |
    |-stream_log.write()-                   |                    |                  |
    |                    |                  |                    |                  |
    v                    v                  v                    v                  v
```

---

## 16. Open Questions

1. **Model selection per iteration**: Should we use a cheaper model (Haiku) for initial broad searches and upgrade to Sonnet/Opus for deep analysis? This could be a strategy the orchestrator applies based on tree depth.

2. ~~**Parallel tool calls**: The sample shows Claude issuing 2-3 search calls per turn. Should we execute these concurrently (via `asyncio.gather`) or sequentially?~~ **Resolved**: Execute in parallel via `asyncio.gather`. All parallel tool calls are sibling nodes under the same model parent. See Section 3.3 and Section 4.2.

3. ~~**Tree pruning**: For very long runs (>50 iterations), should we prune/collapse old branches that the model has abandoned?~~ **Resolved**: Dead-end detection automatically collapses abandoned branches. When the model pivots, sibling tool nodes are marked dead and render as a single line. See Section 3.2.

4. **Caching**: Should we cache Lumina search results across runs (e.g., for the same query)? Could help with evaluation/debugging.

5. **Evaluation harness**: The sample used BrowseComp-style questions. Do we need a built-in eval framework, or is that out of scope for v1?
