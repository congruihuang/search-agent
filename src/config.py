"""Agent configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
    mcp_config_path: Path = field(default_factory=lambda: Path(".mcp.json"))

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./output"))

    # Debug
    enable_debug: bool = True

    # Prompt
    system_prompt_path: Path | None = None

    # Tree rendering
    tree_summary_max_length: int = 180
    tool_response_summary_max_length: int = 160
