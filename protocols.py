"""
A2A (Agent-to-Agent) and MCP (Model Context Protocol) protocol definitions.
These define the communication contracts between agents and tools.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum
import uuid
import time


# ─── MCP: Tool Registry ───────────────────────────────────────────────────────

@dataclass
class MCPTool:
    """A capability registered in the MCP tool registry."""
    name: str
    description: str
    input_schema: dict
    handler: Callable
    tags: list[str] = field(default_factory=list)


class MCPRegistry:
    """MCP Server — agents discover and invoke tools from here."""
    
    def __init__(self):
        self._tools: dict[str, MCPTool] = {}
    
    def register(self, tool: MCPTool):
        self._tools[tool.name] = tool
        return tool
    
    def list_tools(self) -> list[dict]:
        """Return tool manifests (no handlers) for agent discovery."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
                "tags": t.tags,
            }
            for t in self._tools.values()
        ]
    
    def invoke(self, tool_name: str, inputs: dict) -> Any:
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found in MCP registry")
        return self._tools[tool_name].handler(**inputs)


# ─── A2A: Agent Message Protocol ─────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    SKIPPED   = "skipped"


@dataclass
class A2AMessage:
    """Message passed between agents in the A2A protocol."""
    sender: str
    recipient: str
    task_id: str
    content: dict
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class TaskResult:
    """Result returned by an executor agent."""
    task_id: str
    step_name: str
    status: TaskStatus
    output: Any = None
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class ExecutionPlan:
    """The dynamic plan created by the Planner Agent."""
    image_description: str
    steps: list[dict]           # [{"step": "denoise", "reason": "...", "params": {...}}]
    skipped_steps: list[dict]   # [{"step": "ocr", "reason": "no text detected"}]
    confidence: float = 1.0