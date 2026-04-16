"""
Agents: Planner + Executor, communicating via A2A protocol,
using MCP tools for capability discovery and execution.

LLM Backend: Groq — Llama 3.2 11B Vision (FREE tier)
  -> Supports vision/image input via base64
  -> Free tier: generous rate limits, no credit card needed
  -> Get free API key: https://console.groq.com/keys
  -> Set env var: GROQ_API_KEY=your_key
"""

import json
import time
import base64
import io
import os
import re
from typing import Generator
from PIL import Image

from dotenv import load_dotenv
load_dotenv()  # loads GROQ_API_KEY from .env automatically

from groq import Groq

from protocols import (
    MCPRegistry, A2AMessage, TaskResult, ExecutionPlan,
    TaskStatus
)

# ─── Groq Client Setup ────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Llama 4 Scout 17B — free on Groq, supports vision (replaces deprecated llama-3.2-11b-vision)
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _call_groq_vision(image: Image.Image, prompt: str) -> str:
    """
    Call Groq's Llama 3.2 11B Vision with image + text.
    The image is base64-encoded and sent as a data URL in the
    standard OpenAI-compatible vision message format.
    """
    if _client is None:
        raise RuntimeError(
            "GROQ_API_KEY not set.\n"
            "Get a free key at https://console.groq.com/keys\n"
            "Then run:  set GROQ_API_KEY=your_key  (Windows)\n"
            "      or:  export GROQ_API_KEY=your_key  (Linux/Mac)"
        )

    # Encode image as base64 data URL
    img_b64 = _pil_to_b64(image)
    data_url = f"data:image/png;base64,{img_b64}"

    response = _client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        temperature=0.2,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


# ─── Planner Agent ────────────────────────────────────────────────────────────

class PlannerAgent:
    """
    Analyzes the image + user goal using Groq Vision (FREE),
    then dynamically selects which MCP tools to invoke and in what order.
    No hardcoded pipeline — the LLM reasons about each specific image.
    """

    name = "PlannerAgent"

    def __init__(self, registry: MCPRegistry):
        self.registry = registry

    def plan(self, image: Image.Image, user_goal: str) -> ExecutionPlan:
        tools_manifest = json.dumps(self.registry.list_tools(), indent=2)

        prompt = f"""You are an expert image processing planner. Analyze the given image and the user's goal, then create a precise, non-hardcoded execution plan.

USER GOAL: {user_goal}

AVAILABLE MCP TOOLS (these are the ONLY tools you can use):
{tools_manifest}

Your task:
1. Analyze the image carefully - note its quality, content, issues, and characteristics
2. Select ONLY the tools that are truly needed for this specific image and goal
3. Order them correctly (e.g., denoise before sharpen, resize before effects)
4. For each selected tool, provide specific parameter values suited to this image
5. For tools you are NOT selecting, briefly explain why they are skipped

Respond ONLY with valid JSON - no markdown fences, no preamble, no explanation outside the JSON:
{{
  "image_description": "<2-3 sentence description of what you see in the image and its quality>",
  "steps": [
    {{
      "step": "<tool_name from the list above>",
      "reason": "<why this tool is needed for THIS specific image>",
      "params": {{}}
    }}
  ],
  "skipped_steps": [
    {{
      "step": "<tool_name>",
      "reason": "<why this tool is NOT needed for this image>"
    }}
  ],
  "confidence": 0.9
}}

Be selective. Not every image needs every tool. Quality over quantity."""

        raw = _call_groq_vision(image, prompt)

        # Strip markdown fences if model wraps them anyway
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
        raw = raw.strip()

        data = json.loads(raw)
        return ExecutionPlan(
            image_description=data["image_description"],
            steps=data["steps"],
            skipped_steps=data.get("skipped_steps", []),
            confidence=data.get("confidence", 1.0),
        )

    def send_plan(self, plan: ExecutionPlan, executor_name: str) -> A2AMessage:
        """Pack the plan into an A2A message for the executor."""
        return A2AMessage(
            sender=self.name,
            recipient=executor_name,
            task_id=f"plan-{int(time.time())}",
            content={
                "type": "EXECUTE_PLAN",
                "plan": {
                    "image_description": plan.image_description,
                    "steps": plan.steps,
                    "skipped_steps": plan.skipped_steps,
                    "confidence": plan.confidence,
                }
            }
        )


# ─── Executor Agent ───────────────────────────────────────────────────────────

class ExecutorAgent:
    """
    Receives an A2A plan message, executes each step via MCP tools,
    and yields live progress updates.
    Pure Python — no LLM calls here, just deterministic tool invocations.
    """

    name = "ExecutorAgent"

    def __init__(self, registry: MCPRegistry):
        self.registry = registry

    def execute(
        self, message: A2AMessage, image: Image.Image
    ) -> Generator[dict, None, None]:
        plan_data = message.content["plan"]
        steps = plan_data["steps"]
        current_image = image
        current_b64 = _pil_to_b64(current_image)

        yield {
            "event": "plan_received",
            "image_description": plan_data["image_description"],
            "total_steps": len(steps),
            "skipped": plan_data["skipped_steps"],
            "confidence": plan_data["confidence"],
            "image": current_image,
        }

        for i, step_spec in enumerate(steps):
            tool_name = step_spec["step"]
            params = step_spec.get("params", {})
            reason = step_spec.get("reason", "")

            # A2A message: ExecutorAgent -> MCPServer
            step_msg = A2AMessage(
                sender=self.name,
                recipient="MCPServer",
                task_id=f"step-{i}-{tool_name}",
                content={
                    "type": "INVOKE_TOOL",
                    "tool": tool_name,
                    "params": {"image_b64": current_b64, **params},
                    "reason": reason,
                }
            )

            yield {
                "event": "step_start",
                "step_index": i,
                "step_name": tool_name,
                "reason": reason,
                "params": params,
                "message_id": step_msg.message_id,
                "image": current_image,
            }

            t0 = time.time()
            try:
                result = self.registry.invoke(
                    step_msg.content["tool"],
                    step_msg.content["params"]
                )
                duration = (time.time() - t0) * 1000
                current_b64 = result["image_b64"]
                current_image = _b64_to_pil(current_b64)

                yield {
                    "event": "step_done",
                    "step_index": i,
                    "step_name": tool_name,
                    "status": "done",
                    "info": result.get("info", ""),
                    "duration_ms": duration,
                    "image": current_image,
                }

            except Exception as e:
                duration = (time.time() - t0) * 1000
                yield {
                    "event": "step_failed",
                    "step_index": i,
                    "step_name": tool_name,
                    "status": "failed",
                    "error": str(e),
                    "duration_ms": duration,
                    "image": current_image,
                }

        yield {
            "event": "pipeline_complete",
            "total_steps": len(steps),
            "image": current_image,
        }


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class ImageProcessingOrchestrator:
    """
    Top-level coordinator.
    Wires PlannerAgent (Groq Vision FREE) -> A2A -> ExecutorAgent -> MCP tools.
    """

    def __init__(self, registry: MCPRegistry):
        self.planner = PlannerAgent(registry)
        self.executor = ExecutorAgent(registry)

    def run(self, image: Image.Image, goal: str) -> Generator[dict, None, None]:
        yield {"event": "orchestrator_start", "goal": goal, "image": image}

        yield {"event": "planning", "image": image}
        try:
            plan = self.planner.plan(image, goal)
        except Exception as e:
            yield {"event": "planning_failed", "error": str(e), "image": image}
            return

        a2a_msg = self.planner.send_plan(plan, self.executor.name)
        yield {"event": "a2a_dispatch", "message": a2a_msg, "plan": plan, "image": image}

        yield from self.executor.execute(a2a_msg, image)