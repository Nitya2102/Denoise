"""
Gradio UI for the A2A + MCP Image Processing Agent PoC.
"""

import gradio as gr
import time
import json
from PIL import Image
import os

from tools import build_mcp_registry
from agents import ImageProcessingOrchestrator


# ─── Build shared registry + orchestrator ─────────────────────────────────────
registry = build_mcp_registry()
orchestrator = ImageProcessingOrchestrator(registry)

PRESET_GOALS = [
    "Enhance this photo for printing — improve quality, sharpness and colors",
    "Prepare this image for a dark/moody artistic look",
    "Make this image suitable for OCR text extraction (maximize contrast and clarity)",
    "Create a clean edge map for technical analysis",
    "Optimize this noisy low-light photograph",
    "Make colors vivid and punchy for social media",
    "Convert to grayscale and boost contrast for a classic look",
    "Clean and resize this image for use as a thumbnail (256x256)",
]

# ─── CSS ──────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --border: #1e1e2e;
    --accent: #7c3aed;
    --accent2: #06b6d4;
    --accent3: #10b981;
    --warn: #f59e0b;
    --danger: #ef4444;
    --text: #e2e8f0;
    --muted: #64748b;
    --mono: 'Space Mono', monospace;
    --sans: 'DM Sans', sans-serif;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: var(--sans) !important;
    color: var(--text) !important;
}

h1, h2, h3 { font-family: var(--mono) !important; }

.main-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.main-header h1 {
    font-size: 1.8rem;
    color: var(--accent2);
    letter-spacing: -0.02em;
    margin: 0;
}
.main-header p {
    color: var(--muted);
    font-size: 0.9rem;
    margin: 0.5rem 0 0;
}
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-family: var(--mono);
    font-weight: 700;
    margin: 0 3px;
}
.badge-a2a { background: #1e1b4b; color: #818cf8; border: 1px solid #3730a3; }
.badge-mcp { background: #042f2e; color: #34d399; border: 1px solid #065f46; }

#plan-log, #exec-log {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    color: var(--text) !important;
}

.step-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-family: var(--mono);
    font-size: 0.78rem;
}
.step-card.done { border-left: 3px solid var(--accent3); }
.step-card.running { border-left: 3px solid var(--warn); }
.step-card.failed { border-left: 3px solid var(--danger); }
.step-card.skipped { border-left: 3px solid var(--muted); opacity: 0.6; }

.gr-button.primary {
    background: var(--accent) !important;
    border: none !important;
    font-family: var(--mono) !important;
}
"""

# ─── State helpers ─────────────────────────────────────────────────────────────

def format_plan_log(plan_event: dict) -> str:
    lines = []
    lines.append("PLANNER AGENT ANALYSIS")
    lines.append(f"{plan_event.get('image_description', '')}")
    lines.append(f"Confidence: {plan_event.get('confidence', 1.0):.0%}")
    lines.append("\nSELECTED STEPS")
    for i, step in enumerate(plan_event.get("steps_list", []), 1):
        lines.append(f"  {i}. [{step['step'].upper()}]")
        lines.append(f"     -> {step['reason']}")
        if step.get("params"):
            lines.append(f"     params: {json.dumps(step['params'])}")
    skipped = plan_event.get("skipped", [])
    if skipped:
        lines.append("\nSKIPPED (not needed)")
        for s in skipped:
            lines.append(f"  x [{s['step'].upper()}] — {s['reason']}")
    return "\n".join(lines)


def format_exec_line(event: dict) -> str:
    etype = event.get("event")
    if etype == "step_start":
        return (
            f"> STEP {event['step_index']+1}: {event['step_name'].upper()}\n"
            f"  A2A msg_id={event['message_id']}  MCP tool={event['step_name']}\n"
            f"  params={json.dumps(event['params'])}"
        )
    elif etype == "step_done":
        return (
            f"[OK]  {event['step_name'].upper()} "
            f"({event['duration_ms']:.0f}ms) — {event['info']}"
        )
    elif etype == "step_failed":
        return f"[FAIL]  {event['step_name'].upper()} — {event['error']}"
    elif etype == "pipeline_complete":
        return f"\nPIPELINE COMPLETE — {event['total_steps']} steps executed"
    return ""


# ─── Main processing function ─────────────────────────────────────────────────

def run_pipeline(image, goal):
    if image is None:
        yield (
            None, None,
            "Please upload an image first.",
            "No image provided.",
            "Upload an image to begin.",
            gr.update(visible=False),
        )
        return

    pil_img = Image.fromarray(image) if not isinstance(image, Image.Image) else image

    plan_text = ""
    exec_text = ""
    step_html = ""
    current_out = pil_img
    plan_visible = False

    steps_done = 0
    steps_total = 0

    for event in orchestrator.run(pil_img, goal):
        etype = event.get("event")

        if etype == "orchestrator_start":
            exec_text = f"Orchestrator started\n   Goal: {goal}\n\n"
            yield (pil_img, current_out, plan_text, exec_text, "Planning...", gr.update(visible=False))

        elif etype == "planning":
            exec_text += "PlannerAgent -> analyzing image with Groq Vision LLM...\n"
            yield (pil_img, current_out, plan_text, exec_text, "Planning ...", gr.update(visible=False))

        elif etype == "planning_failed":
            exec_text += f"\n[ERROR] Planning failed: {event['error']}\n"
            yield (pil_img, current_out, plan_text, exec_text, f"Planning failed: {event['error']}", gr.update(visible=False))

        elif etype == "a2a_dispatch":
            plan = event["plan"]
            plan_event = {
                "image_description": plan.image_description,
                "confidence": plan.confidence,
                "steps_list": plan.steps,
                "skipped": plan.skipped_steps,
            }
            plan_text = format_plan_log(plan_event)
            steps_total = len(plan.steps)

            exec_text += (
                f"\nA2A MESSAGE DISPATCHED\n"
                f"   From: PlannerAgent -> To: ExecutorAgent\n"
                f"   Task ID: {event['message'].task_id}\n"
                f"   Steps planned: {steps_total}\n\n"
            )

            # Build step card HTML
            step_html = '<div style="font-family:monospace;font-size:0.8rem;">'
            for s in plan.steps:
                step_html += (
                    f'<div class="step-card running">'
                    f'  <span style="color:#f59e0b">PENDING</span> '
                    f'  <b>{s["step"].upper()}</b>'
                    f'  <span style="color:#94a3b8;margin-left:8px">{s["reason"][:60]}...</span>'
                    f'</div>'
                )
            for s in plan.skipped_steps:
                step_html += (
                    f'<div class="step-card skipped">'
                    f'  <span>SKIP</span> '
                    f'  <b style="color:#64748b">{s["step"].upper()}</b> (skipped)'
                    f'</div>'
                )
            step_html += "</div>"

            plan_visible = True
            yield (pil_img, current_out, plan_text, exec_text, f"Plan ready — {steps_total} steps", gr.update(visible=True, value=step_html))

        elif etype == "plan_received":
            exec_text += f"ExecutorAgent received plan — {event['total_steps']} steps\n\n"

        elif etype == "step_start":
            line = format_exec_line(event)
            exec_text += line + "\n"
            yield (pil_img, current_out, plan_text, exec_text,
                   f"Running {event['step_name']} ({event['step_index']+1}/{steps_total})...",
                   gr.update(visible=True, value=step_html))

        elif etype == "step_done":
            steps_done += 1
            current_out = event["image"]
            line = format_exec_line(event)
            exec_text += line + "\n\n"
            # Update step card to done
            sname = event["step_name"].upper()
            step_html = step_html.replace(
                f'<span style="color:#f59e0b">PENDING</span>   <b>{sname}</b>',
                f'<span style="color:#10b981">DONE</span>   <b>{sname}</b>',
                1
            ).replace(
                '"step-card running"',
                '"step-card done"', 1
            )
            yield (pil_img, current_out, plan_text, exec_text,
                   f"{steps_done}/{steps_total} steps done",
                   gr.update(visible=True, value=step_html))

        elif etype == "step_failed":
            line = format_exec_line(event)
            exec_text += line + "\n\n"
            yield (pil_img, current_out, plan_text, exec_text,
                   f"Step failed: {event['step_name']}",
                   gr.update(visible=True, value=step_html))

        elif etype == "complete":
            current_out = event["image"]
            exec_text += format_exec_line(event)
            yield (pil_img, current_out, plan_text, exec_text,
                   f"Done! {event['total_steps']} steps completed.",
                   gr.update(visible=True, value=step_html))


# ─── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui():
    tool_list = registry.list_tools()
    tool_tags_html = ""
    for t in tool_list:
        tags = " ".join([f'<span style="background:#1e1b4b;color:#818cf8;padding:1px 6px;border-radius:3px;font-size:0.65rem;margin-right:3px">{tag}</span>' for tag in t["tags"]])
        tool_tags_html += (
            f'<div style="margin:4px 0;padding:6px 10px;background:#12121a;border:1px solid #1e1e2e;border-radius:6px;font-family:monospace;font-size:0.75rem">'
            f'<b style="color:#06b6d4">{t["name"]}</b>  '
            f'<span style="color:#94a3b8">{t["description"]}</span>  {tags}'
            f'</div>'
        )

    with gr.Blocks(title="A2A+MCP Image Agent PoC") as demo:
        gr.HTML("""
        <div class="main-header">
            <h1>IMAGE PROCESSING AGENT</h1>
            <p>
                <span class="badge badge-a2a">A2A Protocol</span>
                <span class="badge badge-mcp">MCP Tools</span>
                
            </p>
        </div>
        """)

        with gr.Row():
            # ── Left panel ──────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                image_input = gr.Image(label="Upload Image", type="numpy", height=260)

                goal_input = gr.Textbox(
                    label="Processing Goal",
                    placeholder="Describe what you want to achieve...",
                    lines=2,
                    value=PRESET_GOALS[0],
                )

                with gr.Accordion("Example Goals", open=False):
                    for goal in PRESET_GOALS:
                        gr.Button(goal, size="sm").click(
                            fn=lambda g=goal: g,
                            outputs=goal_input,
                        )

                run_btn = gr.Button("Run Agent", variant="primary", size="lg")
                status_text = gr.Textbox(label="Status", interactive=False, lines=1)

                gr.HTML("<hr style='border-color:#1e1e2e;margin:12px 0'>")
                gr.Markdown("### MCP Tool Registry")
                gr.HTML(f'<div style="max-height:350px;overflow-y:auto">{tool_tags_html}</div>')

            # ── Middle panel ─────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("")

                with gr.Row():
                    input_preview  = gr.Image(label="Original", height=200, interactive=False)
                    output_preview = gr.Image(label="Output (live)", height=200, interactive=False)

                step_cards = gr.HTML(label="Steps", visible=False)

                with gr.Accordion("Planner Agent Log (A2A Plan)", open=True):
                    plan_log = gr.Textbox(
                        elem_id="plan-log", label="", interactive=False,
                        lines=12, max_lines=20,
                    )

                with gr.Accordion("Executor Agent Log (MCP Invocations)", open=True):
                    exec_log = gr.Textbox(
                        elem_id="exec-log", label="", interactive=False,
                        lines=12, max_lines=20,
                    )

        run_btn.click(
            fn=run_pipeline,
            inputs=[image_input, goal_input],
            outputs=[input_preview, output_preview, plan_log, exec_log, status_text, step_cards],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
    )