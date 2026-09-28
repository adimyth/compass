"""Render the manager-chain deployment comparison as a self-contained SVG.

    uv run python scripts/render_manager_chain_runtime_chart.py \
      --compass-input dev/results/manager-chain-compass-4090-2026-09-28.json \
      --jev-input dev/results/manager-chain-jev-4090host-2026-09-28.json \
      --qwen-input dev/results/manager-chain-qwen9b-2026-09-27.json \
      --out docs/diagrams/manager-chain-runtime-qwen9b.svg
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path


WIDTH, HEIGHT = 1440, 760
INK = "#0E1014"
MUTED = "#9EA5AF"
WHITE = "#F4F6F8"
COMPASS = "#57D6E8"
JEV = "#F4D35E"
QWEN = "#F17EBF"
PANEL = "#181C22"
JEV_RUN_COST = 0.0028


def escaped(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_text(x: float, y: float, value: str, *, size: int = 22, fill: str = WHITE, anchor: str = "start", weight: int = 400) -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-family="ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escaped(value)}</text>'


def metric_rows(compass_data: dict, jev_data: dict, qwen_data: dict) -> dict[str, float | int]:
    compass_rows = compass_data["compass"]["rows"]
    jev_rows = jev_data["jev"]["rows"]
    qwen_rows = qwen_data["qwen_thinking"]["rows"]
    if len(compass_rows) != len(jev_rows) or len(compass_rows) != len(qwen_rows):
        raise SystemExit("Compass, Jev, and Qwen result files must contain the same number of tasks")
    return {
        "questions": len(compass_rows),
        "compass_latency": statistics.median(row["latency_seconds"] for row in compass_rows),
        "jev_latency": statistics.median(row["latency_seconds"] for row in jev_rows),
        "qwen_latency": statistics.median(row["latency_seconds"] for row in qwen_rows),
        "compass_output": 0,
        "jev_output": sum(row["usage"]["output_tokens"] for row in jev_rows),
        "qwen_output": sum(row["generated_tokens"] or 0 for row in qwen_rows),
        "jev_cost": JEV_RUN_COST,
        "qwen_cost": sum(row["cost_usd"] or 0 for row in qwen_rows),
    }


def bar(parts: list[str], *, x: float, y: float, width: float, label: str, value: str, colour: str) -> None:
    parts.extend([
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="30" rx="15" fill="{colour}"/>',
        svg_text(x, y - 14, label, size=20, fill=colour, weight=700),
        svg_text(x + width + 18, y + 22, value, size=24, fill=WHITE, weight=700),
    ])


def render(metrics: dict[str, float | int]) -> str:
    compass_latency = float(metrics["compass_latency"])
    jev_latency = float(metrics["jev_latency"])
    qwen_latency = float(metrics["qwen_latency"])
    compass_output = int(metrics["compass_output"])
    jev_output = int(metrics["jev_output"])
    qwen_output = int(metrics["qwen_output"])
    jev_cost = float(metrics["jev_cost"])
    qwen_cost = float(metrics["qwen_cost"])
    compass_jev_latency_ratio = jev_latency / compass_latency
    qwen_compass_latency_ratio = qwen_latency / compass_latency
    output_ratio = qwen_output / jev_output
    cost_ratio = qwen_cost / jev_cost
    left_x, middle_x, right_x, top_y, panel_width = 72, 520, 968, 180, 400
    max_bar = 240
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        '<title id="title">Manager-chain deployment profile: Compass, Jev, and Qwen3.5-9B</title>',
        '<desc id="desc">On the same 140 manager-chain questions, Compass has a 0.29-second median localhost endpoint latency on an RTX 4090 and generates no answer tokens. Jev has a 0.33-second median TypeSafe API latency, reports 2,800 output tokens, and costs 0.0028 dollars. Qwen3.5-9B has a 23.54-second median OpenRouter endpoint latency, generates 444,748 tokens, and costs 0.056 dollars.</desc>',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{INK}" rx="28"/>',
        svg_text(72, 84, "the manager-chain deployment profile", size=48, weight=700),
        svg_text(72, 122, f"Same {metrics['questions']} questions. Compass and Jev return a typed probability; Qwen3.5-9B writes a reasoning trace.", size=22, fill=MUTED),
        f'<rect x="{left_x}" y="{top_y}" width="{panel_width}" height="390" rx="20" fill="{PANEL}"/>',
        f'<rect x="{middle_x}" y="{top_y}" width="{panel_width}" height="390" rx="20" fill="{PANEL}"/>',
        f'<rect x="{right_x}" y="{top_y}" width="{panel_width}" height="390" rx="20" fill="{PANEL}"/>',
        svg_text(left_x + 36, top_y + 52, "MEDIAN REQUEST LATENCY", size=18, fill=MUTED, weight=700),
        svg_text(left_x + 36, top_y + 91, f"Compass was {compass_jev_latency_ratio:.1f}× faster than Jev", size=24, weight=700),
        svg_text(middle_x + 36, top_y + 52, "GENERATED ANSWER TOKENS", size=18, fill=MUTED, weight=700),
        svg_text(middle_x + 36, top_y + 91, f"Qwen used {output_ratio:.0f}× Jev's tokens", size=26, weight=700),
        svg_text(right_x + 36, top_y + 52, "HOSTED API RUN CHARGE", size=18, fill=MUTED, weight=700),
        svg_text(right_x + 36, top_y + 91, f"Qwen cost {cost_ratio:.0f}× Jev's API charge", size=24, weight=700),
    ]
    bar(parts, x=left_x + 36, y=top_y + 160, width=max(12, max_bar * compass_latency / qwen_latency), label="Compass, RTX 4090", value=f"{compass_latency:.3f}s", colour=COMPASS)
    bar(parts, x=left_x + 36, y=top_y + 256, width=max(12, max_bar * jev_latency / qwen_latency), label="Jev 1.13.0 API", value=f"{jev_latency:.3f}s", colour=JEV)
    bar(parts, x=left_x + 36, y=top_y + 352, width=max_bar, label="Qwen3.5-9B, thinking on", value=f"{qwen_latency:.2f}s ({qwen_compass_latency_ratio:.0f}×)", colour=QWEN)
    bar(parts, x=middle_x + 36, y=top_y + 160, width=12, label="Compass, RTX 4090", value=f"{compass_output} (fixed readout)", colour=COMPASS)
    bar(parts, x=middle_x + 36, y=top_y + 256, width=max(12, max_bar * jev_output / qwen_output), label="Jev 1.13.0 API", value=f"{jev_output:,}", colour=JEV)
    bar(parts, x=middle_x + 36, y=top_y + 352, width=max_bar, label="Qwen3.5-9B, thinking on", value=f"{qwen_output:,}", colour=QWEN)
    parts.extend([
        svg_text(right_x + 36, top_y + 146, "Compass, RTX 4090", size=20, fill=COMPASS, weight=700),
        svg_text(right_x + 36, top_y + 180, "self-hosted GPU", size=24, fill=WHITE, weight=700),
        svg_text(right_x + 36, top_y + 208, "No hosted API charge assigned", size=17, fill=MUTED),
    ])
    bar(parts, x=right_x + 36, y=top_y + 256, width=max(12, max_bar * jev_cost / qwen_cost), label="Jev 1.13.0 API", value=f"${jev_cost:.4f}", colour=JEV)
    bar(parts, x=right_x + 36, y=top_y + 352, width=max_bar, label="Qwen3.5-9B, thinking on", value=f"${qwen_cost:.3f}", colour=QWEN)
    parts.extend([
        svg_text(72, 640, f"Jev cost ${jev_cost:.4f} for the run (${jev_cost / int(metrics['questions']):.5f} per question). OpenRouter billed Qwen ${qwen_cost:.3f} (${qwen_cost / int(metrics['questions']):.5f} per question).", size=20, fill=WHITE, weight=600),
        svg_text(72, 680, "Latency is client-observed end-to-end time: Compass ran locally on an RTX 4090; Jev was called serially from that host; Qwen used OpenRouter with 12 concurrent requests.", size=17, fill=MUTED),
        svg_text(72, 716, "Compass uses a self-hosted GPU, so its cost is not comparable with the two hosted API charges.", size=17, fill=MUTED),
        '</svg>',
    ])
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compass-input", type=Path, required=True)
    parser.add_argument("--jev-input", type=Path, required=True)
    parser.add_argument("--qwen-input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compass_data = json.loads(args.compass_input.read_text(encoding="utf-8"))
    jev_data = json.loads(args.jev_input.read_text(encoding="utf-8"))
    qwen_data = json.loads(args.qwen_input.read_text(encoding="utf-8"))
    if compass_data["tasks"] != jev_data["tasks"] or compass_data["tasks"] != qwen_data["tasks"]:
        raise SystemExit("Compass, Jev, and Qwen results must use the same generated task suite")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(metric_rows(compass_data, jev_data, qwen_data)), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
