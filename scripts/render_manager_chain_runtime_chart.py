"""Render the manager-chain latency and token-work comparison as a self-contained SVG.

    uv run python scripts/render_manager_chain_runtime_chart.py \
      --jev-input dev/results/manager-chain-jev-2026-09-27.json \
      --qwen-input dev/results/manager-chain-qwen9b-2026-09-27.json \
      --out docs/diagrams/manager-chain-runtime-qwen9b.svg
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path


WIDTH, HEIGHT = 1440, 700
INK = "#0E1014"
MUTED = "#9EA5AF"
WHITE = "#F4F6F8"
JEV = "#F4D35E"
QWEN = "#F17EBF"
PANEL = "#181C22"
JEV_RUN_COST = 0.0028


def escaped(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_text(x: float, y: float, value: str, *, size: int = 22, fill: str = WHITE, anchor: str = "start", weight: int = 400) -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-family="ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escaped(value)}</text>'


def metric_rows(jev_data: dict, qwen_data: dict) -> dict[str, float | int]:
    jev_rows = jev_data["jev"]["rows"]
    qwen_rows = qwen_data["qwen_thinking"]["rows"]
    if len(jev_rows) != len(qwen_rows):
        raise SystemExit("Jev and Qwen result files must contain the same number of tasks")
    jev_latency = statistics.median(row["latency_seconds"] for row in jev_rows)
    qwen_latency = statistics.median(row["latency_seconds"] for row in qwen_rows)
    jev_output = sum(row["usage"]["output_tokens"] for row in jev_rows)
    qwen_output = sum(row["generated_tokens"] or 0 for row in qwen_rows)
    qwen_cost = sum(row["cost_usd"] or 0 for row in qwen_rows)
    return {
        "questions": len(jev_rows),
        "jev_latency": jev_latency,
        "qwen_latency": qwen_latency,
        "jev_output": jev_output,
        "qwen_output": qwen_output,
        "jev_cost": JEV_RUN_COST,
        "qwen_cost": qwen_cost,
    }


def bar(parts: list[str], *, x: float, y: float, width: float, label: str, value: str, colour: str) -> None:
    parts.extend([
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="30" rx="15" fill="{colour}"/>',
        svg_text(x, y - 14, label, size=20, fill=colour, weight=700),
        svg_text(x + width + 18, y + 22, value, size=24, fill=WHITE, weight=700),
    ])


def render(metrics: dict[str, float | int]) -> str:
    jev_latency = float(metrics["jev_latency"])
    qwen_latency = float(metrics["qwen_latency"])
    jev_output = int(metrics["jev_output"])
    qwen_output = int(metrics["qwen_output"])
    jev_cost = float(metrics["jev_cost"])
    qwen_cost = float(metrics["qwen_cost"])
    latency_ratio = qwen_latency / jev_latency
    output_ratio = qwen_output / jev_output
    cost_ratio = qwen_cost / jev_cost
    left_x, middle_x, right_x, top_y, panel_width = 72, 520, 968, 180, 400
    max_bar = 240
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        '<title id="title">Manager-chain speed and cost profile: Jev and Qwen3.5-9B</title>',
        '<desc id="desc">On the same 140 manager-chain questions, Jev has a median endpoint latency of 0.38 seconds, reports 2,800 output tokens, and costs 0.0028 dollars. Qwen3.5-9B has a median endpoint latency of 23.54 seconds, generates 444,748 tokens, and costs 0.056 dollars.</desc>',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{INK}" rx="28"/>',
        svg_text(72, 84, "the manager-chain speed & cost profile", size=48, weight=700),
        svg_text(72, 122, f"Same {metrics['questions']} questions. Jev returns a typed probability; Qwen3.5-9B writes a reasoning trace.", size=22, fill=MUTED),
        f'<rect x="{left_x}" y="{top_y}" width="{panel_width}" height="330" rx="20" fill="{PANEL}"/>',
        f'<rect x="{middle_x}" y="{top_y}" width="{panel_width}" height="330" rx="20" fill="{PANEL}"/>',
        f'<rect x="{right_x}" y="{top_y}" width="{panel_width}" height="330" rx="20" fill="{PANEL}"/>',
        svg_text(left_x + 36, top_y + 52, "MEDIAN REQUEST LATENCY", size=18, fill=MUTED, weight=700),
        svg_text(left_x + 36, top_y + 91, f"Qwen was {latency_ratio:.0f}× slower", size=26, weight=700),
        svg_text(middle_x + 36, top_y + 52, "REPORTED OUTPUT TOKENS", size=18, fill=MUTED, weight=700),
        svg_text(middle_x + 36, top_y + 91, f"Qwen used {output_ratio:.0f}× tokens", size=26, weight=700),
        svg_text(right_x + 36, top_y + 52, "RECORDED RUN COST", size=18, fill=MUTED, weight=700),
        svg_text(right_x + 36, top_y + 91, f"Qwen cost {cost_ratio:.0f}× more", size=26, weight=700),
    ]
    bar(parts, x=left_x + 36, y=top_y + 160, width=max(12, max_bar * jev_latency / qwen_latency), label="Jev", value=f"{jev_latency:.2f}s", colour=JEV)
    bar(parts, x=left_x + 36, y=top_y + 264, width=max_bar, label="Qwen3.5-9B, thinking on", value=f"{qwen_latency:.2f}s", colour=QWEN)
    bar(parts, x=middle_x + 36, y=top_y + 160, width=max(12, max_bar * jev_output / qwen_output), label="Jev", value=f"{jev_output:,}", colour=JEV)
    bar(parts, x=middle_x + 36, y=top_y + 264, width=max_bar, label="Qwen3.5-9B, thinking on", value=f"{qwen_output:,}", colour=QWEN)
    bar(parts, x=right_x + 36, y=top_y + 160, width=max(12, max_bar * jev_cost / qwen_cost), label="Jev", value=f"${jev_cost:.4f}", colour=JEV)
    bar(parts, x=right_x + 36, y=top_y + 264, width=max_bar, label="Qwen3.5-9B, thinking on", value=f"${qwen_cost:.3f}", colour=QWEN)
    parts.extend([
        svg_text(72, 580, f"Jev cost ${jev_cost:.4f} for the run (${jev_cost / int(metrics['questions']):.5f} per question). OpenRouter billed Qwen ${qwen_cost:.3f} (${qwen_cost / int(metrics['questions']):.5f} per question).", size=20, fill=WHITE, weight=600),
        svg_text(72, 620, "Each panel compares the same 140 questions.", size=20, fill=MUTED),
        svg_text(72, 656, "Latency is the median end-to-end request time. Qwen requests ran with 12-way concurrency; this is not total suite wall time.", size=18, fill=MUTED),
        '</svg>',
    ])
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jev-input", type=Path, required=True)
    parser.add_argument("--qwen-input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    jev_data = json.loads(args.jev_input.read_text(encoding="utf-8"))
    qwen_data = json.loads(args.qwen_input.read_text(encoding="utf-8"))
    if jev_data["tasks"] != qwen_data["tasks"]:
        raise SystemExit("Jev and Qwen results must use the same generated task suite")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(metric_rows(jev_data, qwen_data)), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
