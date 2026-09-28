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


WIDTH, HEIGHT = 1440, 1050
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


def metric_row(parts: list[str], *, y: float, label: str, value: str, colour: str, width: float | None) -> None:
    label_x, bar_x, value_x = 108, 390, 1110
    parts.append(svg_text(label_x, y + 10, label, size=21, fill=colour, weight=700))
    if width is not None:
        parts.append(f'<rect x="{bar_x:.1f}" y="{y - 16:.1f}" width="{width:.1f}" height="30" rx="15" fill="{colour}"/>')
    parts.append(svg_text(value_x, y + 10, value, size=24, fill=WHITE, weight=700))


def panel(parts: list[str], *, y: float, heading: str, takeaway: str) -> None:
    parts.extend([
        f'<rect x="72" y="{y:.1f}" width="1296" height="238" rx="20" fill="{PANEL}"/>',
        svg_text(108, y + 48, heading, size=18, fill=MUTED, weight=700),
        svg_text(108, y + 88, takeaway, size=27, weight=700),
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
    max_bar = 610
    latency_ratio = jev_latency / compass_latency
    qwen_latency_ratio = qwen_latency / compass_latency
    output_ratio = qwen_output / jev_output
    cost_ratio = qwen_cost / jev_cost
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        '<title id="title">Manager-chain deployment profile: Compass, Jev, and Qwen3.5-9B</title>',
        '<desc id="desc">On the same 140 manager-chain questions, Compass has a 0.287-second median localhost endpoint latency on an RTX 4090 and generates no answer tokens. Jev has a 0.334-second median TypeSafe API latency, reports 2,800 output tokens, and costs 0.0028 dollars. Qwen3.5-9B has a 23.54-second median OpenRouter endpoint latency, generates 444,748 tokens, and costs 0.056 dollars.</desc>',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{INK}" rx="28"/>',
        svg_text(72, 84, "the manager-chain deployment profile", size=48, weight=700),
        svg_text(72, 122, f"Same {metrics['questions']} questions. Compass and Jev return a typed probability; Qwen3.5-9B writes a reasoning trace.", size=22, fill=MUTED),
    ]
    panel(parts, y=170, heading="MEDIAN REQUEST LATENCY", takeaway=f"Compass was {latency_ratio:.1f}× faster than Jev on these endpoint routes")
    metric_row(parts, y=286, label="Compass, local RTX 4090", value=f"{compass_latency:.3f}s", colour=COMPASS, width=max(18, max_bar * compass_latency / qwen_latency))
    metric_row(parts, y=338, label="Jev 1.13.0, TypeSafe API", value=f"{jev_latency:.3f}s", colour=JEV, width=max(18, max_bar * jev_latency / qwen_latency))
    metric_row(parts, y=390, label="Qwen3.5-9B, thinking on", value=f"{qwen_latency:.2f}s ({qwen_latency_ratio:.0f}×)", colour=QWEN, width=max_bar)
    panel(parts, y=432, heading="GENERATED ANSWER TOKENS", takeaway=f"Qwen used {output_ratio:.0f}× Jev’s answer tokens")
    metric_row(parts, y=548, label="Compass, fixed readout", value=f"{compass_output}", colour=COMPASS, width=18)
    metric_row(parts, y=600, label="Jev 1.13.0, TypeSafe API", value=f"{jev_output:,}", colour=JEV, width=max(18, max_bar * jev_output / qwen_output))
    metric_row(parts, y=652, label="Qwen3.5-9B, thinking on", value=f"{qwen_output:,}", colour=QWEN, width=max_bar)
    panel(parts, y=694, heading="HOSTED API RUN CHARGE", takeaway=f"Qwen cost {cost_ratio:.0f}× Jev’s API charge")
    metric_row(parts, y=810, label="Compass, local RTX 4090", value="self-hosted GPU", colour=COMPASS, width=None)
    metric_row(parts, y=862, label="Jev 1.13.0, TypeSafe API", value=f"${jev_cost:.4f}", colour=JEV, width=max(18, max_bar * jev_cost / qwen_cost))
    metric_row(parts, y=914, label="Qwen3.5-9B, thinking on", value=f"${qwen_cost:.3f}", colour=QWEN, width=max_bar)
    parts.extend([
        svg_text(72, 994, "Latency is client-observed end-to-end time: Compass ran locally on an RTX 4090; Jev was called serially from that host; Qwen used OpenRouter with 12 concurrent requests.", size=17, fill=MUTED),
        svg_text(72, 1028, "The two dollar figures are hosted API charges. Compass uses a self-hosted GPU, so its hardware cost is not shown beside them.", size=17, fill=MUTED),
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
