"""Render the manager-chain experiment as a self-contained SVG chart.

    uv run python scripts/render_manager_chain_chart.py \
      --compass-input dev/results/manager-chain-2026-09-27.json \
      --jev-input dev/results/manager-chain-jev-2026-09-27.json \
      --qwen-input dev/results/manager-chain-qwen9b-2026-09-27.json \
      --out docs/diagrams/manager-chain-results-qwen9b.svg
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path


WIDTH, HEIGHT = 1440, 950
LEFT, RIGHT, TOP, BOTTOM = 170, 1340, 250, 555
COMPASS = "#57D6E8"
QWEN = "#F17EBF"
JEV = "#F4D35E"
TOKEN = "#FFB45E"
INK = "#0E1014"
MUTED = "#9EA5AF"
WHITE = "#F4F6F8"


def position(index: int, count: int) -> float:
    return LEFT + (RIGHT - LEFT) * index / max(count - 1, 1)


def y_position(accuracy: float) -> float:
    return BOTTOM - (BOTTOM - TOP) * accuracy


def escaped(value: object) -> str:
    return html.escape(str(value), quote=True)


def line(x1: float, y1: float, x2: float, y2: float, **attrs: object) -> str:
    values = " ".join(f'{name}="{escaped(value)}"' for name, value in attrs.items())
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {values}/>'


def text(x: float, y: float, value: str, *, size: int = 22, fill: str = WHITE, anchor: str = "start", weight: int = 400) -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-family="ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escaped(value)}</text>'


def point(x: float, y: float, colour: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{colour}" stroke="{INK}" stroke-width="3"/>'


def per_depth(rows: list[dict], depths: list[int]) -> dict[int, dict]:
    output = {}
    for depth in depths:
        group = [row for row in rows if row["depth"] == depth]
        output[depth] = {
            "correct": sum(row["correct"] for row in group),
            "total": len(group),
            "accuracy": sum(row["correct"] for row in group) / len(group),
            "truncated": sum(row.get("truncated", False) for row in group),
            "unanswered": sum(row.get("predicted", "") is None for row in group),
            "reasoning_tokens": statistics.median(row["reasoning_tokens"] for row in group) if group and "reasoning_tokens" in group[0] else None,
        }
    return output


def render(compass_data: dict, jev_data: dict, qwen_data: dict) -> str:
    depths = compass_data["lengths"]
    compass = per_depth(compass_data["compass"]["rows"], depths)
    qwen = per_depth(qwen_data["qwen_thinking"]["rows"], depths)
    jev = per_depth(jev_data["jev"]["rows"], depths)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        '<title id="title">Manager-chain accuracy: Compass, Jev, and Qwen thinking</title>',
        '<desc id="desc">Accuracy on balanced reporting-chain questions as chain length increases. Compass and Jev are fixed-compute System One decision models. Qwen generates a reasoning trace.</desc>',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{INK}" rx="28"/>',
        text(72, 84, "the manager-chain test", size=48, weight=700),
        text(72, 122, "Same reporting questions. System One scoring versus generated reasoning.", size=22, fill=MUTED),
    ]
    for label, value in (("100%", 1.0), ("50%", 0.5), ("0%", 0.0)):
        y = y_position(value)
        parts.append(line(LEFT, y, RIGHT, y, stroke="#3B414A" if value != 0 else WHITE, **{"stroke-width": 2 if value in (0.0, 1.0) else 1}))
        parts.append(text(LEFT - 20, y + 7, label, size=18, fill=MUTED, anchor="end"))
    parts.append(line(LEFT, TOP, LEFT, BOTTOM, stroke=WHITE, **{"stroke-width": 2}))
    for index, depth in enumerate(depths):
        x = position(index, len(depths))
        parts.append(line(x, BOTTOM, x, BOTTOM + 12, stroke=WHITE, **{"stroke-width": 2}))
        parts.append(text(x, BOTTOM + 48, str(depth), size=24, anchor="middle"))
    compass_points = [(position(index, len(depths)), y_position(compass[depth]["accuracy"])) for index, depth in enumerate(depths)]
    jev_points = [(position(index, len(depths)), y_position(jev[depth]["accuracy"])) for index, depth in enumerate(depths)]
    qwen_points = [(position(index, len(depths)), y_position(qwen[depth]["accuracy"])) for index, depth in enumerate(depths)]
    parts.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in compass_points) + f'" fill="none" stroke="{COMPASS}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>')
    parts.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in jev_points) + f'" fill="none" stroke="{JEV}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>')
    parts.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in qwen_points) + f'" fill="none" stroke="{QWEN}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>')
    for x, y in compass_points:
        parts.append(point(x, y, COMPASS))
    for x, y in jev_points:
        parts.append(point(x, y, JEV))
    for x, y in qwen_points:
        parts.append(point(x, y, QWEN))
    parts.extend([
        text(72, 662, "Numbers below axis: Compass / Jev / Qwen correct out of 20", size=16, fill=MUTED),
        line(72, 706, 110, 706, stroke=COMPASS, **{"stroke-width": 5, "stroke-linecap": "round"}),
        text(122, 713, "Compass 0.2.0", size=24, fill=COMPASS, weight=600),
        line(72, 749, 110, 749, stroke=JEV, **{"stroke-width": 5, "stroke-linecap": "round"}),
        text(122, 756, "Jev 1.13.0", size=24, fill=JEV, weight=600),
        line(72, 792, 110, 792, stroke=QWEN, **{"stroke-width": 5, "stroke-linecap": "round"}),
        text(122, 799, f"{qwen_data['qwen_thinking']['model']['model'].split('/')[-1]}, thinking on", size=24, fill=QWEN, weight=600),
        text(72, 856, "20 balanced yes/no questions at each length. Compass emits no answer tokens.", size=18, fill=MUTED),
        text(72, 890, "Qwen can generate a working trace before it commits to an answer.", size=18, fill=MUTED),
    ])
    for index, depth in enumerate(depths):
        x = position(index, len(depths))
        c, j, q = compass[depth], jev[depth], qwen[depth]
        parts.append(text(x, BOTTOM + 83, f"{c['correct']}/{c['total']}   {j['correct']}/{j['total']}   {q['correct']}/{q['total']}", size=16, fill=MUTED, anchor="middle"))
        tokens = q["reasoning_tokens"]
        if tokens is not None:
            parts.append(text(x, TOP - 38, f"{tokens:.0f} tokens", size=15, fill=TOKEN, anchor="middle", weight=600))
        if q["unanswered"]:
            parts.append(text(x, TOP - 14, f"{q['unanswered']}/{q['total']} gave no final answer", size=14, fill=TOKEN, anchor="middle"))
    parts.append('</svg>')
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
    args.out.write_text(render(compass_data, jev_data, qwen_data), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
