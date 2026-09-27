"""Compare Compass's fixed-compute scorer with Qwen3.5-4B in thinking mode on reporting chains.

The suite has two uses. A depth-three item is a compact worked example for an essay or demo. The length sweep measures how accuracy changes as the number of reporting links grows. Every item has two unrelated distractor chains and a balanced yes/no answer. The generator is independent of Compass training, calibration, and release-selection data.

Run on a machine with an accelerator and the release model cached:

    uv run python scripts/manager_chain_eval.py --samples 20 --lengths 1 2 3 4 6 8 12 --out dev/results/manager-chain-2026-09-27.json
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from compass import release
from compass.calibration import Calibrator
from compass.contract import decide


NAMES = [
    "Aadi", "Aiko", "Amir", "Anya", "Arun", "Bela", "Caro", "Chen", "Cleo", "Dara", "Dina", "Eli", "Ena", "Faye", "Gabe", "Gus",
    "Hana", "Hugo", "Inez", "Ira", "Jae", "Jin", "Jo", "Kavi", "Kira", "Lena", "Lio", "Mara", "Mina", "Mo", "Nia", "Niko",
    "Noa", "Omar", "Oto", "Pia", "Quin", "Ravi", "Rhea", "Sana", "Sol", "Tara", "Ted", "Tia", "Uma", "Una", "Vic", "Wen",
    "Xia", "Yara", "Zane", "Zia", "Ava", "Bea", "Cai", "Dee", "Emi", "Finn", "Gia", "Hal", "Ian", "June", "Kai", "Luz",
]


@dataclass(frozen=True)
class Task:
    id: str
    depth: int
    state: str
    question: str
    expected: str


def make_task(rng: random.Random, depth: int, index: int) -> Task:
    """Make a balanced yes/no chain task with two unrelated distractor chains."""
    needed = 3 * (depth + 1)
    if needed > len(NAMES):
        raise ValueError(f"depth {depth} needs {needed} distinct names; the generator has {len(NAMES)}")
    people = rng.sample(NAMES, needed)
    chains = [people[offset:offset + depth + 1] for offset in range(0, needed, depth + 1)]
    main = chains[0]
    expected_yes = index % 2 == 0
    target = main[-1] if expected_yes else (main[-2] if depth > 1 else chains[1][-1])
    lines = []
    for chain in chains:
        lines.extend(f"{person} reports to {manager}." for person, manager in zip(chain, chain[1:]))
    rng.shuffle(lines)
    state = "Organisation chart. Each line gives one direct reporting relationship.\n\n" + "\n".join(f"- {line}" for line in lines)
    question = f"Is {target} exactly {depth} manager links above {main[0]}?"
    return Task(id=f"chain-d{depth}-q{index:02d}", depth=depth, state=state, question=question, expected="yes" if expected_yes else "no")


def build_tasks(lengths: list[int], samples: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    return [make_task(rng, depth, index) for depth in lengths for index in range(samples)]


def compass_question(task: Task) -> dict:
    return {
        "type": "noul",
        "instructions": task.question,
        "criteria": {
            "true": f"{task.question} Answer yes only if following exactly {task.depth} reporting links reaches that person.",
            "false": f"{task.question} Answer no if following exactly {task.depth} reporting links does not reach that person.",
        },
    }


def qwen_prompt(task: Task) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Solve the reporting-chain question from the supplied chart. In every statement, ‘X reports to Y’ means Y is X’s direct manager. Start at the named employee and follow the reporting links exactly the requested number of times. Keep your reasoning trace short: write only the relevant path and its link count. Do not restate the chart, enumerate unrelated relationships, or revisit the definition. End with one line in exactly this format: FINAL: YES or FINAL: NO.",
        },
        {"role": "user", "content": f"{task.state}\n\nQuestion: {task.question}"},
    ]


def parse_qwen_answer(text: str) -> str | None:
    """Read Qwen's final yes/no only after it closes its thinking section."""
    if "</think>" not in text:
        return None
    final = text.rsplit("</think>", 1)[-1]
    final_match = re.findall(r"FINAL\s*:\s*(yes|no)\b", final, flags=re.IGNORECASE)
    if final_match:
        return final_match[-1].lower()
    matches = re.findall(r"\b(yes|no)\b", final, flags=re.IGNORECASE)
    return matches[-1].lower() if matches else None


def reasoning_token_count(tokenizer, text: str) -> int:
    """Count generated tokens before </think>; use the full completion when the trace hits the cap."""
    thought = text.split("</think>", 1)[0]
    return len(tokenizer(thought, add_special_tokens=False).input_ids)


def run_qwen(tasks: list[Task], device: str, max_new_tokens: int, seed: int) -> tuple[list[dict], dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(release.BACKBONE, revision=release.BACKBONE_REVISION)
    model = AutoModelForCausalLM.from_pretrained(release.BACKBONE, revision=release.BACKBONE_REVISION, dtype=torch.bfloat16).to(device).eval()
    torch.manual_seed(seed)
    rows = []
    for task in tasks:
        prompt = tokenizer.apply_chat_template(qwen_prompt(task), tokenize=False, add_generation_prompt=True, enable_thinking=True)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**encoded, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=max_new_tokens, use_cache=True)
        elapsed = time.perf_counter() - started
        text = tokenizer.decode(generated[0, encoded.input_ids.shape[1]:], skip_special_tokens=False)
        answer = parse_qwen_answer(text)
        rows.append({
            "id": task.id,
            "depth": task.depth,
            "expected": task.expected,
            "predicted": answer,
            "correct": answer == task.expected,
            "reasoning_tokens": reasoning_token_count(tokenizer, text),
            "generated_tokens": int(generated.shape[1] - encoded.input_ids.shape[1]),
            "truncated": int(generated.shape[1] - encoded.input_ids.shape[1]) >= max_new_tokens and "</think>" not in text,
            "latency_seconds": elapsed,
            "completion": text,
        })
        print(f"Qwen    {task.id}: expected={task.expected} predicted={answer or 'PARSE_FAIL'} tokens={rows[-1]['reasoning_tokens']} {'ok' if rows[-1]['correct'] else 'MISS'}")
    del model
    del tokenizer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return rows, {"model": release.BACKBONE, "thinking": True, "max_new_tokens": max_new_tokens, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": seed}


def run_compass(tasks: list[Task], device: str) -> tuple[list[dict], dict]:
    from compass.backbone import BackboneScorer

    scorer = BackboneScorer(
        release.BACKBONE,
        revision=release.BACKBONE_REVISION,
        device=device,
        readout=release.READOUT,
        fusion_weight=release.FUSION_WEIGHT,
        adapter=release.ADAPTER,
        adapter_revision=release.ADAPTER_REVISION,
    )
    calibrator = Calibrator.load(release.CALIBRATION)
    rows = []
    for task in tasks:
        started = time.perf_counter()
        answer = decide({"state": task.state, "questions": {"chain": compass_question(task)}}, scorer, calibrator)["answers"]["chain"]
        elapsed = time.perf_counter() - started
        predicted = "yes" if answer["noul"] >= 0.5 else "no"
        rows.append({
            "id": task.id,
            "depth": task.depth,
            "expected": task.expected,
            "predicted": predicted,
            "correct": predicted == task.expected,
            "probability_yes": answer["noul"],
            "latency_seconds": elapsed,
        })
        print(f"Compass {task.id}: expected={task.expected} predicted={predicted} p_yes={answer['noul']:.3f} {'ok' if rows[-1]['correct'] else 'MISS'}")
    model = {"model": scorer.model_id, "readout": release.READOUT, "adapter": release.ADAPTER, "output_tokens": 0}
    del scorer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return rows, model


def summary(rows: list[dict], lengths: list[int], include_reasoning: bool) -> list[dict]:
    output = []
    for depth in lengths:
        group = [row for row in rows if row["depth"] == depth]
        record = {"depth": depth, "correct": sum(row["correct"] for row in group), "total": len(group), "accuracy": sum(row["correct"] for row in group) / len(group)}
        if include_reasoning:
            record["median_reasoning_tokens"] = statistics.median(row["reasoning_tokens"] for row in group)
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 2 or args.samples % 2:
        raise SystemExit("--samples must be an even number of at least 2 so every depth has balanced yes/no labels")
    if any(depth < 1 for depth in args.lengths):
        raise SystemExit("--lengths must contain positive integers")
    tasks = build_tasks(args.lengths, args.samples, args.seed)
    print(f"Running {len(tasks)} balanced tasks on {args.device}: lengths={args.lengths}, samples per length={args.samples}")
    qwen_rows, qwen_model = run_qwen(tasks, args.device, args.max_new_tokens, args.seed)
    compass_rows, compass_model = run_compass(tasks, args.device)
    result = {
        "suite": "manager_chain",
        "seed": args.seed,
        "lengths": args.lengths,
        "samples_per_length": args.samples,
        "tasks": [asdict(task) for task in tasks],
        "qwen_thinking": {"model": qwen_model, "summary": summary(qwen_rows, args.lengths, include_reasoning=True), "rows": qwen_rows},
        "compass": {"model": compass_model, "summary": summary(compass_rows, args.lengths, include_reasoning=False), "rows": compass_rows},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\nResults")
    print("depth | Compass | Qwen thinking | Qwen median reasoning tokens")
    for compass, qwen in zip(result["compass"]["summary"], result["qwen_thinking"]["summary"]):
        print(f"{compass['depth']:>5} | {compass['correct']:>2}/{compass['total']:<2} ({compass['accuracy']:.0%}) | {qwen['correct']:>2}/{qwen['total']:<2} ({qwen['accuracy']:.0%}) | {qwen['median_reasoning_tokens']:.0f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
