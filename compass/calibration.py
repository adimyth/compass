"""Calibration layer: turns raw candidate logits into probabilities.

Temperatures are fitted once, after the decision model is frozen, on a validation split that shares nothing with training data or JevBench (SPEC.md §5, §7). The default of 1.0 for every type is the uncalibrated identity, used until a fitted file exists.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

DEFAULT_TEMPERATURES = {"choice": 1.0, "score": 1.0, "noul": 1.0}


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    """Float64 softmax, not rounded. The result sums to 1 within float error."""
    scaled = [x / temperature for x in logits]
    top = max(scaled)
    exps = [math.exp(x - top) for x in scaled]
    total = math.fsum(exps)
    return [e / total for e in exps]


@dataclass(frozen=True)
class Calibrator:
    temperatures: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TEMPERATURES))
    source: str = "identity (uncalibrated)"

    def __post_init__(self) -> None:
        for qtype, t in self.temperatures.items():
            if not (isinstance(t, (int, float)) and math.isfinite(t) and t > 0):
                raise ValueError(f"temperature for {qtype!r} must be a positive finite number, got {t!r}")
        missing = set(DEFAULT_TEMPERATURES) - set(self.temperatures)
        if missing:
            raise ValueError(f"calibration has no temperature for {', '.join(sorted(missing))}")

    def probabilities(self, qtype: str, logits: list[float]) -> list[float]:
        return softmax(logits, self.temperatures[qtype])

    @classmethod
    def load(cls, path: str) -> "Calibrator":
        """Read a fitted calibration file: {"temperatures": {"choice": t, "score": t, "noul": t}, ...}."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(temperatures={k: float(v) for k, v in data["temperatures"].items()}, source=path)
