"""Scorers: the model side of Compass.

A scorer receives a `CompiledRequest` and returns one raw logit per candidate plus the number of input tokens it processed. `backbone` is the real one (compass/backbone.py); `uniform` exercises the serving path without a model.
"""

from __future__ import annotations

from .contract import CompiledRequest, ScoreOutput


class UniformScorer:
    """Scores every candidate equally and reads nothing.

    It exists to exercise the serving path end to end. It is not a model and must never be submitted: every answer is uniform, and its choice answers fall to the smallest option key.
    """

    model_id = "compass-uniform-0.0.0"

    def score(self, request: CompiledRequest) -> ScoreOutput:
        return ScoreOutput(logits={q.qid: [0.0] * len(q.candidates) for q in request.questions}, input_tokens=0)


def build(name: str, **kwargs):
    if name == "uniform":
        return UniformScorer()
    if name == "backbone":
        from .backbone import BackboneScorer  # imports torch; kept out of the uniform path

        return BackboneScorer(**{k: v for k, v in kwargs.items() if v is not None})
    raise ValueError(f"unknown scorer {name!r}")


SCORERS = ("uniform", "backbone")
