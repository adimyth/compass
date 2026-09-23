"""The /v1/systemone contract: request validation, rubric compilation and response assembly.

Everything here is deterministic and independent of the model. A scorer only ever sees a `CompiledRequest` and returns one raw logit per candidate; this module owns the wire format, so a scorer cannot produce an invalid answer shape. See the README and api/systemone.schema.json.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Protocol

from .calibration import Calibrator

QUESTION_TYPES = ("choice", "score", "noul")
LATEST_ALIAS = "compass-latest"
MAX_QUESTIONS = 100
MIN_OPTIONS, MAX_OPTIONS = 2, 64
# Noul is scored as a symmetric pair of propositions, in this fixed order.
NOUL_KEYS = ("false", "true")
NOUL_DEFAULT_TEXT = {"false": "no", "true": "yes"}


class ContractError(ValueError):
    """The request breaks the contract (HTTP 422)."""


class UnknownModel(ValueError):
    """The request names a model this server does not serve (HTTP 404)."""


class ModelOutputError(RuntimeError):
    """The scorer returned something that cannot become a distribution (HTTP 500). Never repaired."""


@dataclass(frozen=True)
class Candidate:
    key: str
    text: str


@dataclass(frozen=True)
class CompiledQuestion:
    qid: str
    type: str
    instructions: str
    # Choice: sorted by key, so option order in the request cannot reach the model.
    # Score: level order, which carries meaning. Noul: NOUL_KEYS order.
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class CompiledRequest:
    state: str
    questions: tuple[CompiledQuestion, ...]


@dataclass(frozen=True)
class ScoreOutput:
    logits: dict[str, list[float]]  # qid -> one raw logit per candidate, in candidate order
    input_tokens: int  # every token the backbone processed for this request


class Scorer(Protocol):
    model_id: str

    def score(self, request: CompiledRequest) -> ScoreOutput: ...


def render(value: Any, where: str) -> str:
    """Serialise state, instructions or a criterion. JSON structure keeps its key order, so a document keeps its reading order."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    raise ContractError(f"{where} must be a string, object or array, got {type(value).__name__}")


def _require_keys(obj: dict, allowed: set, required: set, where: str) -> None:
    missing = sorted(required - obj.keys())
    if missing:
        raise ContractError(f"{where} is missing {', '.join(missing)}")
    extra = sorted(obj.keys() - allowed)
    if extra:
        raise ContractError(f"{where} has unknown field(s) {', '.join(extra)}")


def _option_count(n: int, where: str) -> None:
    if not MIN_OPTIONS <= n <= MAX_OPTIONS:
        raise ContractError(f"{where} needs {MIN_OPTIONS}-{MAX_OPTIONS} entries, got {n}")


def compile_question(qid: str, q: Any) -> CompiledQuestion:
    where = f"questions.{qid}"
    if not isinstance(q, dict):
        raise ContractError(f"{where} must be an object")
    qtype = q.get("type")
    if qtype not in QUESTION_TYPES:
        raise ContractError(f"{where}.type must be one of {', '.join(QUESTION_TYPES)}")
    required = {"type", "instructions"} | ({"criteria"} if qtype != "noul" else set())
    _require_keys(q, {"type", "instructions", "criteria"}, required, where)
    instructions = render(q["instructions"], f"{where}.instructions")
    criteria = q.get("criteria")

    if qtype == "choice":
        if not isinstance(criteria, dict):
            raise ContractError(f"{where}.criteria must map option keys to descriptions")
        _option_count(len(criteria), f"{where}.criteria")
        if any(not key for key in criteria):
            raise ContractError(f"{where}.criteria has an empty option key")
        candidates = tuple(Candidate(key, render(criteria[key], f"{where}.criteria.{key}")) for key in sorted(criteria))
    elif qtype == "score":
        if not isinstance(criteria, list):
            raise ContractError(f"{where}.criteria must be a list of level descriptions")
        _option_count(len(criteria), f"{where}.criteria")
        candidates = tuple(Candidate(str(i), render(level, f"{where}.criteria[{i}]")) for i, level in enumerate(criteria))
    else:
        if criteria is None:
            criteria = {}
        if not isinstance(criteria, dict):
            raise ContractError(f"{where}.criteria must be an object with 'true' and/or 'false'")
        _require_keys(criteria, set(NOUL_KEYS), set(), f"{where}.criteria")
        candidates = tuple(
            Candidate(key, render(criteria[key], f"{where}.criteria.{key}") if key in criteria else NOUL_DEFAULT_TEXT[key])
            for key in NOUL_KEYS
        )
    return CompiledQuestion(qid=qid, type=qtype, instructions=instructions, candidates=candidates)


def compile_request(body: Any) -> tuple[str | None, CompiledRequest]:
    """Validate a request body. Returns the requested model id (or None) and the compiled request."""
    if not isinstance(body, dict):
        raise ContractError("request body must be a JSON object")
    _require_keys(body, {"model", "state", "questions"}, {"state", "questions"}, "request")
    model = body.get("model")
    if model is not None and not isinstance(model, str):
        raise ContractError("model must be a string")
    questions = body["questions"]
    if not isinstance(questions, dict) or not questions:
        raise ContractError("questions must be a non-empty object")
    if len(questions) > MAX_QUESTIONS:
        raise ContractError(f"at most {MAX_QUESTIONS} questions per request, got {len(questions)}")
    if any(not qid for qid in questions):
        raise ContractError("question ids must be non-empty")
    compiled = tuple(compile_question(qid, q) for qid, q in questions.items())
    return model, CompiledRequest(state=render(body["state"], "state"), questions=compiled)


def resolve_model(requested: str | None, served: str) -> None:
    if requested not in (None, LATEST_ALIAS, served):
        raise UnknownModel(f"unknown model {requested!r}; this server serves {served!r} (alias {LATEST_ALIAS!r})")


def confidence(probs: list[float]) -> float:
    """How peaked a distribution is: 0 when uniform, 1 when all mass is on one entry."""
    n = len(probs)
    return min(1.0, max(0.0, (n * max(probs) - 1.0) / (n - 1)))


def _check_output(request: CompiledRequest, output: ScoreOutput) -> None:
    if set(output.logits) != {q.qid for q in request.questions}:
        raise ModelOutputError("scorer answered a different set of questions than it was asked")
    for q in request.questions:
        logits = output.logits[q.qid]
        if len(logits) != len(q.candidates):
            raise ModelOutputError(f"scorer returned {len(logits)} logits for {len(q.candidates)} candidates in {q.qid!r}")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in logits):
            raise ModelOutputError(f"scorer returned a non-finite logit for {q.qid!r}")
    if isinstance(output.input_tokens, bool) or not isinstance(output.input_tokens, int) or output.input_tokens < 0:
        raise ModelOutputError("scorer returned an invalid input token count")


def _answer(q: CompiledQuestion, probs: list[float]) -> dict:
    keys = [c.key for c in q.candidates]
    by_key = dict(zip(keys, probs))
    if q.type == "noul":
        return {"type": "noul", "noul": by_key["true"]}
    if q.type == "choice":
        # Ties go to the lexicographically smallest key, the same rule JevBench's argmax uses.
        choice = min(keys, key=lambda k: (-by_key[k], k))
        return {"type": "choice", "choice": choice, "confidence": confidence(probs), "probabilities": by_key}
    return {
        "type": "score",
        "score": math.fsum(i * p for i, p in enumerate(probs)),
        "confidence": confidence(probs),
        "legend": {c.key: c.text for c in q.candidates},
        "probabilities": by_key,
    }


def assemble_response(request: CompiledRequest, output: ScoreOutput, calibrator: Calibrator, model_id: str) -> dict:
    _check_output(request, output)
    answers = {q.qid: _answer(q, calibrator.probabilities(q.type, output.logits[q.qid])) for q in request.questions}
    return {"model": model_id, "answers": answers, "usage": {"input_tokens": output.input_tokens, "output_tokens": 0}}


def decide(body: Any, scorer: Scorer, calibrator: Calibrator) -> dict:
    """One /v1/systemone call: validate, score, calibrate, assemble."""
    model, request = compile_request(body)
    resolve_model(model, scorer.model_id)
    return assemble_response(request, scorer.score(request), calibrator, scorer.model_id)
