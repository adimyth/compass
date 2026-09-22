import copy
import itertools
import math

import pytest

from compass.calibration import Calibrator, softmax
from compass.contract import (
    ContractError,
    ModelOutputError,
    ScoreOutput,
    UnknownModel,
    compile_request,
    decide,
)
from compass.scorer import UniformScorer


class PositionBiasedScorer:
    """A deliberately order-sensitive fake: each logit depends on the candidate's text and its position. Output is order-invariant only if the compiler canonicalises order."""

    model_id = "test-positional"

    def score(self, request):
        logits = {q.qid: [0.3 * len(c.text) - 0.7 * i for i, c in enumerate(q.candidates)] for q in request.questions}
        return ScoreOutput(logits=logits, input_tokens=17)


class FixedScorer:
    model_id = "test-fixed"

    def __init__(self, logits, input_tokens=5):
        self.logits, self.input_tokens = logits, input_tokens

    def score(self, request):
        return ScoreOutput(logits=self.logits, input_tokens=self.input_tokens)


ROUTING = {
    "type": "choice",
    "instructions": "Which team should handle this ticket?",
    "criteria": {
        "billing": "Payment, invoice or subscription issues",
        "technical": "Bugs, errors or integration problems",
        "sales": "Pricing questions or new accounts",
        "legal": "Contracts or compliance",
    },
}
SEVERITY = {
    "type": "score",
    "instructions": "How severe is the reported problem?",
    "criteria": ["Cosmetic", "Degraded with a workaround", "Blocked, no workaround"],
}
URGENT = {"type": "noul", "instructions": "Does the message ask for a same-day response?"}


def body(**questions):
    return {"model": "compass-latest", "state": "Our Stripe sync has failed for three days and we are losing orders.", "questions": questions}


def jevbench_strict_valid(probs, labels):
    """JevBench's strict rule (jevbench/scoring.py): exact keys, values in [0, 1], sum within 1e-3."""
    return set(probs) == set(labels) and all(0.0 <= v <= 1.0 for v in probs.values()) and abs(sum(probs.values()) - 1.0) <= 1e-3


def test_all_three_types_return_valid_typed_answers():
    out = decide(body(team=ROUTING, severity=SEVERITY, urgent=URGENT), PositionBiasedScorer(), Calibrator())
    assert out["model"] == "test-positional"
    assert out["usage"] == {"input_tokens": 17, "output_tokens": 0}

    team = out["answers"]["team"]
    assert team["type"] == "choice"
    assert jevbench_strict_valid(team["probabilities"], ROUTING["criteria"])
    assert team["choice"] == max(team["probabilities"], key=team["probabilities"].get)
    assert 0.0 <= team["confidence"] <= 1.0

    sev = out["answers"]["severity"]
    assert sev["type"] == "score"
    assert jevbench_strict_valid(sev["probabilities"], ["0", "1", "2"])
    assert sev["legend"] == {"0": "Cosmetic", "1": "Degraded with a workaround", "2": "Blocked, no workaround"}
    assert math.isclose(sev["score"], sum(int(k) * p for k, p in sev["probabilities"].items()))

    urgent = out["answers"]["urgent"]
    assert set(urgent) == {"type", "noul"}
    assert isinstance(urgent["noul"], float) and 0.0 <= urgent["noul"] <= 1.0


def test_choice_output_is_identical_under_every_option_order():
    items = list(ROUTING["criteria"].items())
    reference = decide(body(team=ROUTING), PositionBiasedScorer(), Calibrator())
    for perm in itertools.permutations(items):
        q = {**ROUTING, "criteria": dict(perm)}
        assert decide(body(team=q), PositionBiasedScorer(), Calibrator()) == reference


def test_score_levels_keep_their_order():
    _, compiled = compile_request(body(severity=SEVERITY))
    assert [c.text for c in compiled.questions[0].candidates] == SEVERITY["criteria"]
    assert [c.key for c in compiled.questions[0].candidates] == ["0", "1", "2"]


def test_noul_is_a_symmetric_pair_using_criteria_when_given():
    q = {**URGENT, "criteria": {"true": "Asks for a reply today", "false": "No deadline stated"}}
    _, compiled = compile_request(body(urgent=q))
    assert [(c.key, c.text) for c in compiled.questions[0].candidates] == [("false", "No deadline stated"), ("true", "Asks for a reply today")]
    _, default = compile_request(body(urgent=URGENT))
    assert [(c.key, c.text) for c in default.questions[0].candidates] == [("false", "no"), ("true", "yes")]


def test_noul_probability_is_p_true():
    out = decide(body(urgent=URGENT), FixedScorer({"urgent": [0.0, math.log(3.0)]}), Calibrator())
    assert math.isclose(out["answers"]["urgent"]["noul"], 0.75)


def test_choice_ties_go_to_the_smallest_key_like_jevbench():
    out = decide(body(team=ROUTING), UniformScorer(), Calibrator())
    assert out["answers"]["team"]["choice"] == "billing"
    assert out["answers"]["team"]["confidence"] == 0.0


def test_structured_instructions_and_criteria_are_serialised_in_order():
    q = {"type": "choice", "instructions": {"task": "route", "rule": "prefer specialists"}, "criteria": {"b": {"covers": "x", "not": "y"}, "a": ["one", "two"]}}
    _, compiled = compile_request({"state": {"z": 1, "a": 2}, "questions": {"q": q}})
    assert compiled.state == '{"z": 1, "a": 2}'
    assert compiled.questions[0].instructions == '{"task": "route", "rule": "prefer specialists"}'
    assert [(c.key, c.text) for c in compiled.questions[0].candidates] == [("a", '["one", "two"]'), ("b", '{"covers": "x", "not": "y"}')]


def test_temperature_flattens_the_distribution():
    logits = [2.0, 0.0, -1.0]
    sharp, flat = softmax(logits, 1.0), softmax(logits, 3.0)
    assert max(flat) < max(sharp)
    assert math.isclose(sum(flat), 1.0, abs_tol=1e-12)


def test_probabilities_stay_valid_for_extreme_logits():
    out = decide(body(team=ROUTING), FixedScorer({"team": [1e6, -1e6, 0.0, 700.0]}), Calibrator())
    assert jevbench_strict_valid(out["answers"]["team"]["probabilities"], ROUTING["criteria"])


def test_many_options_and_many_questions_are_accepted():
    wide = {"type": "choice", "instructions": "Pick one.", "criteria": {f"opt_{i:02d}": f"option {i}" for i in range(64)}}
    questions = {f"q{i}": wide for i in range(100)}
    out = decide({"state": "x", "questions": questions}, PositionBiasedScorer(), Calibrator())
    assert len(out["answers"]) == 100
    assert all(jevbench_strict_valid(a["probabilities"], wide["criteria"]) for a in out["answers"].values())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: [],
        lambda b: b.pop("state"),
        lambda b: b.pop("questions"),
        lambda b: b.update(questions={}),
        lambda b: b.update(extra=1),
        lambda b: b.update(model=3),
        lambda b: b.update(state=42),
        lambda b: b["questions"].update({"": URGENT}),
        lambda b: b["questions"]["q"].update(type="multi"),
        lambda b: b["questions"]["q"].pop("instructions"),
        lambda b: b["questions"]["q"].update(instructions=None),
        lambda b: b["questions"]["q"].update(criteria={"only": "one"}),
        lambda b: b["questions"]["q"].update(criteria=["a", "b"]),
        lambda b: b["questions"]["q"].update(criteria={"": "empty key", "b": "fine"}),
        lambda b: b["questions"]["q"].update(criteria={"a": 1, "b": "fine"}),
        lambda b: b["questions"]["q"].update(options=["a", "b"]),
        lambda b: b["questions"].update(s={**SEVERITY, "criteria": ["only one"]}),
        lambda b: b["questions"].update(s={**SEVERITY, "criteria": {"0": "a", "1": "b"}}),
        lambda b: b["questions"].update(n={**URGENT, "criteria": {"true": "a", "maybe": "b"}}),
        lambda b: b["questions"].update(n={**URGENT, "criteria": ["a", "b"]}),
        lambda b: b.update(questions={f"q{i}": URGENT for i in range(101)}),
    ],
)
def test_malformed_requests_are_rejected(mutate):
    b = copy.deepcopy(body(q=ROUTING))
    result = mutate(b)
    with pytest.raises(ContractError):
        decide(result if isinstance(result, list) else b, UniformScorer(), Calibrator())


def test_unknown_model_is_refused_and_aliases_resolve():
    for model in (None, "compass-latest", "compass-uniform-0.0.0"):
        decide({**body(u=URGENT), "model": model}, UniformScorer(), Calibrator())
    with pytest.raises(UnknownModel):
        decide({**body(u=URGENT), "model": "jev-latest"}, UniformScorer(), Calibrator())


@pytest.mark.parametrize(
    "logits, tokens",
    [
        ({"team": [0.0, float("nan"), 0.0, 0.0]}, 1),
        ({"team": [0.0, float("inf"), 0.0, 0.0]}, 1),
        ({"team": [0.0, 0.0]}, 1),
        ({"other": [0.0, 0.0, 0.0, 0.0]}, 1),
        ({"team": [0.0, True, 0.0, 0.0]}, 1),
        ({"team": [0.0, 0.0, 0.0, 0.0]}, -1),
    ],
)
def test_bad_scorer_output_fails_closed(logits, tokens):
    with pytest.raises(ModelOutputError):
        decide(body(team=ROUTING), FixedScorer(logits, tokens), Calibrator())


def test_calibrator_rejects_bad_temperatures():
    with pytest.raises(ValueError):
        Calibrator(temperatures={"choice": 0.0, "score": 1.0, "noul": 1.0})
    with pytest.raises(ValueError):
        Calibrator(temperatures={"choice": 1.0})
