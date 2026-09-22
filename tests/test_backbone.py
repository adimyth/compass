"""Integration test of the backbone scorer on the smallest Qwen3.5 checkpoint. Skipped when torch or the weights are unavailable."""

import itertools
import math

import pytest

torch = pytest.importorskip("torch")

from compass.calibration import Calibrator
from compass.contract import compile_request, decide

MODEL = "Qwen/Qwen3.5-0.8B"


@pytest.fixture(scope="module")
def scorer():
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache(MODEL, "config.json"), str):
        pytest.skip(f"{MODEL} is not in the local Hugging Face cache")
    from compass.backbone import BackboneScorer

    return BackboneScorer(MODEL)


STATE = "Ticket 4411: since this morning's release the export button on the reports page returns HTTP 500 for every user. Billing is unaffected."
ROUTING = {
    "type": "choice",
    "instructions": "Which team should handle this ticket?",
    "criteria": {"billing": "Charges, invoices, refunds", "engineering": "Bugs, errors, outages", "sales": "Pricing and new accounts"},
}
SEVERITY = {"type": "score", "instructions": "How severe is the incident?", "criteria": ["Cosmetic", "Degraded with a workaround", "Core function blocked"]}
BILLING = {"type": "noul", "instructions": "Is billing affected?", "criteria": {"true": "The ticket says billing is affected", "false": "The ticket says billing is not affected"}}


def test_prompt_segments_split_cleanly(scorer):
    _, req = compile_request({"state": STATE, "questions": {"team": ROUTING}})
    prefix, [(rubric, branches)] = scorer.render(req)
    assert prefix.endswith("</document>\n\n")
    assert rubric.startswith("Question: Which team") and "  billing: Charges" in rubric and rubric.index("billing") < rubric.index("engineering")
    assert len(branches) == 3 and all("Proposed answer: the correct answer is" in b for b in branches)
    assert branches[0].endswith(prefix[-0:] or "") and "<|im_start|>assistant" in branches[0] or "assistant" in branches[0]
    assert "␞" not in prefix + rubric + "".join(branches)


def test_answers_are_valid_and_tokens_are_counted(scorer):
    out = decide({"state": STATE, "questions": {"team": ROUTING, "sev": SEVERITY, "billing": BILLING}}, scorer, Calibrator())
    for qid, labels in (("team", ROUTING["criteria"]), ("sev", ["0", "1", "2"])):
        probs = out["answers"][qid]["probabilities"]
        assert set(probs) == set(labels) and math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)
    assert 0.0 <= out["answers"]["billing"]["noul"] <= 1.0
    _, req = compile_request({"state": STATE, "questions": {"team": ROUTING, "sev": SEVERITY, "billing": BILLING}})
    prefix, questions = scorer.render(req)
    expected = len(scorer.tok(prefix, add_special_tokens=False).input_ids) + sum(
        len(scorer.tok(rubric, add_special_tokens=False).input_ids) + sum(len(scorer.tok(b, add_special_tokens=False).input_ids) for b in branches)
        for rubric, branches in questions
    )
    assert out["usage"] == {"input_tokens": expected, "output_tokens": 0}
    assert out["model"].startswith("compass-vocab-qwen3.5-0.8b")


def test_option_order_cannot_change_the_answer(scorer):
    reference = decide({"state": STATE, "questions": {"team": ROUTING}}, scorer, Calibrator())
    for perm in itertools.permutations(ROUTING["criteria"].items()):
        q = {**ROUTING, "criteria": dict(perm)}
        assert decide({"state": STATE, "questions": {"team": q}}, scorer, Calibrator()) == reference


def test_forked_branches_match_unforked_scoring(scorer):
    """Scoring through the three-level fork must equal scoring each candidate as one flat sequence (up to bf16 noise)."""
    _, req = compile_request({"state": STATE, "questions": {"team": ROUTING}})
    forked = scorer.score(req).logits["team"]
    prefix, [(rubric, branches)] = scorer.render(req)
    flat = []
    for b in branches:
        enc = scorer.tok(prefix + rubric + b, return_tensors="pt", add_special_tokens=False).to(scorer.device)
        with torch.no_grad():
            logits = scorer.model(**enc).logits[:, -1]
        flat.append(scorer._logodds(logits).item())
    assert all(abs(a - b) < 0.15 for a, b in zip(forked, flat)), (forked, flat)
