"""An untrained head must reproduce the release fusion exactly, and a saved head must round-trip."""

import math

import pytest

torch = pytest.importorskip("torch")

from compass.contract import compile_request
from compass.head import CompassHead

MODEL = "Qwen/Qwen3.5-0.8B"
STATE = "Ticket 4411: since this morning's release the export button returns HTTP 500 for every user. Billing is unaffected."
ROUTING = {"type": "choice", "instructions": "Which team should handle this ticket?", "criteria": {"billing": "Charges, invoices, refunds", "engineering": "Bugs, errors, outages", "sales": "Pricing and new accounts"}}


def test_untrained_head_equals_fusion(tmp_path):
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache(MODEL, "config.json"), str):
        pytest.skip(f"{MODEL} is not cached")
    from compass.backbone import BackboneScorer

    fusion = BackboneScorer(MODEL, readout="fusion", fusion_weight=0.5)
    _, req = compile_request({"state": STATE, "questions": {"d": ROUTING}})
    reference = fusion.score(req).logits["d"]
    CompassHead(fusion.model.config.hidden_size).save(str(tmp_path / "head"))
    headed = BackboneScorer(MODEL, head_path=str(tmp_path / "head"))
    got = headed.score(req).logits["d"]
    assert all(math.isclose(a, b, abs_tol=1e-3) for a, b in zip(got, reference)), (got, reference)
    assert headed.model_id.startswith("compass-head-")


def test_head_round_trip(tmp_path):
    head = CompassHead(16, width=8)
    with torch.no_grad():
        head.mlp[-1].weight.fill_(0.3)
        head.readout_weights.copy_(torch.tensor([0.7, 0.2]))
    head.eval()
    head.save(str(tmp_path / "h"))
    again = CompassHead.load(str(tmp_path / "h"), torch.device("cpu"))
    x = torch.randn(4, 16)
    assert torch.allclose(head(x, torch.zeros(4), torch.zeros(4), torch.zeros(4, dtype=torch.long)), again(x, torch.zeros(4), torch.zeros(4), torch.zeros(4, dtype=torch.long)))
