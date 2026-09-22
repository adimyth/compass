"""The backbone scorer: candidate verification over a shared read of the state.

Layout (STRATEGY.md, decisions 2 and 3). The state is prefilled once. For each question the cache is forked and the instructions plus the rubric, in canonical candidate order, are appended. For each candidate the question cache is forked again and a short branch stating the candidate as a proposition is appended; all branches of a question run as one batch. The raw logit of a candidate is the model's log-odds of "yes" against "no" at the readout position (Stage A, vocabulary readout) or the output of a trained head on the hidden state there (Stage B).

Reported input tokens are every token the backbone processed. The prompt wording below is Compass's own and is part of the versioned model.
"""

from __future__ import annotations

import copy
import math

import torch

from .contract import CompiledQuestion, CompiledRequest, ScoreOutput

PROMPT_VERSION = "compass-prompt-2"
SPLIT = "␞"  # record separator, never expected in real text; marks prefix/branch boundaries

SYSTEM = (
    "You are a decision verifier. You will read a document, a question about it and a rubric listing every allowed answer. "
    "Then you will be shown one proposed answer. Judge strictly from the document and the rubric whether that proposed answer is the correct one. "
    "Reply with yes or no only."
)
TYPE_NAMES = {"choice": "Allowed answers", "score": "Levels, in increasing order", "noul": "Allowed answers"}
# Noul candidates are keyed false/true internally; the model sees them as the answers no/yes. Verifying "the correct answer is no: ..." read as a double negation and lost 2 of 7 own dev items (prompt-1); this framing is prompt-2.
NOUL_NAMES = {"false": "no", "true": "yes"}
YES_FORMS = ("yes", "Yes", " yes", " Yes", "YES")
NO_FORMS = ("no", "No", " no", " No", "NO")


def rubric_text(q: CompiledQuestion) -> str:
    lines = [f"Question: {q.instructions}", f"{TYPE_NAMES[q.type]}:"]
    if q.type == "score":
        lines += [f"  level {c.key}: {c.text}" for c in q.candidates]
    elif q.type == "noul":
        lines += [f"  {NOUL_NAMES[c.key]}: {c.text}" for c in q.candidates]
    else:
        lines += [f"  {c.key}: {c.text}" for c in q.candidates]
    return "\n".join(lines)


def proposition(q: CompiledQuestion, index: int) -> str:
    c = q.candidates[index]
    if q.type == "score":
        head = f"the correct level is {c.key}"
    elif q.type == "noul":
        return f"Proposed answer: {NOUL_NAMES[c.key]} — {c.text}\nIs this proposed answer correct under the rubric?"
    else:
        head = f"the correct answer is {c.key}"
    return f"Proposed answer: {head}: {c.text}\nIs this proposed answer correct under the rubric?"


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BackboneScorer:
    def __init__(self, model_name: str = "Qwen/Qwen3.5-0.8B", revision: str | None = None, device: str | None = None,
                 dtype: torch.dtype = torch.bfloat16, head_path: str | None = None, model_id: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, dtype=dtype).to(self.device).eval()
        self.head = torch.load(head_path, map_location=self.device) if head_path else None
        self.model_id = model_id or f"compass-{'head' if self.head else 'vocab'}-{model_name.split('/')[-1].lower()}-{PROMPT_VERSION}"
        self.backbone = {"model": model_name, "revision": revision, "readout": "head" if self.head else "vocab"}
        self.prompt_version = PROMPT_VERSION
        self.yes_ids = self._form_ids(YES_FORMS)
        self.no_ids = self._form_ids(NO_FORMS)

    def _form_ids(self, forms) -> list[int]:
        ids = {i[0] for f in forms if len(i := self.tok.encode(f, add_special_tokens=False)) == 1}
        if not ids:
            raise RuntimeError("tokenizer has no single-token form for the readout words")
        return sorted(ids)

    # ---- prompt assembly -------------------------------------------------

    def render(self, request: CompiledRequest) -> tuple[str, list[tuple[str, list[str]]]]:
        """Return the state prefix text and, per question, (rubric text, branch texts). The chat template is applied once, so its markup lands in the right segments."""
        # Render one full conversation with markers, then cut it: the template's end-of-user and assistant markup ends up in the branch.
        user = f"<document>\n{request.state}\n</document>\n\n{SPLIT}{SPLIT}"
        rendered = self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prefix, _, tail = rendered.partition(SPLIT + SPLIT)
        questions = []
        for q in request.questions:
            rubric = rubric_text(q) + "\n\n"
            branches = [proposition(q, i) + tail for i in range(len(q.candidates))]
            questions.append((rubric, branches))
        return prefix, questions

    # ---- cache handling --------------------------------------------------

    def _fork(self, cache, n: int):
        forked = copy.deepcopy(cache)
        forked.reorder_cache(torch.zeros(n, dtype=torch.long, device=self.device))
        return forked

    def _run(self, texts: list[str], cache, past_len: int, want_hidden: bool):
        enc = self.tok(texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False).to(self.device)
        n = len(texts)
        mask = torch.cat([torch.ones(n, past_len, dtype=enc.attention_mask.dtype, device=self.device), enc.attention_mask], dim=1)
        out = self.model(input_ids=enc.input_ids, attention_mask=mask, past_key_values=cache, use_cache=True, output_hidden_states=want_hidden)
        lengths = enc.attention_mask.sum(dim=1)
        return out, lengths

    # ---- scoring ---------------------------------------------------------

    def _logodds(self, logits: torch.Tensor) -> torch.Tensor:
        lp = torch.log_softmax(logits.float(), dim=-1)
        return torch.logsumexp(lp[:, self.yes_ids], dim=-1) - torch.logsumexp(lp[:, self.no_ids], dim=-1)

    @torch.no_grad()
    def score(self, request: CompiledRequest) -> ScoreOutput:
        prefix, questions = self.render(request)
        state_out, state_len = self._run([prefix], None, 0, False)
        state_cache = state_out.past_key_values
        tokens = int(state_len.item())
        logits: dict[str, list[float]] = {}
        for q, (rubric, branches) in zip(request.questions, questions):
            q_out, q_len = self._run([rubric], self._fork(state_cache, 1), tokens, False)
            q_tokens = int(q_len.item())
            n = len(branches)
            b_out, b_len = self._run(branches, self._fork(q_out.past_key_values, n), tokens + q_tokens, self.head is not None)
            last = b_len - 1
            rows = torch.arange(n, device=self.device)
            if self.head is None:
                raw = self._logodds(b_out.logits[rows, last])
            else:
                raw = self.head(b_out.hidden_states[-1][rows, last].float()).squeeze(-1)
            values = raw.tolist()
            if not all(math.isfinite(v) for v in values):
                raise RuntimeError(f"non-finite verification logit for question {q.qid!r}")
            logits[q.qid] = values
            tokens += q_tokens + int(b_len.sum().item())
        return ScoreOutput(logits=logits, input_tokens=tokens)
