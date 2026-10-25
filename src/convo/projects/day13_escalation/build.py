"""Replay every conversation through the real stack, then score the policies.

    uv sync --extra train
    uv run python -m convo.projects.day13_escalation.build

The machine state the signals read is **measured, not asserted**. Each `faq`
conversation goes through days 8 to 10 for real — hybrid retrieval, the
cross-encoder, the evidence gate and the citation gate — so a refusal in the
record is a refusal day 10's thresholds actually produced. Each `booking`
conversation goes through day 4's engine, so a reprompt count is one its
extractors actually reached.

That distinction is the whole reason this build exists. The labelled set says
*when a person should have taken over*; it says nothing about what the system
did, because a set that hand-writes both sides is grading the labels against
themselves. Writing "the bot refused here" next to "escalate here" would make
the refusal signal look perfect and prove nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from convo.core import get_logger, setup
from convo.core.retrieval import reciprocal_rank_fusion
from convo.projects.day03_embed_router.embed import BASE_MODEL, embed, load_encoder
from convo.projects.day08_rag_cites.answer import cite_in_document
from convo.projects.day08_rag_cites.chunking import chunk_corpus, load_corpus
from convo.projects.day08_rag_cites.index import BM25, DenseIndex
from convo.projects.day09_retrieval_evals.build import DEPTH, load_reranker, rerank
from convo.projects.day10_guardrails.groundedness import Evidence, decide
from convo.projects.day13_escalation import policy as policies
from convo.projects.day13_escalation.conversations import CONVERSATIONS, Conversation, Exchange
from convo.projects.day13_escalation.packet import (
    build_packet,
    minimal_packet,
    score_packet,
    transcript_only,
)
from convo.projects.day13_escalation.signals import ALL_SIGNALS, read

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
STRATEGY = "structural"

log = get_logger("day13.build")


class Stack:
    """Days 8 to 10, loaded once and asked one question at a time."""

    def __init__(self) -> None:
        self.documents = load_corpus()
        self.by_id = {d.doc_id: d for d in self.documents}
        self.chunks = chunk_corpus(self.documents, STRATEGY)
        self.texts = {c.chunk_id: c.text for c in self.chunks}
        encoder, tokenizer = load_encoder(BASE_MODEL)
        self.encoder, self.tokenizer = encoder, tokenizer
        vectors = embed([c.text for c in self.chunks], encoder, tokenizer, max_len=256)
        self.bm25 = BM25(self.chunks)
        self.dense = DenseIndex(self.chunks, vectors)
        self.reranker, self.reranker_tok = load_reranker()
        self.by_chunk = {c.chunk_id: c for c in self.chunks}

        from convo.projects.day10_guardrails.results import load as load_guard

        guard = load_guard()
        if guard is None:
            raise SystemExit(
                "day 10 has no evidence artifact; run its build first so the "
                "threshold here is the one day 10 actually tuned"
            )
        self.threshold = guard.best().threshold

    def ask(self, question: str) -> Exchange:
        """Retrieve, rerank, gate — exactly as day 10 does."""
        lexical = [h.chunk.chunk_id for h in self.bm25.search(question, k=DEPTH)]
        vector = embed([question], self.encoder, self.tokenizer, max_len=64)[0]
        dense_ids = [h.chunk.chunk_id for h in self.dense.search(vector, k=DEPTH)]
        fused = reciprocal_rank_fusion({"bm25": lexical, "dense": dense_ids})[:DEPTH]
        ranked, _ = rerank(self.reranker, self.reranker_tok, question, fused, self.texts)

        if not ranked:
            return Exchange(user=question, reply="", answered=False)

        top = self.by_chunk[ranked[0]]
        scores = _rerank_scores(self, question, ranked[:1])
        citation = cite_in_document(top, question, self.by_id[top.doc_id])
        evidence = Evidence(
            question=question,
            doc_id=top.doc_id,
            chunk_id=top.chunk_id,
            score=scores[0],
            citation=citation,
        )
        decision = decide(evidence, self.threshold)
        return Exchange(
            user=question,
            reply=decision.text,
            answered=decision.answered,
            doc_id=top.doc_id if decision.answered else "",
            evidence_score=scores[0],
        )


def _rerank_scores(stack: Stack, question: str, chunk_ids: list[str]) -> list[float]:
    """The cross-encoder's raw relevance for specific chunks.

    `rerank` returns an order, not the scores, and day 10's evidence gate
    compares against a score — so this re-runs the model on the shortlist
    rather than inventing a stand-in for it.
    """
    import torch

    encoded = stack.reranker_tok(
        [question] * len(chunk_ids),
        [stack.texts[c] for c in chunk_ids],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = stack.reranker(**encoded).logits.squeeze(-1)
    return [float(x) for x in logits.reshape(-1)]


def run_booking(conversation: Conversation) -> list[Exchange]:
    """Replay a booking through day 4's engine, keeping its own counters."""
    from convo.projects.day04_dialog_engine.engine import Session, turn

    session = Session()
    out: list[Exchange] = []
    for text in conversation.user_turns:
        result = turn(session, text)
        out.append(
            Exchange(
                user=text,
                reply=result.reply,
                answered=True,
                reprompts=result.reprompts,
                slots=dict(result.slots_after),
            )
        )
    return out


def run_faq(conversation: Conversation, stack: Stack) -> list[Exchange]:
    return [stack.ask(text) for text in conversation.user_turns]


def main() -> int:
    setup()
    stack = Stack()

    replayed: list[Conversation] = []
    for conversation in CONVERSATIONS:
        exchanges = (
            run_booking(conversation)
            if conversation.kind == "booking"
            else run_faq(conversation, stack)
        )
        replayed.append(
            Conversation(
                name=conversation.name,
                kind=conversation.kind,
                exchanges=exchanges,
                escalate_at=conversation.escalate_at,
                note=conversation.note,
                needs=conversation.needs,
            )
        )
        log.info(
            "conversation replayed",
            extra={
                # Not "name": logging reserves it on LogRecord and raises.
                "conversation": conversation.name,
                "kind": conversation.kind,
                "refusals": sum(1 for e in exchanges if not e.answered),
                "max_reprompts": max((e.reprompts for e in exchanges), default=0),
            },
        )

    named_policies = {
        "never": policies.never,
        "always": policies.always,
        "explicit_only": policies.explicit_only,
        "signals": policies.signals_policy(),
        # The only tunable, swept rather than chosen. `any one` lets a single
        # weak signal escalate; `three agree` demands more corroboration.
        "signals (any one)": policies.signals_policy(soft_limit=1),
        "signals (three agree)": policies.signals_policy(soft_limit=3),
    }
    reports = {
        name: policies.score(name, policy, replayed).as_dict()
        for name, policy in named_policies.items()
    }

    # One signal removed at a time, so a signal that earns nothing says so.
    ablation = {
        f"without {name}": policies.score(
            f"without {name}", policies.without(name), replayed
        ).as_dict()
        for name in ALL_SIGNALS
    }

    # A counterfactual, not a fix. `asks-for-human-mid-booking` is missed
    # because day 4's HUMAN_PATTERNS has no entry for "put me through to
    # someone" — an entirely ordinary way to ask. Rerunning with that one
    # phrase added measures what the gap costs, without quietly editing the
    # list after seeing which test it failed.
    from convo.projects.day04_dialog_engine.flow import HUMAN_PATTERNS
    from convo.projects.day13_escalation import signals as signals_module

    patched = HUMAN_PATTERNS + ("put me through",)
    original = signals_module.HUMAN_PATTERNS
    signals_module.HUMAN_PATTERNS = patched
    try:
        counterfactual = policies.score(
            "signals + one phrase", policies.signals_policy(), replayed
        ).as_dict()
    finally:
        signals_module.HUMAN_PATTERNS = original

    # What the handoff carries, under three ways of writing it.
    packets: list[dict] = []
    full_policy = policies.signals_policy()
    for conversation in replayed:
        handoff = full_policy(conversation.exchanges)
        if handoff is None:
            continue
        for style, builder in (
            ("full", build_packet),
            ("ticket", lambda c, h: minimal_packet(c, h)),
            ("transcript", transcript_only),
        ):
            packet = builder(conversation, handoff)
            packets.append(
                {
                    "style": style,
                    "packet": packet.as_dict(),
                    "score": score_packet(conversation, packet).as_dict(),
                }
            )

    # Every signal's state at every turn, so the tab can show the case.
    readings = []
    for conversation in replayed:
        for index in range(len(conversation.exchanges)):
            reading = read(conversation.exchanges, index)
            readings.append(
                {
                    "conversation": conversation.name,
                    "turn": index,
                    "signals": [s.as_dict() for s in reading.signals],
                }
            )

    payload = {
        "threshold": stack.threshold,
        "strategy": STRATEGY,
        "conversations": [
            {
                "name": c.name,
                "kind": c.kind,
                "note": c.note,
                "escalate_at": c.escalate_at,
                "exchanges": [e.as_dict() for e in c.exchanges],
                "needs": [
                    {"key": n.key, "must_contain": n.must_contain, "why": n.why} for n in c.needs
                ],
            }
            for c in replayed
        ],
        "policies": reports,
        "ablation": ablation,
        "counterfactual": counterfactual,
        "packets": packets,
        "readings": readings,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT_DIR / "escalation.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log.info(
        "artifact written",
        extra={
            "path": str(out),
            "kilobytes": out.stat().st_size // 1024,
            "conversations": len(replayed),
            "packets": len(packets),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
