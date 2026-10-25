"""Two ways to produce the reply, with very different latency.

`cited` is the stack days 8 to 11 built: retrieve, check the evidence clears
the bar, and speak the verified span. There is no generation step, so there is
no time-to-first-token — the whole sentence exists at once, roughly as fast as
BM25 can score 49 chunks.

`generated` is a small instruct model streaming tokens, which is what most
people mean by a voice assistant. It is included because leaving it out would
make the latency result look like a free win rather than a trade: the cited
path is fast *and* cannot answer anything the corpus does not already say in
one sentence.

Both record two different moments, and the difference between them is the
point of the day:

  first token     the metric everyone quotes
  first sentence  the metric that matters, because a token cannot be spoken

For the cited path they are the same instant. For the generated one they are
seconds apart.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from convo.core import get_logger

LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

SENTENCE_END = re.compile(r"[.!?](?:\s|$)")

SYSTEM = (
    "You answer a restaurant's phone. Use only the context. "
    "Answer in two or three short sentences, as speech, with no lists and no "
    "markdown: state the answer, then offer the obvious next step."
)
"""Two or three sentences, not one.

A one-sentence reply makes the streaming-TTS comparison meaningless — the
first sentence *is* the whole reply, so speaking early and waiting for the end
are the same operation on the same string. Real spoken answers confirm and
then offer the next step, and that shape is the only one where starting to
speak before the model has finished can buy anything."""

log = get_logger("day12.think")


@dataclass(frozen=True)
class Reply:
    """A reply, with when each part of it became available."""

    text: str
    first_token_ms: float
    first_sentence_ms: float
    """When a *complete* sentence first existed — what TTS can start on."""
    complete_ms: float
    path: str
    first_sentence: str

    @property
    def token_to_sentence_ms(self) -> float:
        """The gap voice pays for and chat does not."""
        return round(self.first_sentence_ms - self.first_token_ms, 2)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "path": self.path,
            "first_token_ms": round(self.first_token_ms, 2),
            "first_sentence_ms": round(self.first_sentence_ms, 2),
            "complete_ms": round(self.complete_ms, 2),
            "first_sentence": self.first_sentence,
        }


def split_first_sentence(text: str) -> tuple[str, str]:
    """(first sentence, rest). The whole text when it has no terminator yet."""
    match = SENTENCE_END.search(text)
    if not match:
        return text, ""
    cut = match.end()
    return text[:cut].strip(), text[cut:].strip()


def cited_reply(question: str) -> Reply:
    """Day 10's guarded reply, spoken. No model, so no tokens to wait for."""
    from convo.projects.day10_guardrails.results import load as load_guard

    started = time.perf_counter()
    results = load_guard()
    text = ""
    if results is not None:
        best = results.best()
        for evidence, _ in results.answerable:
            if evidence.question == question:
                from convo.projects.day10_guardrails.groundedness import decide

                decision = decide(evidence, best.threshold)
                text = decision.text
                break
    if not text:
        from convo.projects.day10_guardrails.groundedness import REFUSAL_TEXT

        text = REFUSAL_TEXT

    elapsed = (time.perf_counter() - started) * 1000
    first, _ = split_first_sentence(text)
    return Reply(
        text=text,
        first_token_ms=elapsed,
        first_sentence_ms=elapsed,
        complete_ms=elapsed,
        path="cited",
        first_sentence=first,
    )


@dataclass
class Generator:
    """A small instruct model, streamed token by token on CPU."""

    model_id: str = LLM_MODEL
    max_new_tokens: int = 72
    _model: Any = None
    _tokenizer: Any = None
    _threads: int = field(default=4)

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(self._threads)
        started = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch.float32
        ).eval()
        log.info(
            "llm loaded",
            extra={"model": self.model_id, "seconds": round(time.perf_counter() - started, 1)},
        )

    def reply(self, question: str, context: str) -> Reply:
        import threading

        import torch
        from transformers import TextIteratorStreamer

        self.load()
        prompt = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}"},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        encoded = self._tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        started = time.perf_counter()

        def _run() -> None:
            with torch.no_grad():
                self._model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # greedy: a latency run must be repeatable
                    streamer=streamer,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

        worker = threading.Thread(target=_run)
        worker.start()

        text = ""
        first_token_ms = 0.0
        first_sentence_ms = 0.0
        first_sentence = ""
        for piece in streamer:
            if not piece:
                continue
            if first_token_ms == 0.0 and piece.strip():
                first_token_ms = (time.perf_counter() - started) * 1000
            text += piece
            if not first_sentence:
                candidate, _ = split_first_sentence(text)
                if SENTENCE_END.search(text):
                    first_sentence = candidate
                    first_sentence_ms = (time.perf_counter() - started) * 1000
        worker.join()

        complete_ms = (time.perf_counter() - started) * 1000
        if not first_sentence:
            # The model never terminated a sentence inside its token budget,
            # so the whole output is what TTS would have had to wait for.
            first_sentence = text.strip()
            first_sentence_ms = complete_ms
        return Reply(
            text=text.strip(),
            first_token_ms=first_token_ms or complete_ms,
            first_sentence_ms=first_sentence_ms,
            complete_ms=complete_ms,
            path="generated",
            first_sentence=first_sentence,
        )
