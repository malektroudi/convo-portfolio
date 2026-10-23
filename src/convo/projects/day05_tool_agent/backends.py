"""Model backends. The loop does not care which one it is talking to.

`Scripted` makes the agent testable: canned responses mean the retry logic,
the parser and the limits can be driven through every branch deterministically
and in milliseconds, without a model and without a network.

`LocalTransformers` is the real thing — a small instruct model on CPU. Small
models are bad at tool calling, and that is the point: the validation and
retry layer is what makes them usable at all, and day 5 measures how much work
it is doing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from convo.core import get_logger
from convo.projects.day05_tool_agent.agent import ModelCall

log = get_logger("day05.backend")

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class Scripted:
    """Returns canned generations in order. Deterministic and instant."""

    responses: list[str]
    name: str = "scripted"
    fallback: str = "Sorry, could you repeat that?"
    calls: list[list[dict[str, Any]]] = field(default_factory=list)

    def generate(
        self, messages: list[dict[str, Any]], tool_specs: list[dict[str, Any]]
    ) -> ModelCall:
        self.calls.append(list(messages))
        index = len(self.calls) - 1
        text = self.responses[index] if index < len(self.responses) else self.fallback
        return ModelCall(text=text, tokens_in=len(messages), tokens_out=1, latency_ms=0.0)

    @property
    def exhausted(self) -> bool:
        return len(self.calls) >= len(self.responses)


@dataclass
class LocalTransformers:
    """A small instruct model on CPU, greedy so a run is reproducible."""

    model_id: str = DEFAULT_MODEL
    max_new_tokens: int = 160
    name: str = ""
    _model: Any = None
    _tokenizer: Any = None

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.model_id

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._model is not None:
            return
        started = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.float32)
        self._model.eval()
        log.info(
            "model loaded",
            extra={"model": self.model_id, "seconds": round(time.perf_counter() - started, 1)},
        )

    def generate(
        self, messages: list[dict[str, Any]], tool_specs: list[dict[str, Any]]
    ) -> ModelCall:
        import torch

        self.load()
        prompt = self._tokenizer.apply_chat_template(
            messages, tools=tool_specs, add_generation_prompt=True, tokenize=False
        )
        encoded = self._tokenizer(prompt, return_tensors="pt")
        started = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # greedy: the comparison must be reproducible
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = out[0][encoded["input_ids"].shape[1] :]
        return ModelCall(
            text=self._tokenizer.decode(generated, skip_special_tokens=True),
            tokens_in=int(encoded["input_ids"].shape[1]),
            tokens_out=int(len(generated)),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
