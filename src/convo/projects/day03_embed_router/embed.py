"""Sentence embeddings from the BASE encoder — no fine-tuning anywhere.

That is the comparison day 3 is making: an off-the-shelf encoder plus a
nearest-neighbour index, against the same encoder fine-tuned into a classifier.
Using the fine-tuned weights here would quietly smuggle day 2's supervision
into day 3 and make the router look better than it is.

Mean pooling over the attention mask, then L2 normalisation, which is what
all-MiniLM-L6-v2 was trained to produce.
"""

from __future__ import annotations

import time

import numpy as np

from convo.core import get_logger

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
log = get_logger("day03.embed")


def load_encoder(model_name: str = BASE_MODEL):
    """Return (model, tokenizer) for the plain encoder. Needs the `train` extra."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def embed(
    texts: list[str],
    model,
    tokenizer,
    max_len: int = 24,
    batch_size: int = 128,
) -> np.ndarray:
    """Mean-pooled, L2-normalised sentence embeddings, shape (n, dim)."""
    import torch

    device = next(model.parameters()).device
    out: list[np.ndarray] = []
    started = time.perf_counter()

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state

            expanded = mask.unsqueeze(-1).float()
            pooled = (hidden * expanded).sum(1) / expanded.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.cpu().float().numpy())

    log.info(
        "embedded",
        extra={"texts": len(texts), "elapsed_s": round(time.perf_counter() - started, 2)},
    )
    return np.concatenate(out, axis=0).astype(np.float32)
