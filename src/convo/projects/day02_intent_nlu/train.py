"""Fine-tune a small encoder on CLINC150's 150 in-scope intents.

Runs on CPU in minutes, or on a free Kaggle T4 in well under one. Writes two
things:

  artifacts/day02/test_logits.npz   test-set logits + labels (committed, ~3 MB)
  models/day02/                     the fine-tuned weights (gitignored, ~90 MB)

The hub reads only the artifact, which is why it needs no torch and still fits
Cloud Run's free tier. Usage:

    uv sync --extra train
    uv run python -m convo.projects.day02_intent_nlu.train
    uv run python -m convo.projects.day02_intent_nlu.train --epochs 1 --limit 600
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from convo.core import get_logger, setup
from convo.projects.day02_intent_nlu.data import DEFAULT_CONFIG, Corpus, load_corpus

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Inside the package, so the artifact installs with the wheel and the hub can
# read it on Cloud Run without a separate COPY or a torch dependency.
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_DIR = Path("models/day02")

log = get_logger("day02.train")


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _encode(tokenizer, texts: list[str], max_len: int):
    batch = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return batch["input_ids"], batch["attention_mask"]


def infer_logits(model, tokenizer, texts: list[str], max_len: int, batch_size: int) -> np.ndarray:
    """Run the model over texts and return raw logits as float32."""
    import torch

    model.eval()
    device = next(model.parameters()).device
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            ids, mask = _encode(tokenizer, texts[start : start + batch_size], max_len)
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device))
            chunks.append(out.logits.detach().cpu().float().numpy())
    return np.concatenate(chunks, axis=0)


def train(
    corpus: Corpus,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_len: int,
    seed: int,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device selected", extra={"device": str(device), "model": model_name})

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=corpus.num_classes,
        id2label=dict(enumerate(corpus.class_names)),
        label2id={name: i for i, name in enumerate(corpus.class_names)},
    ).to(device)

    ids, mask = _encode(tokenizer, corpus.train.texts, max_len)
    labels = torch.tensor(corpus.train.labels, dtype=torch.long)
    loader = DataLoader(TensorDataset(ids, mask, labels), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_scheduler(
        "linear", optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    log.info("training", extra={"steps": total_steps, "examples": len(corpus.train)})
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for step, (batch_ids, batch_mask, batch_labels) in enumerate(loader, start=1):
            out = model(
                input_ids=batch_ids.to(device),
                attention_mask=batch_mask.to(device),
                labels=batch_labels.to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running += out.loss.detach().item() * len(batch_labels)
            seen += len(batch_labels)
            if step % 50 == 0:
                log.info(
                    "step",
                    extra={"epoch": epoch, "step": step, "loss": round(running / seen, 4)},
                )

        val_logits = infer_logits(model, tokenizer, corpus.validation.texts, max_len, 128)
        val_acc = float((val_logits.argmax(1) == corpus.validation.labels).mean())
        log.info(
            "epoch done",
            extra={
                "epoch": epoch,
                "train_loss": round(running / max(seen, 1), 4),
                "val_accuracy": round(val_acc, 4),
                "elapsed_s": round(time.perf_counter() - started, 1),
            },
        )

    return model, tokenizer


def main(argv: list[str] | None = None) -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--config", default=DEFAULT_CONFIG, choices=["small", "imbalanced", "plus"])
    # Defaults chosen after a 3-epoch / 3e-5 run underfit badly: 74% in-scope
    # accuracy and a max softmax confidence of 0.09 across the whole test set,
    # i.e. logits that never sharpened. A 150-way head on a 6-layer encoder
    # needs a higher rate and more passes than a BERT-base recipe would.
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    # p99 of CLINC150 is 21 tokens, max 33; 24 truncates 0.2% and is 25% faster.
    parser.add_argument("--max-len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--limit", type=int, default=0, help="cap training examples, for a smoke run"
    )
    parser.add_argument("--save-model", action="store_true", help="also write the weights")
    parser.add_argument(
        "--from-model",
        default="",
        metavar="DIR",
        help="skip training and re-export the artifact from weights saved earlier",
    )
    args = parser.parse_args(argv)

    corpus = load_corpus(args.config)
    if args.limit:
        from convo.projects.day02_intent_nlu.data import Split

        corpus = Corpus(
            train=Split(corpus.train.texts[: args.limit], corpus.train.labels[: args.limit]),
            validation=corpus.validation,
            test=corpus.test,
            class_names=corpus.class_names,
        )
    log.info(
        "corpus loaded",
        extra={
            "classes": corpus.num_classes,
            "train": len(corpus.train),
            "validation": len(corpus.validation),
            "test_in_scope": corpus.test.in_scope_count,
            "test_oos": corpus.test.oos_count,
        },
    )

    if args.from_model:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        log.info("re-exporting from saved weights", extra={"path": args.from_model})
        tokenizer = AutoTokenizer.from_pretrained(args.from_model)
        model = AutoModelForSequenceClassification.from_pretrained(args.from_model)
        if model.config.num_labels != corpus.num_classes:
            raise SystemExit(
                f"saved model has {model.config.num_labels} labels, corpus has {corpus.num_classes}"
            )
    else:
        model, tokenizer = train(
            corpus,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_len=args.max_len,
            seed=args.seed,
        )

    test_logits = infer_logits(model, tokenizer, corpus.test.texts, args.max_len, 128)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ARTIFACT_DIR / "test_logits.npz",
        logits=test_logits.astype(np.float32),
        labels=corpus.test.labels.astype(np.int64),
    )
    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "dataset": f"clinc/clinc_oos:{args.config}",
                "trained_on": "150 in-scope intents only; out-of-scope is rejected at inference",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "max_len": args.max_len,
                "seed": args.seed,
                "exported_from": args.from_model or "this training run",
                "train_examples": len(corpus.train),
                "test_in_scope": corpus.test.in_scope_count,
                "test_oos": corpus.test.oos_count,
                "class_names": corpus.class_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("artifact written", extra={"path": str(ARTIFACT_DIR / "test_logits.npz")})

    if args.save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        log.info("weights written", extra={"path": str(MODEL_DIR)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
