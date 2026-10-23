"""The held-out-intent experiment: fine-tuned classifier vs embedding router.

Ten of CLINC150's intents are withheld from the classifier entirely, so it is
trained on 140 classes and its output layer has no slot for the other ten. The
router indexes the same 140 intents in full plus `--n-shot` examples each of
the ten new ones. Both are then scored on the untouched test set.

This is the honest form of the comparison. Asking a 150-class model about a
151st intent is not a hard question for it, it is an impossible one, and the
only way to show that is to make the new intents genuinely absent from
training rather than simulating it.

Both approaches emit scores over the SAME canonical 150-class space, so
`convo.core.classification` scores them with identical code. The classifier's
140 softmax columns are scattered into their canonical positions and the ten
held-out columns are left at zero — which is exactly the truth about what it
can predict.

    uv sync --extra train
    uv run python -m convo.projects.day03_embed_router.benchmark

Reuses the day-2 checkpoint for nothing: it trains its own 140-class model,
because day 2's saw all 150 intents.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from convo.core import get_logger, setup
from convo.projects.day02_intent_nlu.data import DEFAULT_CONFIG, Corpus, Split, load_corpus
from convo.projects.day02_intent_nlu.train import DEFAULT_MODEL, infer_logits, train
from convo.projects.day03_embed_router.embed import BASE_MODEL, embed, load_encoder
from convo.projects.day03_embed_router.router import EmbeddingRouter

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
log = get_logger("day03.benchmark")


def choose_held_out(n_classes: int, n_held_out: int, seed: int) -> list[int]:
    """A deterministic sample of class indices to withhold."""
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n_classes, size=n_held_out, replace=False))


def restrict_train(train_split: Split, known: list[int]) -> tuple[Split, dict[int, int]]:
    """Training data for the known intents only, relabelled 0..len(known)-1."""
    to_local = {canonical: local for local, canonical in enumerate(known)}
    keep = np.array([int(label) in to_local for label in train_split.labels])
    texts = [t for t, k in zip(train_split.texts, keep, strict=True) if k]
    labels = np.array([to_local[int(label)] for label in train_split.labels[keep]])
    return Split(texts=texts, labels=labels), to_local


def few_shot_indices(
    train_split: Split, held_out: list[int], n_shot: int, seed: int
) -> dict[int, list[int]]:
    """`n_shot` training-row indices per held-out intent, chosen deterministically."""
    rng = np.random.default_rng(seed + 1)
    picked: dict[int, list[int]] = {}
    for canonical in held_out:
        rows = np.flatnonzero(train_split.labels == canonical)
        take = min(n_shot, len(rows))
        picked[canonical] = sorted(int(i) for i in rng.choice(rows, size=take, replace=False))
    return picked


def expand_to_canonical(local_scores: np.ndarray, known: list[int], n_canonical: int) -> np.ndarray:
    """Scatter a 140-class score matrix into the 150-class space, zeros elsewhere.

    The zeros are the finding, not a padding convenience: the classifier assigns
    exactly zero probability to an intent it was never given a slot for.
    """
    out = np.zeros((len(local_scores), n_canonical), dtype=np.float32)
    out[:, known] = local_scores
    return out


def time_per_query(fn, samples: int = 40) -> float:
    """Median single-query wall time in milliseconds."""
    timings = []
    for _ in range(samples):
        started = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - started) * 1000)
    return round(float(np.median(timings)), 2)


def main(argv: list[str] | None = None) -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--held-out", type=int, default=10, help="intents withheld")
    parser.add_argument("--n-shot", type=int, default=5, help="examples per new intent")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=24)
    parser.add_argument("--strategy", default="knn", choices=["knn", "centroid"])
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args(argv)

    corpus = load_corpus(args.config)
    n_canonical = corpus.num_classes
    held_out = choose_held_out(n_canonical, args.held_out, args.seed)
    known = [i for i in range(n_canonical) if i not in set(held_out)]
    log.info(
        "experiment",
        extra={
            "canonical_classes": n_canonical,
            "known": len(known),
            "held_out": [corpus.class_names[i] for i in held_out],
            "n_shot": args.n_shot,
        },
    )

    # ---- the router: base encoder, no training at all -------------------
    model, tokenizer = load_encoder(BASE_MODEL)
    train_vectors = embed(corpus.train.texts, model, tokenizer, args.max_len)
    test_vectors = embed(corpus.test.texts, model, tokenizer, args.max_len)

    shots = few_shot_indices(corpus.train, held_out, args.n_shot, args.seed)
    index_rows = [i for i, label in enumerate(corpus.train.labels) if int(label) in set(known)]
    index_rows += [i for rows in shots.values() for i in rows]
    index_rows.sort()

    started = time.perf_counter()
    router = EmbeddingRouter.fit(
        train_vectors[index_rows],
        corpus.train.labels[index_rows],
        corpus.class_names,
        strategy=args.strategy,
        k=args.k,
    )
    router_build_s = round(time.perf_counter() - started, 3)

    # The cold-start operation, timed: one more intent from n_shot examples.
    spare = corpus.class_names[held_out[0]]
    probe = EmbeddingRouter.fit(
        train_vectors[index_rows],
        corpus.train.labels[index_rows],
        corpus.class_names,
        strategy=args.strategy,
        k=args.k,
    )
    started = time.perf_counter()
    probe.add_intent(f"{spare}__probe", train_vectors[shots[held_out[0]]])
    router_add_intent_s = round(time.perf_counter() - started, 4)

    router_scores = router.scores(test_vectors)
    one_vector = test_vectors[:1]
    router_latency_ms = time_per_query(lambda: router.scores(one_vector))
    embed_latency_ms = time_per_query(
        lambda: embed([corpus.test.texts[0]], model, tokenizer, args.max_len)
    )

    # ---- the classifier: fine-tuned on the 140 known intents only -------
    known_train, _ = restrict_train(corpus.train, known)
    known_corpus = Corpus(
        train=known_train,
        validation=restrict_train(corpus.validation, known)[0],
        test=corpus.test,
        class_names=[corpus.class_names[i] for i in known],
    )
    log.info("training the 140-class classifier", extra={"examples": len(known_train)})
    started = time.perf_counter()
    clf, clf_tokenizer = train(
        known_corpus,
        model_name=DEFAULT_MODEL,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_len=args.max_len,
        seed=args.seed,
    )
    classifier_train_s = round(time.perf_counter() - started, 1)

    local_logits = infer_logits(clf, clf_tokenizer, corpus.test.texts, args.max_len, 128)
    from convo.core.classification import softmax

    classifier_scores = expand_to_canonical(softmax(local_logits), known, n_canonical)
    clf_latency_ms = time_per_query(
        lambda: infer_logits(clf, clf_tokenizer, corpus.test.texts[:1], args.max_len, 1)
    )

    # ---- save -----------------------------------------------------------
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ARTIFACT_DIR / "comparison.npz",
        router_scores=router_scores.astype(np.float32),
        classifier_scores=classifier_scores.astype(np.float32),
        labels=corpus.test.labels.astype(np.int64),
        held_out=np.array(held_out, dtype=np.int64),
    )
    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": f"clinc/clinc_oos:{args.config}",
                "canonical_classes": n_canonical,
                "known_classes": len(known),
                "held_out_names": [corpus.class_names[i] for i in held_out],
                "n_shot": args.n_shot,
                "seed": args.seed,
                "router": {
                    "encoder": BASE_MODEL,
                    "fine_tuned": False,
                    "strategy": args.strategy,
                    "k": args.k,
                    "exemplars": int(router.n_exemplars),
                    "build_s": router_build_s,
                    "add_intent_s": router_add_intent_s,
                    "embed_latency_ms": embed_latency_ms,
                    "score_latency_ms": router_latency_ms,
                },
                "classifier": {
                    "model": DEFAULT_MODEL,
                    "fine_tuned": True,
                    "epochs": args.epochs,
                    "learning_rate": args.lr,
                    "train_examples": len(known_train),
                    "train_s": classifier_train_s,
                    "latency_ms": clf_latency_ms,
                },
                "class_names": corpus.class_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(
        "comparison written",
        extra={
            "path": str(ARTIFACT_DIR / "comparison.npz"),
            "router_build_s": router_build_s,
            "router_add_intent_s": router_add_intent_s,
            "classifier_train_s": classifier_train_s,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
