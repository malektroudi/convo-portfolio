"""CLINC150 loading. Needs the `train` extra; the hub never imports this.

The dataset ships 150 in-scope intents plus an `oos` class. We train on the
150 ONLY and leave out-of-scope entirely to inference-time rejection, because
a client's real out-of-scope traffic never looks like a curated OOS training
split. See analysis.py for why that shapes everything downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convo.projects.day02_intent_nlu.analysis import OOS

DATASET = "clinc/clinc_oos"
DEFAULT_CONFIG = "plus"
OOS_INTENT_NAME = "oos"


@dataclass(frozen=True)
class Split:
    texts: list[str]
    labels: np.ndarray  # model-space class index, or OOS (-1)

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def in_scope_count(self) -> int:
        return int((self.labels >= 0).sum())

    @property
    def oos_count(self) -> int:
        return int((self.labels == OOS).sum())


@dataclass(frozen=True)
class Corpus:
    train: Split
    validation: Split
    test: Split
    class_names: list[str]  # 150 in-scope names, in model-class order

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def load_corpus(config: str = DEFAULT_CONFIG, drop_oos_from_train: bool = True) -> Corpus:
    """Load CLINC150 and remap its 151 labels into 150 model classes plus OOS.

    The dataset's own label ids are not contiguous once `oos` is removed, so we
    build an explicit mapping rather than assuming anything about ordering.
    """
    from datasets import load_dataset  # imported lazily: `train` extra only

    raw = load_dataset(DATASET, config)
    dataset_names: list[str] = raw["train"].features["intent"].names
    oos_id = dataset_names.index(OOS_INTENT_NAME)

    in_scope_ids = [i for i in range(len(dataset_names)) if i != oos_id]
    class_names = [dataset_names[i] for i in in_scope_ids]
    to_model_class = {dataset_id: slot for slot, dataset_id in enumerate(in_scope_ids)}

    def convert(split_name: str, drop_oos: bool) -> Split:
        texts: list[str] = []
        labels: list[int] = []
        for row in raw[split_name]:
            intent = int(row["intent"])
            if intent == oos_id:
                if drop_oos:
                    continue
                texts.append(row["text"])
                labels.append(OOS)
            else:
                texts.append(row["text"])
                labels.append(to_model_class[intent])
        return Split(texts=texts, labels=np.array(labels, dtype=np.int64))

    return Corpus(
        train=convert("train", drop_oos=drop_oos_from_train),
        validation=convert("validation", drop_oos=True),
        test=convert("test", drop_oos=False),
        class_names=class_names,
    )
