"""Turn a follow-up into a question that stands on its own, with no model.

Day 9 ended by saying a question sharing no words with its answer needs
rewriting *before* retrieval rather than reordering after. Multi-turn makes
that sharper: "is it free on Sundays" shares no words with anything because it
barely *is* a question — it is a fragment plus a pronoun, and the thing it
refers to was said a turn ago.

The usual move here is to hand the transcript to an LLM and ask for a
standalone question. That works, and it costs a model call on the critical
path of every turn before retrieval has even started. This file asks how far
four rules get instead, and the build measures the answer against a
hand-written oracle so the gap is a number rather than a claim.

The rules, applied in order, each one switchable so the build can ablate them:

  PRONOUN    "is *it* free on Sundays"        -> ... street parking ...
  SUBSTITUTE "what about *the back room*"     -> swaps for "the dining room"
  ELLIPSIS   "what about for a private event" -> grafts the previous question
  CARRYOVER  anything still too thin          -> appends recent content words

Order matters and is not arbitrary: substitution is tried before grafting
because replacing a phrase keeps a question short, while grafting makes it
longer, and a longer query dilutes BM25's term weighting. The cheapest edit
that resolves the reference is the best one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from convo.projects.day08_rag_cites.index import STOPWORDS, WORD
from convo.projects.day11_multiturn_rag.memory import Memory

STANDALONE = "standalone"
PRONOUN = "pronoun"
SUBSTITUTE = "substitute"
ELLIPSIS = "ellipsis"
CARRYOVER = "carryover"

GUARD = "guard"

ALL_RULES: tuple[str, ...] = (PRONOUN, SUBSTITUTE, ELLIPSIS, CARRYOVER)
"""The rewriter as designed, before any of its results were looked at."""

GUARDED_RULES: tuple[str, ...] = (*ALL_RULES, GUARD)
"""ALL_RULES plus a thinness guard, added *after* seeing the failures below.

Its number is therefore optimistic — it was derived from the same twenty turns
it is scored on, so it has seen its own test. It is reported as a separate row
rather than folded into the headline, because a rule tuned on the measurement
it improves is a hypothesis, not a result.
"""

GUARD_MIN = 3
"""Content words above which a turn is left alone even if it has a pronoun.

Resolving a pronoun replaces one stopword with a multi-word article title, and
for a lexical retriever that is not free: "and if it is five days before"
already contains "five" and "days", which find the right passage on their own,
and expanding it to "and if cancelling a private event is five days before"
adds three terms that pull towards the article the conversation *started* in.
A turn with enough content words of its own does not need the help.
"""

SINGULAR_ANAPHORS = frozenset(["it", "that", "this", "one"])
PLURAL_ANAPHORS = frozenset(["they", "them", "those", "these"])
ANAPHORS = SINGULAR_ANAPHORS | PLURAL_ANAPHORS

# "what about X", "how about X", "and X?" — a question with no predicate of its
# own, which has to borrow the previous one.
FRAGMENT = re.compile(
    r"^\s*(?:and\s+|but\s+|so\s+)?(?:what|how)\s+about\s+(?P<rest>.+?)\s*\??$",
    re.IGNORECASE,
)
# "even in the evening", "and on a Sunday" — a modifier offered on its own.
MODIFIER = re.compile(
    r"^\s*(?:and\s+|but\s+)?(?:even\s+|also\s+)?(?P<rest>(?:in|on|at|for|with|during)\s+.+?)\s*\??$",
    re.IGNORECASE,
)

MIN_CONTENT = 2
"""Content words below which a question cannot stand on its own."""


@dataclass(frozen=True)
class Rewrite:
    """What the rewriter did, and why — the `why` is shown in the tab."""

    original: str
    text: str
    strategy: str
    detail: str = ""

    @property
    def changed(self) -> bool:
        return self.text.strip().lower() != self.original.strip().lower()

    def as_dict(self) -> dict:
        return {
            "original": self.original,
            "text": self.text,
            "strategy": self.strategy,
            "detail": self.detail,
        }


def content_words(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOPWORDS]


def anaphors_in(text: str) -> list[str]:
    """Pronouns that need an antecedent, in order.

    Existential "there" ("is there a dress code") is not one, and neither is
    "that" used as a complementiser ("I heard that you close at ten"). Only
    the pronoun readings are anaphors, and distinguishing them without a
    parser means a short list of contexts to exclude.
    """
    words = WORD.findall(text.lower())
    found = []
    for index, word in enumerate(words):
        if word not in ANAPHORS:
            continue
        previous = words[index - 1] if index else ""
        if word == "that" and previous in {"heard", "said", "think", "know", "hope"}:
            continue
        found.append(word)
    return found


def _head(phrase: str) -> str:
    """The last content word, which is the noun a phrase is about."""
    words = content_words(phrase)
    return words[-1] if words else ""


def _resolve_pronouns(text: str, memory: Memory) -> Rewrite | None:
    """Replace each anaphor with the most recent agreeing topic."""
    found = anaphors_in(text)
    if not found:
        return None

    out = text
    swapped: list[str] = []
    for word in found:
        plural = word in PLURAL_ANAPHORS
        topic = memory.antecedent(plural=plural)
        if topic is None:
            continue
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        out, count = pattern.subn(topic.name, out, count=1)
        if count:
            swapped.append(f"{word} -> {topic.name}")

    if not swapped:
        return None
    return Rewrite(text, out, PRONOUN, "; ".join(swapped))


def _substitute(text: str, memory: Memory) -> Rewrite | None:
    """Swap a phrase in the previous question for the one offered now.

    "is the dining room wheelchair accessible" then "what about the back room"
    share the head noun "room", so the new phrase replaces the old one and the
    question stays short. Without this the fragment would be appended, leaving
    both rooms in the query and letting the stale one compete for the match.
    """
    match = FRAGMENT.match(text)
    previous = memory.last_question()
    if not match or not previous:
        return None

    rest = match.group("rest").strip()
    head = _head(rest)
    if not head:
        return None

    # Find the phrase in the previous question ending in the same head noun,
    # and take the whole of it. Replacing only the head leaves its modifiers
    # stranded: "is the dining room accessible" with just "room" swapped gives
    # "is the dining the back room accessible", which keeps the stale word the
    # substitution exists to remove.
    words = previous.split()
    for index, word in enumerate(words):
        if _head(word) != head:
            continue
        start = index
        while start > 0 and _modifies(words[start - 1]):
            start -= 1
        replaced = " ".join(words[:start] + [rest] + words[index + 1 :])
        return Rewrite(
            text, replaced, SUBSTITUTE, f"{' '.join(words[start : index + 1])} -> {rest}"
        )
    return None


DETERMINERS = frozenset(["the", "a", "an", "my", "your", "our", "their", "its", "this", "that"])


def _modifies(word: str) -> bool:
    """Whether `word` can be part of the noun phrase ending to its right.

    Determiners and content words can ("the", "dining"); other stopwords
    cannot, which is what stops the walk at the verb in "is the dining room".
    """
    lowered = word.lower()
    if lowered in DETERMINERS:
        return True
    return bool(content_words(lowered))


def _graft(text: str, memory: Memory) -> Rewrite | None:
    """Append the fragment to the previous question, which supplies the verb."""
    previous = memory.last_question()
    if not previous:
        return None

    match = FRAGMENT.match(text)
    if match:
        rest = match.group("rest").strip()
        return Rewrite(text, f"{previous} {rest}", ELLIPSIS, f"grafted onto {previous!r}")

    match = MODIFIER.match(text)
    if match and len(content_words(text)) < MIN_CONTENT + 2:
        rest = match.group("rest").strip()
        return Rewrite(text, f"{previous} {rest}", ELLIPSIS, f"grafted onto {previous!r}")
    return None


def _carryover(text: str, memory: Memory) -> Rewrite | None:
    """Last resort: append what the conversation has recently been about."""
    recent = [w for w in memory.recent_content(turns=1) if w not in content_words(text)]
    if not recent:
        return None
    return Rewrite(text, f"{text} {' '.join(recent)}", CARRYOVER, f"added {', '.join(recent)}")


def rewrite(text: str, memory: Memory, rules: tuple[str, ...] = ALL_RULES) -> Rewrite:
    """Resolve `text` against `memory`, applying the enabled rules in order.

    The first turn of a conversation has an empty memory, so every rule that
    needs history declines and the question is returned untouched — which is
    correct, and is why the build scores follow-up turns separately. Averaging
    first turns in would hide the whole effect under questions that were
    already standalone.
    """
    current = text
    applied: list[Rewrite] = []

    # A fragment needs its predicate however many content words it carries, so
    # the guard suppresses pronoun resolution only — never grafting.
    thin = len(content_words(text)) < GUARD_MIN
    may_resolve = PRONOUN in rules and (thin or GUARD not in rules)

    if may_resolve:
        step = _resolve_pronouns(current, memory)
        if step and step.changed:
            applied.append(step)
            current = step.text

    if SUBSTITUTE in rules:
        step = _substitute(current, memory)
        if step and step.changed:
            applied.append(step)
            current = step.text

    if ELLIPSIS in rules and not any(a.strategy == SUBSTITUTE for a in applied):
        step = _graft(current, memory)
        if step and step.changed:
            applied.append(step)
            current = step.text

    if CARRYOVER in rules and len(content_words(current)) < MIN_CONTENT:
        step = _carryover(current, memory)
        if step and step.changed:
            applied.append(step)
            current = step.text

    if not applied:
        return Rewrite(text, text, STANDALONE, "already stands on its own")
    return Rewrite(
        original=text,
        text=current,
        strategy="+".join(a.strategy for a in applied),
        detail="; ".join(a.detail for a in applied),
    )
