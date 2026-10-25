"""A bounded conversation buffer: what a rewriter is allowed to look back at.

Three things are worth saying about the bound.

It is not a token budget. There is no model here to run out of context, so the
limit exists for a different reason: **stale topics actively hurt**. A buffer
that remembers everything will happily resolve "is it free on Sundays" against
a deposit mentioned six turns ago. Forgetting is a feature, and the build
sweeps the depth so the cost of remembering too much is a measurement rather
than an assertion.

What is stored is not the transcript. Each turn keeps the words the user typed,
the query that was actually retrieved with, and the article that came back —
because the article is the single most reliable signal of what the
conversation is about, and it is free. Storing the assistant's reply text
would add nothing a rewriter can use and would make the buffer a transcript
with a length problem.

Turns are appended after retrieval, so a rewrite never sees its own result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from convo.projects.day08_rag_cites.index import STOPWORDS, WORD
from convo.projects.day11_multiturn_rag.lexicon import Lexicon, Topic, lexicon

DEFAULT_DEPTH = 3
"""Turns of history a rewrite may use. Swept in the build rather than guessed."""


@dataclass(frozen=True)
class Turn:
    """One user turn, after it has been resolved and retrieved."""

    text: str
    """What the user actually typed."""
    query: str
    """What retrieval was given — the rewritten form, or `text` if unchanged."""
    doc_id: str | None = None
    """The article the top hit came from, if anything was retrieved."""
    heading: str = ""
    topics: tuple[Topic, ...] = ()
    """Lexicon topics named in this turn, most specific first."""

    @property
    def content_words(self) -> tuple[str, ...]:
        return tuple(w for w in WORD.findall(self.text.lower()) if w not in STOPWORDS)


@dataclass
class Memory:
    """The last `depth` turns, oldest first."""

    depth: int = DEFAULT_DEPTH
    turns: list[Turn] = field(default_factory=list)
    lex: Lexicon = field(default_factory=lexicon)

    def remember(self, turn: Turn) -> None:
        self.turns.append(turn)
        if len(self.turns) > self.depth:
            del self.turns[: len(self.turns) - self.depth]

    def observe(self, text: str, query: str, doc_id: str | None, heading: str = "") -> Turn:
        """Build a turn from a resolved query and its result, and store it."""
        topics = list(self.lex.find(text))
        # The article retrieval landed on is the strongest available signal of
        # what this turn was about, and it costs nothing — most questions name
        # no topic in their own words ("where can I park" names none), so
        # without it the next turn's pronoun has nothing to resolve against.
        # It goes last: what the user said outranks what retrieval inferred.
        from_doc = self.lex.topic_for(doc_id) if doc_id else None
        if from_doc is not None and from_doc not in topics:
            topics.append(from_doc)

        turn = Turn(
            text=text,
            query=query,
            doc_id=doc_id,
            heading=heading,
            topics=tuple(topics),
        )
        self.remember(turn)
        return turn

    @property
    def last(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    def antecedent(self, plural: bool | None = None) -> Topic | None:
        """The most recent topic a pronoun could refer to.

        Evidence first, recency second. A topic the user *said* outranks one
        retrieval merely landed on, however recent — which is the same rule
        `observe` applies inside a turn, extended across turns because that is
        where it actually matters. The deposit conversation is the case:

            do I need a deposit for a table of eight   -> names "deposits"
            what if there are twenty of us             -> names nothing, and
                                                          retrieval misses
            how much is it                             -> "it" = ?

        Taking the most recent turn's topic resolves "it" to whatever article
        the failed middle turn happened to hit. Preferring what the user said
        reaches back past it to the deposit, which is what "it" means.

        Within each pass it is recency then specificity — "the private event
        deposit" beats "deposit" in the same breath. Number agreement filters
        when it can ("they" cannot refer to "street parking"); when nothing
        agrees, agreement is dropped rather than returning nothing, because a
        wrong-number antecedent still retrieves better than a bare pronoun.
        """
        # Four passes, each a complete scan of the buffer. Doing agreement
        # inside a single scan abandons it after the first turn that offers
        # any topic at all, which is not "preferring" agreement — it is
        # checking one turn for it and then giving up.
        passes = (
            (True, True),  # the user said it, and the number agrees
            (True, False),  # the user said it
            (False, True),  # retrieval inferred it, and the number agrees
            (False, False),  # anything at all
        )
        for spoken_only, agree in passes:
            if agree and plural is None:
                continue
            for turn in reversed(self.turns):
                for topic in turn.topics:
                    if spoken_only and not topic.spoken:
                        continue
                    if agree and topic.plural != plural:
                        continue
                    return topic
        return None

    def last_question(self) -> str | None:
        """The previous turn's *resolved* query, which is what to graft onto.

        Using the resolved form rather than the raw text makes resolution
        compound: once turn 2 has become "how long before do I have to cancel
        for a private event", turn 3's "and if it is five days before" inherits
        the private event rather than falling back to turn 1's standard
        booking. It also means an early mistake propagates, which the
        per-turn-depth measurement in the build is there to expose.
        """
        return self.turns[-1].query if self.turns else None

    def recent_content(self, turns: int = 1) -> tuple[str, ...]:
        """Content words from the last `turns` turns, most recent first."""
        words: list[str] = []
        for turn in reversed(self.turns[-turns:] if turns else []):
            for word in turn.content_words:
                if word not in words:
                    words.append(word)
        return tuple(words)
