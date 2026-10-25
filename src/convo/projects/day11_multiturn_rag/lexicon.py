"""The topics a follow-up can be about, derived from the corpus itself.

A rewriter needs to know what counts as a *thing* in this domain, because
"what about for a private event" is only expandable if "private event" is
recognised as an entity rather than three words.

Hand-writing that list would work and would rot: add an article about gift
cards and the rewriter silently stops handling questions about them. So the
lexicon is built from the corpus — document titles and markdown headings are
already a human-curated list of what this help centre is about, which is the
most honest source available and costs nothing to keep current.

What this does not do is generalise. A closed domain with fourteen articles
has a vocabulary small enough to enumerate; an open-domain assistant does not,
and would need an entity linker or a model. That limit is the point of
measuring the rewriter against an oracle rather than claiming it is solved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from functools import lru_cache

from convo.projects.day08_rag_cites.chunking import Document, load_corpus
from convo.projects.day08_rag_cites.index import STOPWORDS, WORD
from convo.projects.day08_rag_cites.stemming import stem

HEADING = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)

# Leading words that make a heading a sentence fragment rather than a topic:
# "## When a deposit applies" is about deposits, not about "when".
#
# This list is short on purpose. An earlier version also stripped leading
# pronouns and gerunds to tidy headings like "## Letting us know", and it ate
# "## Thanksgiving and Canada Day" because that word ends in -ing. The tidying
# was not worth it: a topic is only ever matched against what a *user* typed,
# and nobody asks a question containing the words "letting us know", so an
# untidy topic is inert rather than wrong. Guessing at grammar is not.
HEADING_NOISE = frozenset(["what", "when", "where", "how", "why", "who"])

# Words that look like entities by frequency but carry no topic: every article
# is about "the restaurant", so matching it tells a rewriter nothing.
TOO_GENERAL = frozenset(
    ["restaurant", "article", "guest", "guests", "us", "we", "one", "it", "that", "here"]
)


@dataclass(frozen=True)
class Topic:
    """One thing a conversation can be about."""

    name: str
    """The phrase as a person would say it: "private event"."""
    doc_id: str
    """The article it came from, used to explain a rewrite afterwards."""
    stems: tuple[str, ...]
    """Its content words, stemmed, for matching against what someone typed."""
    plural: bool = False
    """Whether "they"/"them" can refer to it, as opposed to "it"/"that"."""
    spoken: bool = True
    """True when the user named it; False when retrieval inferred it.

    An inferred topic is a guess built on another guess — retrieval can be
    wrong, and then the pronoun resolves against the wrong article. Keeping
    the distinction lets `Memory.antecedent` prefer evidence over inference
    instead of treating both as equally good.
    """

    def mentioned_in(self, text: str) -> bool:
        """Whether this topic's words all appear in `text`, in any order."""
        if not self.stems:
            return False
        typed = {stem(w) for w in WORD.findall(text.lower())}
        return all(s in typed for s in self.stems)


def _content_stems(phrase: str) -> tuple[str, ...]:
    return tuple(stem(w) for w in WORD.findall(phrase.lower()) if w not in STOPWORDS and len(w) > 2)


def _is_plural(phrase: str) -> bool:
    """Crude, and deliberately so: the last word carries the number."""
    words = WORD.findall(phrase.lower())
    if not words:
        return False
    last = words[-1]
    return last.endswith("s") and not last.endswith("ss") and len(last) > 3


def _clean(heading: str) -> str:
    """Drop a leading question word so a heading reads as a topic."""
    words = heading.strip().split()
    while words and words[0].lower() in HEADING_NOISE:
        words = words[1:]
    return " ".join(words)


def _heads_a_noun_phrase(name: str) -> bool:
    """Whether a phrase ends on the thing it is about.

    "## We need from you" is a sentence, and its last word is "you", so it
    names nothing a follow-up can refer to — yet its one content stem is
    "need", which matches any question containing the word. Requiring the
    final word to carry content is enough to drop that whole class of heading
    without guessing at grammar anywhere else.
    """
    words = name.split()
    if not words:
        return False
    last = WORD.findall(words[-1].lower())
    return bool(last) and last[-1] not in STOPWORDS and stem(last[-1]) not in TOO_GENERAL


@dataclass
class Lexicon:
    topics: tuple[Topic, ...] = field(default_factory=tuple)

    def find(self, text: str) -> list[Topic]:
        """Every topic mentioned in `text`, longest phrase first.

        Longest first matters: "private event" and "event" can both match, and
        the specific one is the one a follow-up is about.
        """
        found = [t for t in self.topics if t.mentioned_in(text)]
        return sorted(found, key=lambda t: (-len(t.stems), t.name))

    def best(self, text: str) -> Topic | None:
        found = self.find(text)
        return found[0] if found else None

    def topic_for(self, doc_id: str) -> Topic | None:
        """An article's own topic — its title, or the first heading that
        survives the noun-phrase filter when the title does not.

        This is the fallback antecedent, and in practice the most useful one:
        "where can I park" names no topic in its own words, but it retrieved
        the parking article, and that article is what "it" refers to next.
        Parking's title is "Parking and getting here", which ends on a word
        that names nothing, so the topic falls through to "street parking" —
        more specific than the title and no less correct.
        """
        for topic in self.topics:
            if topic.doc_id == doc_id:
                return replace(topic, spoken=False)
        return None


def build_lexicon(documents: list[Document] | None = None) -> Lexicon:
    """Titles and headings, deduplicated by their stem signature."""
    documents = documents if documents is not None else load_corpus()
    seen: dict[tuple[str, ...], Topic] = {}

    for document in documents:
        phrases = [document.title] + [_clean(h) for h in HEADING.findall(document.text)]
        for phrase in phrases:
            name = phrase.strip().lower()
            if not name:
                continue
            stems = _content_stems(name)
            if not stems or all(s in TOO_GENERAL for s in stems):
                continue
            if not _heads_a_noun_phrase(name):
                continue
            # First article to claim a phrase keeps it, so a topic resolves to
            # one place rather than wherever it was last seen.
            seen.setdefault(
                stems,
                Topic(name=name, doc_id=document.doc_id, stems=stems, plural=_is_plural(name)),
            )

    return Lexicon(tuple(seen.values()))


@lru_cache(maxsize=1)
def lexicon() -> Lexicon:
    return build_lexicon()
