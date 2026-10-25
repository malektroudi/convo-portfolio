"""Three chunking strategies, all carrying exact offsets into the source.

Every chunk satisfies one invariant:

    document.text[chunk.start:chunk.end] == chunk.text

That is what makes a citation a citation. Most RAG demos return a chunk id and
call it a source, which is unfalsifiable — you cannot tell whether the cited
text is really in the document, because the chunk IS the only record of itself.
Keeping offsets means the claim can be checked against the original, a reader
can be shown the exact span highlighted in context, and a test can assert it.

Line endings are normalised to LF on load, because an offset is an index into a
particular string and a CRLF checkout would silently shift every one of them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "corpus"


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    """Normalised to LF. All offsets index into this."""

    def span(self, start: int, end: int) -> str:
        return self.text[start:end]


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    text: str
    start: int
    end: int
    strategy: str
    heading: str = ""
    """The nearest enclosing markdown heading, where the strategy tracks it."""
    index: int = 0

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}#{self.strategy}:{self.index}"

    def verify(self, document: Document) -> bool:
        """The invariant. A chunk that fails this cannot be cited."""
        return document.span(self.start, self.end) == self.text


def load_corpus(directory: Path | None = None) -> list[Document]:
    """Read the help centre, normalising line endings so offsets are stable."""
    directory = directory or CORPUS_DIR
    documents: list[Document] = []
    for path in sorted(directory.glob("*.md")):
        raw = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        first = next((ln for ln in raw.splitlines() if ln.startswith("# ")), "")
        documents.append(
            Document(
                doc_id=path.stem,
                title=first.removeprefix("# ").strip() or path.stem,
                text=raw,
            )
        )
    return documents


def _trim(text: str, start: int) -> tuple[str, int]:
    """Drop leading whitespace, moving `start` with it so offsets stay true."""
    stripped = text.lstrip()
    moved = start + (len(text) - len(stripped))
    return stripped.rstrip(), moved


def fixed(document: Document, size: int = 400, overlap: int = 0) -> list[Chunk]:
    """Cut every `size` characters, stepping by `size - overlap`.

    The baseline, and the one that cuts sentences in half. Included because it
    is what you get if you do not think about it, and day 9 measures what that
    costs.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be at least 0 and smaller than size")

    strategy = "fixed" if not overlap else f"fixed-overlap-{overlap}"
    step = size - overlap
    chunks: list[Chunk] = []
    for start in range(0, max(len(document.text), 1), step):
        raw = document.text[start : start + size]
        if not raw.strip():
            continue
        text, moved = _trim(raw, start)
        if not text:
            continue
        chunks.append(
            Chunk(
                doc_id=document.doc_id,
                text=text,
                start=moved,
                end=moved + len(text),
                strategy=strategy,
                index=len(chunks),
            )
        )
        if start + size >= len(document.text):
            break
    return chunks


HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def structural(document: Document, max_size: int = 700) -> list[Chunk]:
    """Split on markdown headings, then on blank lines if a section is too long.

    Respects the shape the author already gave the document. A section heading
    is prepended to its own chunk's text so a retrieved fragment still says what
    it is about — but the heading is NOT part of the cited span, because it is
    not where the answer lives.
    """
    boundaries = [m.start() for m in HEADING.finditer(document.text)]
    if not boundaries:
        boundaries = [0]
    if boundaries[0] != 0:
        boundaries.insert(0, 0)
    boundaries.append(len(document.text))

    chunks: list[Chunk] = []
    for begin, finish in zip(boundaries[:-1], boundaries[1:], strict=True):
        section = document.text[begin:finish]
        if not section.strip():
            continue
        match = HEADING.match(section)
        heading = match.group(2).strip() if match else ""
        body_start = begin + (match.end() if match else 0)

        body = document.text[body_start:finish]
        if len(body.strip()) <= max_size:
            pieces = [(body, body_start)]
        else:
            pieces = _split_paragraphs(body, body_start, max_size)

        for raw, offset in pieces:
            text, moved = _trim(raw, offset)
            if not text:
                continue
            chunks.append(
                Chunk(
                    doc_id=document.doc_id,
                    text=text,
                    start=moved,
                    end=moved + len(text),
                    strategy="structural",
                    heading=heading,
                    index=len(chunks),
                )
            )
    return chunks


def _split_paragraphs(body: str, offset: int, max_size: int) -> list[tuple[str, int]]:
    """Break an over-long section on blank lines, keeping offsets exact."""
    pieces: list[tuple[str, int]] = []
    cursor = 0
    buffer_start = 0
    buffer: list[str] = []

    for para in re.split(r"(\n\s*\n)", body):
        if not para:
            continue
        if para.strip() == "" and buffer:
            buffer.append(para)
            cursor += len(para)
            continue
        if buffer and sum(len(p) for p in buffer) + len(para) > max_size:
            pieces.append(("".join(buffer), offset + buffer_start))
            buffer, buffer_start = [], cursor
        if not buffer:
            buffer_start = cursor
        buffer.append(para)
        cursor += len(para)

    if buffer:
        pieces.append(("".join(buffer), offset + buffer_start))
    return pieces


@dataclass(frozen=True)
class Strategy:
    name: str
    describe: str
    run: object

    def __call__(self, document: Document) -> list[Chunk]:
        return self.run(document)  # type: ignore[operator]


STRATEGIES: tuple[Strategy, ...] = (
    Strategy(
        "fixed",
        "400 characters, no overlap. Cuts mid-sentence and loses the heading.",
        lambda d: fixed(d, size=400, overlap=0),
    ),
    Strategy(
        "fixed-overlap",
        "400 characters with 80 of overlap, so a cut sentence survives in a neighbour.",
        lambda d: fixed(d, size=400, overlap=80),
    ),
    Strategy(
        "structural",
        "Split on markdown headings, then paragraphs. Respects the author's shape.",
        lambda d: structural(d, max_size=700),
    ),
)

BY_NAME: dict[str, Strategy] = {s.name: s for s in STRATEGIES}


def chunk_corpus(documents: list[Document], strategy: str) -> list[Chunk]:
    """Every chunk of every document under one strategy."""
    if strategy not in BY_NAME:
        raise ValueError(f"unknown strategy {strategy!r}; have {sorted(BY_NAME)}")
    runner = BY_NAME[strategy]
    out: list[Chunk] = []
    for document in documents:
        out.extend(runner(document))
    return out


@dataclass
class CorpusStats:
    strategy: str
    chunks: int
    mean_chars: float
    median_chars: float
    min_chars: int
    max_chars: int
    sentence_breaks: int = 0
    """Chunks that start or end mid-sentence — the cost of ignoring structure."""
    covered: float = 0.0
    """Share of corpus characters reachable through at least one chunk."""
    detail: dict = field(default_factory=dict)
