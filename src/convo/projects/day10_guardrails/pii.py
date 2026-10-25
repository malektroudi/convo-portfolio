"""PII detection and reversible redaction, on the way in and on the way out.

Two directions, because they fail differently. Inbound redaction stops a card
number reaching a log, an index, or a model provider. Outbound scanning catches
the assistant repeating something back — which day 6's scorers already test for
and which a model will happily do when the number is sitting in its context.

Redaction is reversible by design. An irreversible scrub means the pipeline
cannot tell a guest "the card ending 1111" afterwards, and teams then work
around the redactor rather than through it. Placeholders map back through a
`Vault` that stays in memory and is never logged.

Card numbers are Luhn-checked rather than pattern-matched alone. A sixteen-digit
string is usually not a card — booking references, order numbers and timestamps
all look like one — and a redactor that cries wolf gets switched off.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Digit, then (separator? digit) repeated — so the match always ENDS on a
# digit. The obvious `(?:\d[ -]?){13,19}` lets the last repetition swallow a
# trailing space, which silently eats the space after the number and makes the
# same card redact to two different placeholders depending on what follows it.
CARD = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")
# North American and common international shapes, deliberately not exhaustive.
PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\d)")

PLACEHOLDER = re.compile(r"\[(CARD|EMAIL|PHONE)_(\d+)\]")


def luhn(digits: str) -> bool:
    """The checksum every real card number satisfies."""
    numbers = [int(d) for d in digits if d.isdigit()]
    if len(numbers) < 13:
        return False
    total = 0
    for position, digit in enumerate(reversed(numbers)):
        if position % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


@dataclass
class Finding:
    kind: str
    text: str
    start: int
    end: int
    placeholder: str = ""

    @property
    def tail(self) -> str:
        """The last four digits, for saying "the card ending 1111" safely."""
        digits = [c for c in self.text if c.isdigit()]
        return "".join(digits[-4:])


@dataclass
class Vault:
    """Placeholder to original. Lives in memory, never logged, never persisted."""

    entries: dict[str, Finding] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    def store(self, finding: Finding) -> str:
        # The same value redacts to the same placeholder within a conversation,
        # so "is that the card ending 1111" stays answerable.
        for placeholder, existing in self.entries.items():
            if existing.kind == finding.kind and existing.text == finding.text:
                return placeholder
        index = self._counts.get(finding.kind, 0) + 1
        self._counts[finding.kind] = index
        placeholder = f"[{finding.kind}_{index}]"
        self.entries[placeholder] = finding
        return placeholder

    def restore(self, text: str) -> str:
        """Put the originals back. Only for showing a guest their own data."""

        def swap(match: re.Match) -> str:
            found = self.entries.get(match.group(0))
            return found.text if found else match.group(0)

        return PLACEHOLDER.sub(swap, text)

    def __len__(self) -> int:
        return len(self.entries)


def detect(text: str) -> list[Finding]:
    """Everything sensitive in `text`, left to right, without overlaps."""
    found: list[Finding] = []
    for kind, pattern in (("CARD", CARD), ("EMAIL", EMAIL), ("PHONE", PHONE)):
        for match in pattern.finditer(text):
            value = match.group(0)
            if kind == "CARD" and not luhn(value):
                continue
            found.append(Finding(kind, value, match.start(), match.end()))

    # A card number also matches the phone pattern in places; the longer, more
    # specific match wins and anything it covers is dropped.
    found.sort(key=lambda f: (f.start, -(f.end - f.start)))
    kept: list[Finding] = []
    for finding in found:
        if any(k.start < finding.end and finding.start < k.end for k in kept):
            continue
        kept.append(finding)
    return kept


def redact(text: str, vault: Vault | None = None) -> tuple[str, Vault]:
    """Replace sensitive values with stable placeholders."""
    vault = vault if vault is not None else Vault()
    findings = detect(text)
    if not findings:
        return text, vault

    out = []
    cursor = 0
    for finding in findings:
        placeholder = vault.store(finding)
        finding.placeholder = placeholder
        out.append(text[cursor : finding.start])
        out.append(placeholder)
        cursor = finding.end
    out.append(text[cursor:])
    return "".join(out), vault


def leaks(text: str) -> list[Finding]:
    """Sensitive values present in text that should not contain any.

    The outbound check. Separate from `detect` only in intent, which is the
    point — the same detector runs in both directions so a value that was
    redacted on the way in cannot slip out through a gap in a second
    implementation.
    """
    return detect(text)


def safe_reference(finding: Finding) -> str:
    """How to refer to a redacted value without repeating it."""
    if finding.kind == "CARD":
        return f"the card ending {finding.tail}"
    if finding.kind == "EMAIL":
        domain = finding.text.split("@")[-1]
        return f"the address at {domain}"
    return f"the number ending {finding.tail}"
