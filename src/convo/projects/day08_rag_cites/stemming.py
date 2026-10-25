"""A small suffix stemmer, so "park" matches "parking".

Day 10's two red-team failures were both over-refusals with this one cause: the
citation gate compared tokens as exact strings, so `where can I park` shared no
word with *there is paid street parking on Dundas* and the pipeline declined to
cite a passage retrieval had found correctly.

The design problem is that stripping suffixes alone does not make forms meet.
"closes" loses its -s to give "close" while "closing" loses its -ing to give
"clos", and the two still do not match. So after the suffix comes off, a
trailing -e comes off too, and every form lands on the same stem:

    close, closes, closing, closed   ->  clos
    park, parks, parking, parked     ->  park

This is a blunt instrument and deliberately so. A real Porter stemmer carries
rules for -ational, -fulness and a dozen more; a lemmatizer would be more
accurate still and would need a dictionary. For a restaurant help centre the
cost of being wrong is one over-matched word, and the cost of the alternative
is a dependency and a lookup table.

Known over-stemming it accepts: "universities" and "universe" both reduce to
"univers". Nothing in this corpus trips it, and a test pins the cases that
matter rather than claiming it is generally correct.
"""

from __future__ import annotations

# Doubled consonants that an -ed or -ing ending created: "stopped" -> "stopp".
DOUBLED = "bdfglmnprt"


def stem(word: str) -> str:
    """Reduce a lowercase word to a form its inflections share."""
    stemmed = word
    stripped_verb = False

    if len(stemmed) > 4 and stemmed.endswith("ing"):
        stemmed, stripped_verb = stemmed[:-3], True
    elif len(stemmed) > 3 and stemmed.endswith("ed"):
        stemmed, stripped_verb = stemmed[:-2], True
    elif len(stemmed) > 3 and stemmed.endswith("ies"):
        stemmed = stemmed[:-3] + "y"
    elif len(stemmed) > 3 and stemmed.endswith("es") and not stemmed.endswith("ses"):
        stemmed = stemmed[:-2]
    elif len(stemmed) > 3 and stemmed.endswith("s") and not stemmed.endswith("ss"):
        stemmed = stemmed[:-1]

    # "stopped" -> "stopp" -> "stop", so it meets "stop" and "stopping".
    if stripped_verb and len(stemmed) > 2 and stemmed[-1] == stemmed[-2] and stemmed[-1] in DOUBLED:
        stemmed = stemmed[:-1]

    # The step that makes the forms actually converge.
    if len(stemmed) > 3 and stemmed.endswith("e"):
        stemmed = stemmed[:-1]

    return stemmed
