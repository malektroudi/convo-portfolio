"""The stemmer, tested on convergence rather than on prettiness of output.

What matters is not that `stem("closing")` returns a word — it returns "clos",
which is not one. What matters is that every inflection of a word lands on the
same string, because that is the only property the citation gate uses.
"""

from __future__ import annotations

import pytest

from convo.projects.day08_rag_cites.stemming import stem


class TestConvergence:
    """Every form of a word must reduce to the same stem."""

    @pytest.mark.parametrize(
        "forms",
        [
            ("park", "parks", "parking", "parked"),
            ("close", "closes", "closing", "closed"),
            ("book", "books", "booking", "booked"),
            ("serve", "serves", "serving", "served"),
            ("cancel", "cancels", "cancelling", "cancelled"),
            ("charge", "charges", "charging", "charged"),
            ("seat", "seats", "seating", "seated"),
            ("open", "opens", "opening", "opened"),
        ],
    )
    def test_all_inflections_meet(self, forms):
        stems = {stem(f) for f in forms}
        assert len(stems) == 1, f"{forms} gave {stems}"

    def test_the_two_day_10_failures_now_match(self):
        assert stem("park") == stem("parking")
        assert stem("close") == stem("closes")


class TestPlurals:
    @pytest.mark.parametrize(
        ("plural", "singular"),
        [
            ("guests", "guest"),
            ("hours", "hour"),
            ("dishes", "dish"),
            ("policies", "policy"),
            ("allergens", "allergen"),
            ("tables", "table"),
        ],
    )
    def test_a_plural_meets_its_singular(self, plural, singular):
        assert stem(plural) == stem(singular)


class TestItDoesNotOverreach:
    @pytest.mark.parametrize("word", ["gas", "bus", "is", "as"])
    def test_short_words_ending_in_s_are_left_alone(self, word):
        assert stem(word) == word

    @pytest.mark.parametrize("word", ["glass", "class", "address"])
    def test_a_double_s_is_not_stripped(self, word):
        assert stem(word) == word

    def test_unrelated_words_do_not_collide(self):
        assert stem("parking") != stem("packing")
        assert stem("booking") != stem("baking")

    def test_numbers_are_untouched(self):
        assert stem("200") == "200"
        assert stem("2026") == "2026"

    def test_an_empty_string_is_safe(self):
        assert stem("") == ""

    def test_most_words_are_idempotent(self):
        for word in ("parking", "policies", "cancelled", "dishes", "hours"):
            once = stem(word)
            assert stem(once) == once, f"{word} -> {once} -> {stem(once)}"


class TestKnownLimitations:
    """Pinned rather than hidden. Each is a real cost of a suffix stemmer."""

    def test_it_is_not_idempotent_for_stems_ending_in_s(self):
        # "close" -> "clos", and "clos" would strip again to "clo". Safe here
        # only because tokenise stems each side exactly once; a second pass
        # anywhere would quietly break matching, so it is pinned.
        assert stem("close") == "clos"
        assert stem("clos") == "clo"

    def test_it_over_stems_words_that_differ_by_a_trailing_e(self):
        assert stem("rating") == stem("rate") == stem("rat") == "rat"

    def test_irregular_plurals_are_not_handled(self):
        # A lemmatizer would catch these; a suffix stemmer cannot.
        assert stem("people") != stem("person")
        assert stem("children") != stem("child")

    def test_it_does_not_collide_universities_with_universe(self):
        # The -ies rule happens to avoid the collision a cruder stemmer makes.
        assert stem("universities") != stem("universe")
