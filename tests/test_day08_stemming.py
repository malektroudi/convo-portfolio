"""The stemmer, tested on convergence rather than on prettiness of output.

What matters is not that `stem("closing")` returns a word — it returns "clos",
which is not one. What matters is that every inflection of a word lands on the
same string, because that is the only property the citation gate uses.
"""

from __future__ import annotations

import pytest

from convo.projects.day08_rag_cites.stemming import MIN_STEM, stem


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


class TestShortWordsSurviveTheRules:
    """Words that merely *end* in a suffix must not be stripped into nothing.

    Day 11's lexicon found these: document headings use vocabulary the
    retrieval path never reached, and three families were broken by rules
    firing on words too short to survive them.
    """

    @pytest.mark.parametrize(
        "forms",
        [
            ("need", "needs", "needed"),
            ("add", "adds", "added"),
            ("feed", "feeds"),
            ("seed", "seeds"),
        ],
    )
    def test_a_short_base_form_is_not_stripped_below_its_family(self, forms):
        stems = {stem(f) for f in forms}
        assert len(stems) == 1, f"{forms} gave {stems}"

    def test_the_base_form_is_not_the_odd_one_out(self):
        # The original bug: "needs" and "needed" both gave "need" while "need"
        # itself gave "ne", so the base form matched neither of its own
        # inflections. That is worse than not stemming at all.
        assert stem("need") == stem("needs") == stem("needed")

    def test_undoubling_is_skipped_rather_than_abandoned(self):
        # "added" -> "add" is already correct; undoubling it to "ad" was the
        # bug. Skipping that one step must not cost the whole strip.
        assert stem("added") == "add"
        assert stem("stopped") == "stop"

    def test_no_stem_is_shorter_than_the_floor(self):
        words = ("need", "added", "used", "goes", "seed", "deed", "agreed", "being")
        for word in words:
            assert len(stem(word)) >= MIN_STEM, f"{word} -> {stem(word)!r}"


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

    def test_short_irregular_verbs_do_not_converge(self):
        # The cost of the floor that fixed "need". Stripping these would leave
        # one or two characters, so they are left as typed and their forms do
        # not meet. A lemmatizer would catch them; a suffix stemmer cannot.
        assert stem("go") != stem("going")
        assert stem("use") != stem("used")

    def test_irregular_plurals_are_not_handled(self):
        # A lemmatizer would catch these; a suffix stemmer cannot.
        assert stem("people") != stem("person")
        assert stem("children") != stem("child")

    def test_it_does_not_collide_universities_with_universe(self):
        # The -ies rule happens to avoid the collision a cruder stemmer makes.
        assert stem("universities") != stem("universe")
