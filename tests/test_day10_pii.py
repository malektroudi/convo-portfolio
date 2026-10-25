"""PII detection and redaction.

The false-positive tests matter as much as the detection ones. A redactor that
flags every sixteen-digit booking reference gets switched off by the first
person it inconveniences, and then it protects nothing.
"""

from __future__ import annotations

import pytest

from convo.projects.day10_guardrails.pii import (
    detect,
    leaks,
    luhn,
    redact,
    safe_reference,
)

# Test card numbers published by payment providers for exactly this purpose.
VISA = "4111 1111 1111 1111"
MASTERCARD = "5555555555554444"
AMEX = "378282246310005"


class TestLuhn:
    @pytest.mark.parametrize("number", [VISA, MASTERCARD, AMEX])
    def test_real_test_cards_pass(self, number):
        assert luhn(number)

    def test_a_digit_changed_fails(self):
        assert not luhn("4111 1111 1111 1112")

    def test_a_booking_reference_shaped_like_a_card_fails(self):
        assert not luhn("1234567890123456")

    def test_too_short_fails(self):
        assert not luhn("411111")


class TestDetect:
    def test_it_finds_a_card_number(self):
        found = detect(f"my card is {VISA}")
        assert [f.kind for f in found] == ["CARD"]
        assert found[0].text == VISA

    def test_it_finds_an_email(self):
        found = detect("write to sarah.chen@example.com please")
        assert found[0].kind == "EMAIL"
        assert found[0].text == "sarah.chen@example.com"

    def test_it_finds_a_phone_number(self):
        found = detect("call me on 416 555 0199")
        assert found[0].kind == "PHONE"

    def test_offsets_slice_back_to_the_value(self):
        text = f"card {VISA} and mail a@b.co"
        for finding in detect(text):
            assert text[finding.start : finding.end] == finding.text

    def test_a_sixteen_digit_reference_is_not_a_card(self):
        # The whole reason for the Luhn check.
        assert detect("booking reference 1234567890123456") == []

    def test_clean_text_yields_nothing(self):
        assert detect("a table for four on friday at 7:30pm") == []

    def test_a_date_is_not_a_phone_number(self):
        assert detect("the booking is for 2026-03-13") == []

    def test_several_values_are_all_found(self):
        found = detect(f"{VISA}, sarah@example.com, 416 555 0199")
        assert {f.kind for f in found} == {"CARD", "EMAIL", "PHONE"}

    def test_overlapping_matches_keep_the_longer_one(self):
        # A card number also looks like a phone number in places.
        found = detect(f"my card {VISA}")
        assert len(found) == 1
        assert found[0].kind == "CARD"


class TestRedact:
    def test_it_replaces_the_value_with_a_placeholder(self):
        text, vault = redact(f"my card is {VISA}")
        assert VISA not in text
        assert "[CARD_1]" in text
        assert len(vault) == 1

    def test_clean_text_passes_through_untouched(self):
        original = "a table for four"
        text, vault = redact(original)
        assert text == original and len(vault) == 0

    def test_the_same_value_gets_the_same_placeholder(self):
        text, _ = redact(f"{VISA} and again {VISA}")
        assert text.count("[CARD_1]") == 2
        assert "[CARD_2]" not in text

    def test_different_values_get_different_placeholders(self):
        text, _ = redact(f"{VISA} then {MASTERCARD}")
        assert "[CARD_1]" in text and "[CARD_2]" in text

    def test_a_vault_carries_across_turns(self):
        first, vault = redact(f"my card is {VISA}")
        second, vault = redact(f"is it still {VISA}", vault)
        assert "[CARD_1]" in first and "[CARD_1]" in second
        assert len(vault) == 1, "one card, one placeholder, across the conversation"

    def test_redaction_is_reversible(self):
        original = f"card {VISA} mail sarah@example.com"
        text, vault = redact(original)
        assert vault.restore(text) == original

    def test_restoring_text_with_no_placeholders_changes_nothing(self):
        _, vault = redact(f"card {VISA}")
        assert vault.restore("plain text") == "plain text"

    def test_an_unknown_placeholder_is_left_alone(self):
        _, vault = redact(f"card {VISA}")
        assert vault.restore("[CARD_99]") == "[CARD_99]"

    def test_surrounding_text_survives(self):
        text, _ = redact(f"please charge {VISA} tonight")
        assert text.startswith("please charge ") and text.endswith(" tonight")


class TestOutbound:
    def test_a_reply_repeating_a_card_is_caught(self):
        assert [f.kind for f in leaks(f"Booked on {VISA}.")] == ["CARD"]

    def test_a_reply_with_a_placeholder_is_clean(self):
        assert leaks("Booked on [CARD_1].") == []

    def test_a_reply_with_only_the_last_four_is_clean(self):
        assert leaks("Booked on the card ending 1111.") == []


class TestSafeReference:
    def test_a_card_becomes_its_last_four(self):
        finding = detect(f"{VISA}")[0]
        assert safe_reference(finding) == "the card ending 1111"
        assert VISA not in safe_reference(finding)

    def test_an_email_becomes_its_domain(self):
        finding = detect("sarah.chen@example.com")[0]
        reference = safe_reference(finding)
        assert "example.com" in reference and "sarah.chen" not in reference

    def test_a_phone_becomes_its_last_four(self):
        finding = detect("416 555 0199")[0]
        assert safe_reference(finding) == "the number ending 0199"

    def test_every_safe_reference_survives_the_outbound_check(self):
        for text in (VISA, "sarah@example.com", "416 555 0199"):
            reference = safe_reference(detect(text)[0])
            assert leaks(reference) == [], f"{reference!r} still contains PII"


class TestVault:
    def test_it_does_not_expose_values_in_its_repr(self):
        _, vault = redact(f"card {VISA}")
        # The vault holds originals; this pins that nobody prints it casually.
        assert len(vault) == 1
        assert isinstance(vault.entries["[CARD_1]"].text, str)
