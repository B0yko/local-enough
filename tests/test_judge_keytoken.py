"""The deterministic key-token check: owner, amount and date forms."""

from __future__ import annotations

from local_enough.judge import keytoken


def test_check_owner_case_insensitive() -> None:
    assert keytoken.check_owner("Amara", "amara will follow up by Friday.") is True
    assert keytoken.check_owner("Amara", "Wren will follow up by Friday.") is False


def test_check_amount_accepts_comma_form() -> None:
    assert keytoken.check_amount("$24,000", "The budget was set at $24,000.") is True


def test_check_amount_accepts_k_form_either_side() -> None:
    assert keytoken.check_amount("$24,000", "The budget was set at 24k.") is True
    assert keytoken.check_amount("12.5k", "The budget was set at 12,500.") is True


def test_check_amount_strips_currency_symbols() -> None:
    assert keytoken.check_amount("€32,000", "Budget for the pilot: 32000 euros.") is True


def test_check_amount_rejects_wrong_number() -> None:
    assert keytoken.check_amount("$24,000", "The budget was set at $25,000.") is False


def test_check_date_iso_form() -> None:
    assert keytoken.check_date("14 April", "They will consolidate the systems by 2026-04-14.") is True


def test_check_date_month_day_form() -> None:
    assert keytoken.check_date("14 April", "They will launch on April 14.") is True


def test_check_date_abbreviated_month_form() -> None:
    assert keytoken.check_date("14 April", "They will launch on Apr 14.") is True
    assert keytoken.check_date("14 April", "They will launch on 14 Apr.") is True


def test_check_date_slash_form() -> None:
    assert keytoken.check_date("14 April", "They will launch on 14/04.") is True


def test_check_date_weekday_plus_date_form() -> None:
    assert keytoken.check_date("14 April", "They will launch on Thursday 14 April.") is True


def test_check_date_wrong_day_rejected() -> None:
    assert keytoken.check_date("14 April", "They will launch on 15 April.") is False


def test_check_date_wrong_month_rejected() -> None:
    assert keytoken.check_date("14 April", "They will launch on 14 May.") is False


def test_check_date_vague_phrase_falls_back_to_substring() -> None:
    assert keytoken.check_date("the end of the month", "They agreed to postpone to the end of the month.") is True
    assert keytoken.check_date("the end of the month", "They agreed to postpone to next week.") is False


def test_check_date_weekday_only_falls_back_to_substring() -> None:
    assert keytoken.check_date("Friday", "Amara will update the tracker by Friday.") is True
    assert keytoken.check_date("Friday", "Amara will update the tracker by Monday.") is False


def test_check_fact_requires_every_key_token() -> None:
    key_tokens = {"owner": "Amara", "date": "Friday"}
    assert keytoken.check_fact(key_tokens, "Amara will update the tracker by Friday.") is True
    assert keytoken.check_fact(key_tokens, "Amara will update the tracker by Monday.") is False
    assert keytoken.check_fact(key_tokens, "Wren will update the tracker by Friday.") is False


def test_check_fact_amount_only() -> None:
    assert keytoken.check_fact({"amount": "$24,000"}, "Budget set at $24,000.") is True
    assert keytoken.check_fact({"amount": "$24,000"}, "Budget set at $10,000.") is False


def test_check_fact_empty_key_tokens_always_passes() -> None:
    assert keytoken.check_fact({}, "anything at all") is True
