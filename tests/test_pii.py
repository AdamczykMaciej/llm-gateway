from llm_gateway.pii import mask_pii


def test_masks_email():
    assert mask_pii("contact alex@example.com now") == "contact [EMAIL] now"


def test_masks_phone_international():
    assert "[PHONE]" in mask_pii("call +48 123 456 789 today")


def test_masks_phone_local_dashed():
    assert "[PHONE]" in mask_pii("call 123-456-7890 today")


def test_masks_pesel_like_number():
    assert "[ID]" in mask_pii("PESEL: 12345678901")


def test_masks_iban():
    assert "[IBAN]" in mask_pii("send to PL61109010140000071219812874")


def test_masks_card_number():
    assert "[CARD]" in mask_pii("card 4111 1111 1111 1111 expires soon")


def test_masks_multiple_pii_types_together():
    text = "Email me at a@b.com or call 123-456-7890."
    result = mask_pii(text)
    assert "a@b.com" not in result
    assert "[EMAIL]" in result
    assert "[PHONE]" in result


def test_plain_text_passes_through_unchanged():
    text = "The quarterly report is due Friday."
    assert mask_pii(text) == text


def test_empty_string():
    assert mask_pii("") == ""


def test_none_passthrough():
    assert mask_pii(None) is None
