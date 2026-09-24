"""Tests for prompt building, reply parsing and retry logic. No model or GPU needed."""
import json

from classifier import classifier as clf
from classifier.classifier import build_messages, classify_email, get_categories, parse_response


def test_every_category_is_in_prompt():
    prompt = build_messages("Hi", "a@b.com", "body")[-1]["content"]
    for name in get_categories():
        assert f"- {name}:" in prompt


def test_body_is_trimmed():
    prompt = build_messages("s", "x", "A" * 5000, body_limit=100)[-1]["content"]
    assert "A" * 100 in prompt and "A" * 101 not in prompt


def test_parse_plain_json():
    r, err = parse_response('{"category": "Orders", "confidence": 0.93, "reasoning": "Shipping update."}')
    assert err is None
    assert r == {"category": "Orders", "confidence": 0.93, "reasoning": "Shipping update."}


def test_parse_code_fence_think_and_extra_text():
    text = '<think>\nhmm\n</think>\nSure! ```json\n{"category": "social media", "confidence": "0.7", "reasoning": "x"}\n``` done'
    r, err = parse_response(text)
    assert err is None
    assert r["category"] == "Social Media"
    assert r["confidence"] == 0.7


def test_parse_percent_confidence_and_clamping():
    assert parse_response('{"category": "OTP", "confidence": 85}')[0]["confidence"] == 0.85
    assert parse_response('{"category": "OTP", "confidence": -3}')[0]["confidence"] == 0.0
    assert parse_response('{"category": "OTP", "confidence": "high"}')[0]["confidence"] == 0.0


def test_parse_rejects_bad_output():
    assert parse_response("")[0] is None
    assert parse_response("I think it's promotions")[0] is None
    r, err = parse_response('{"category": "Spam", "confidence": 0.9}')
    assert r is None and "Spam" in err


def test_retry_then_success():
    replies = iter(["not json", '{"category": "Finance", "confidence": 0.8, "reasoning": "Bank alert."}'])
    seen = []

    def fake_generate(messages, temperature):
        seen.append((len(messages), temperature))
        return next(replies)

    r = classify_email("Debit alert", "alerts@bank.com", "Rs 500 debited", generate_fn=fake_generate)
    assert r["category"] == "Finance" and r["valid"] and r["attempts"] == 2
    # second call must include the bad reply + correction message, with some randomness
    assert seen[0][0] == 2 and seen[1][0] == 4 and seen[1][1] > 0


def test_all_attempts_fail_falls_back_to_other():
    calls = []

    def fake_generate(messages, temperature):
        calls.append(1)
        return json.dumps({"category": "Nonsense"})

    r = classify_email("x", "y", "z", generate_fn=fake_generate)
    max_retries = clf.get_settings()["classify"]["max_retries"]
    assert len(calls) == 1 + max_retries
    assert r["category"] == "Other" and r["confidence"] == 0.0 and r["valid"] is False
