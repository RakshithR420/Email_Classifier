import pandas as pd

from classifier.classifier import get_categories, parse_response
from finetune.data import build_examples, split_train_test, target_json


class FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return "".join(m["content"] for m in messages) + "<assistant>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_target_json_is_parseable_by_classifier():
    for cat in get_categories():
        assert parse_response(target_json(cat))[0]["category"] == cat


def test_split_is_stratified_and_disjoint():
    df = pd.DataFrame({"subject": [f"s{i}" for i in range(21)], "sender": "", "body": "",
                       "true_category": ["Job Search"] * 12 + ["Orders"] * 8 + ["OTP"]})
    train, test = split_train_test(df, test_size=0.25)
    assert set(train["subject"]).isdisjoint(test["subject"])
    assert len(train) + len(test) == 21
    assert "OTP" not in set(test["true_category"])  # single example stays in train
    assert set(test["true_category"]) == {"Job Search", "Orders"}


def test_only_answer_tokens_are_trained_and_long_bodies_are_shortened():
    df = pd.DataFrame({"subject": ["Hi"], "sender": "a", "body": ["x" * 3000], "true_category": ["Orders"]})
    ex = build_examples(df, FakeTokenizer(), get_categories(), max_length=4400)[0]
    assert len(ex["input_ids"]) <= 4400
    trained = "".join(chr(t) for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100)
    assert trained == target_json("Orders") + "<eos>"
    # instructions survive truncation
    prompt = "".join(chr(t) for t, l in zip(ex["input_ids"], ex["labels"]) if l == -100)
    assert "Respond ONLY with valid JSON" in prompt


def test_example_skipped_when_instructions_alone_are_too_long():
    df = pd.DataFrame({"subject": ["Hi"], "sender": "a", "body": ["x"], "true_category": ["Orders"]})
    assert build_examples(df, FakeTokenizer(), get_categories(), max_length=100) == []
