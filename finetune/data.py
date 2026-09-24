"""Data helpers for fine-tuning (kept separate so they can be unit-tested
without a GPU)."""

import json

import pandas as pd

from classifier.classifier import build_messages, get_settings


def load_labeled(path, categories):
    """Reads a CSV with subject, sender, body, true_category and keeps only
    rows whose label is a real category."""
    df = pd.read_csv(path).fillna("")
    df["true_category"] = df["true_category"].astype(str).str.strip()
    known = df["true_category"].isin(list(categories))
    dropped = df[~known & (df["true_category"] != "")]
    if len(dropped):
        print(f"WARNING: {path}: skipping {len(dropped)} rows with unknown labels: "
              f"{sorted(dropped['true_category'].unique())}")
    return df[known].reset_index(drop=True)


def split_train_test(df, test_size=0.25, seed=42):
    """Stratified split. Categories with fewer than 2 emails go to train only
    (they can't be split)."""
    counts = df["true_category"].value_counts()
    rare = df["true_category"].map(counts) < 2
    splittable = df[~rare]

    test_parts = []
    for _, group in splittable.groupby("true_category"):
        n_test = max(1, int(round(len(group) * test_size)))
        n_test = min(n_test, len(group) - 1)  # keep at least one for training
        test_parts.append(group.sample(n=n_test, random_state=seed))
    test_df = pd.concat(test_parts) if test_parts else df.iloc[0:0]
    train_df = df.drop(test_df.index)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def target_json(category):
    """The answer we teach the model to give."""
    return json.dumps({
        "category": category,
        "confidence": 0.9,
        "reasoning": f"The content matches the {category} category.",
    })


def build_examples(df, tokenizer, categories, max_length=1024):
    """Tokenizes (prompt, answer) pairs. Prompt tokens get label -100 so the
    model is only trained on producing the answer."""
    examples = []
    for _, row in df.iterrows():
        answer_text = target_json(row["true_category"]) + (tokenizer.eos_token or "")
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

        # If too long, shorten the email body (never the instructions or answer).
        body_limit = None
        while True:
            messages = build_messages(row["subject"], row["sender"], row["body"], categories, body_limit)
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            if len(prompt_ids) + len(answer_ids) <= max_length:
                break
            if body_limit is None:
                default_limit = get_settings().get("fetch", {}).get("body_char_limit", 1000)
                current = min(len(str(row["body"])), default_limit)
            else:
                current = body_limit
            if current <= 0:
                prompt_ids = None
                break
            body_limit = int(current * 0.7)

        if prompt_ids is None:
            continue
        examples.append({
            "input_ids": prompt_ids + answer_ids,
            "labels": [-100] * len(prompt_ids) + answer_ids,
        })
    return examples
