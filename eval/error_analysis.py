"""Lists every misclassified email from the personal benchmark.

By default it reads eval/data/personal_predictions.csv written by
evaluate.py (fast, no model needed). Use --rerun to classify again."""
import argparse
import os

import pandas as pd

from _paths import PERSONAL_PREDICTIONS, PERSONAL_TEST

parser = argparse.ArgumentParser()
parser.add_argument("--rerun", action="store_true", help="re-classify instead of reading saved predictions")
args = parser.parse_args()

if args.rerun or not os.path.exists(PERSONAL_PREDICTIONS):
    from tqdm import tqdm
    from classifier.classifier import classify_email, load_model

    load_model()
    df = pd.read_csv(PERSONAL_TEST).fillna("")
    df = df[df["true_category"].str.strip() != ""]
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        result = classify_email(row["subject"], row["sender"], row["body"])
        rows.append({
            "subject": row["subject"],
            "sender": row["sender"],
            "true_category": row["true_category"].strip(),
            "predicted_category": result["category"],
            "confidence": result.get("confidence"),
            "reasoning": result.get("reasoning"),
        })
    out = pd.DataFrame(rows)
    out["correct"] = out["true_category"] == out["predicted_category"]
    out.to_csv(PERSONAL_PREDICTIONS, index=False)
else:
    out = pd.read_csv(PERSONAL_PREDICTIONS).fillna("")
    out["correct"] = out["correct"].astype(str).str.lower() == "true"

wrong = out[~out["correct"]].copy()
wrong["subject"] = wrong["subject"].astype(str).str[:70]
print(f"\n{len(wrong)} / {len(out)} rows misclassified\n")

print("Most common mistakes (true -> predicted):")
pairs = wrong.groupby(["true_category", "predicted_category"]).size().sort_values(ascending=False)
print(pairs.to_string())

print("\nAll misclassified rows:")
pd.set_option("display.max_colwidth", 70)
print(wrong[["subject", "true_category", "predicted_category", "confidence"]].to_string(index=False))
