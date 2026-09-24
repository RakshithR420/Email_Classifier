"""Accuracy evaluation.

    python eval/evaluate.py                     # both benchmarks
    python eval/evaluate.py --only phishing
    python eval/evaluate.py --only personal --limit 50
    python eval/evaluate.py --only personal --personal-file eval/data/personal_holdout.csv

Prints a classification report + confusion matrix for each benchmark and
saves per-row predictions to eval/data/*_predictions.csv."""
import argparse
import os

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from _paths import PERSONAL_PREDICTIONS, PERSONAL_TEST, PHISHING_PREDICTIONS, PHISHING_TEST
from classifier.classifier import classify_email, get_settings, load_model


def run(df, desc):
    preds = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        preds.append(classify_email(row["subject"], row["sender"], row["body"]))
    return preds


def report(title, y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    print(f"\n=== {title} ===")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)
    print("Confusion matrix (rows = true, columns = predicted):")
    print(cm.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["phishing", "personal"])
    parser.add_argument("--limit", type=int, help="only use the first N rows of each dataset")
    parser.add_argument("--personal-file", default=PERSONAL_TEST,
                        help="labeled CSV for the personal benchmark (e.g. eval/data/personal_holdout.csv)")
    args = parser.parse_args()

    load_model()
    adapter = get_settings()["model"].get("adapter_path") or "(base model)"
    print(f"Model: {get_settings()['model']['name']}  adapter: {adapter}")

    if args.only in (None, "phishing"):
        df = pd.read_csv(PHISHING_TEST).fillna("")
        if args.limit:
            df = df.head(args.limit)
        preds = run(df, "Phishing benchmark")
        y_pred = ["Phishing" if p["category"] == "Phishing" else "Not Phishing" for p in preds]
        report("Phishing benchmark", list(df["true_label"]), y_pred)
        df.assign(predicted=y_pred, confidence=[p["confidence"] for p in preds]).drop(columns=["body"]).to_csv(
            PHISHING_PREDICTIONS, index=False)

    if args.only in (None, "personal"):
        if not os.path.exists(args.personal_file):
            print(f"\nSkipping personal benchmark: {args.personal_file} not found (run eval/build_personal_template.py).")
            return
        df = pd.read_csv(args.personal_file).fillna("")
        unlabeled = (df["true_category"].str.strip() == "").sum()
        df = df[df["true_category"].str.strip() != ""]
        if unlabeled:
            print(f"Skipping {unlabeled} rows with no true_category.")
        if args.limit:
            df = df.head(args.limit)
        preds = run(df, "Personal inbox benchmark")
        y_pred = [p["category"] for p in preds]
        report(f"Personal inbox benchmark ({len(df)} emails)", list(df["true_category"].str.strip()), y_pred)
        out = df.drop(columns=["body"]).assign(
            predicted_category=y_pred,
            confidence=[p["confidence"] for p in preds],
            reasoning=[p["reasoning"] for p in preds],
        )
        out["correct"] = out["true_category"].str.strip() == out["predicted_category"]
        out.to_csv(PERSONAL_PREDICTIONS, index=False)
        print(f"\nPer-row predictions saved to {PERSONAL_PREDICTIONS}")


if __name__ == "__main__":
    main()
