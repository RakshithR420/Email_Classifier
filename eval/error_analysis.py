import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm import tqdm
from classifier.classifier import classify_email, load_model

load_model()

personal_df = pd.read_csv("eval/data/personal_test.csv").fillna("")

rows = []
for _, row in tqdm(personal_df.iterrows(), total=len(personal_df), desc="Classifying"):
    result = classify_email(row["subject"], row["sender"], row["body"])
    rows.append({
        "subject": row["subject"][:70],
        "true_category": row["true_category"],
        "predicted_category": result["category"],
        "confidence": result.get("confidence"),
        "correct": row["true_category"] == result["category"],
    })

out = pd.DataFrame(rows)
out.to_csv("eval/data/personal_predictions.csv", index=False)

wrong = out[~out["correct"]]
print(f"\n{len(wrong)} / {len(out)} rows misclassified\n")
print(wrong[["subject", "true_category", "predicted_category", "confidence"]].to_string(index=False))
