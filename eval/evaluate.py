import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.metrics import classification_report
from classifier.classifier import classify_email, load_model
from tqdm import tqdm

load_model()

phishing_df = pd.read_csv("eval/data/phishing_test.csv").fillna("")
true_p, pred_p = [], []
for _, row in tqdm(phishing_df.iterrows(), total=phishing_df.shape[0], desc="Processing Phishing Data"):
    result = classify_email(row["subject"], row["sender"], row["body"])
    predicted = "Phishing" if result["category"] == "Phishing" else "Not Phishing"
    true_p.append(row["true_label"])
    pred_p.append(predicted)

print("=== Phishing benchmark ===")
print(classification_report(true_p, pred_p))

personal_df = pd.read_csv("eval/data/personal_test.csv").fillna("")
true_c, pred_c = [], []
for _, row in tqdm(personal_df.iterrows(), total=personal_df.shape[0], desc="Processing Personal Data"):
    result = classify_email(row["subject"], row["sender"], row["body"])
    true_c.append(row["true_category"])
    pred_c.append(result["category"])

print("=== Personal inbox benchmark ===")
print(classification_report(true_c, pred_c))
