"""Builds eval/data/phishing_test.csv — a balanced 150-row sample of the
Kaggle "Phishing Email" dataset. Download Phishing_Email.csv from Kaggle
into eval/data/ first."""
import pandas as pd

from _paths import PHISHING_RAW, PHISHING_TEST

SAMPLES_PER_CLASS = 75

df = pd.read_csv(PHISHING_RAW)
df = df.dropna(subset=["Email Text", "Email Type"])
df = df.groupby("Email Type", group_keys=False).apply(
    lambda x: x.sample(min(len(x), SAMPLES_PER_CLASS), random_state=42)
).reset_index(drop=True)

out = pd.DataFrame({
    "subject": "",  # this dataset has no separate subject field
    "sender": "",   # or sender field — classify_email() handles empty strings fine
    "body": df["Email Text"].astype(str).str[:2000],
    "true_label": ["Phishing" if t == "Phishing Email" else "Not Phishing" for t in df["Email Type"]],
})
out.to_csv(PHISHING_TEST, index=False)
print(f"Wrote {len(out)} rows to {PHISHING_TEST}")
