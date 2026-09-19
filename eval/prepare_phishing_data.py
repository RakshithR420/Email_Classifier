import pandas as pd

df = pd.read_csv("eval/data/Phishing_Email.csv")

# Keep it manageable — sample 150 rows total (balanced-ish) instead of using the whole dataset
df = df.dropna(subset=["Email Text", "Email Type"])
df = df.groupby("Email Type").apply(lambda x: x.sample(min(len(x), 75), random_state=42))
df = df.reset_index(level=0).reset_index(drop=True)

rows = []
for _, row in df.iterrows():
    rows.append({
        "subject": "",  # this dataset has no separate subject field
        "sender": "",   # or sender field — leave blank, classify_email() handles empty strings fine
        "body": str(row["Email Text"])[:2000],
        "true_label": "Phishing" if row["Email Type"] == "Phishing Email" else "Not Phishing",
    })

out = pd.DataFrame(rows)
out.to_csv("eval/data/phishing_test.csv", index=False)
print(f"Wrote {len(out)} rows to eval/data/phishing_test.csv")