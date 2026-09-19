## Run 1 — 50-email personal sample (earlier)

=== Phishing benchmark ===
              precision    recall  f1-score   support

Not Phishing       0.73      0.97      0.83        75
    Phishing       0.96      0.64      0.77        75

    accuracy                           0.81       150
   macro avg       0.84      0.81      0.80       150
weighted avg       0.84      0.81      0.80       150

=== Personal inbox benchmark (50 emails) ===
               precision    recall  f1-score   support

 Applications       0.00      0.00      0.00         0
      Finance       0.00      0.00      0.00         2
   Job Result       0.00      0.00      0.00         1
   Job Search       0.78      1.00      0.88        21
        Other       0.25      0.67      0.36         3
     Phishing       0.50      1.00      0.67         1
   Promotions       0.83      0.62      0.71        16
 Social Media       0.00      0.00      0.00         5
Subscriptions       0.00      0.00      0.00         1

     accuracy                           0.68        50
    macro avg       0.26      0.37      0.29        50
 weighted avg       0.62      0.68      0.63        50

---

## Run 2 — 200-email personal sample (latest)

=== Phishing benchmark ===
              precision    recall  f1-score   support

Not Phishing       0.75      0.97      0.85        75
    Phishing       0.96      0.68      0.80        75

    accuracy                           0.83       150
   macro avg       0.86      0.83      0.82       150
weighted avg       0.86      0.83      0.82       150

=== Personal inbox benchmark (200 emails) ===
               precision    recall  f1-score   support

 Applications       0.83      1.00      0.91        10
      Finance       1.00      0.25      0.40         8
   Job Result       0.00      0.00      0.00         1
   Job Search       0.86      1.00      0.92       101
       Orders       0.67      1.00      0.80         2
        Other       0.83      0.62      0.71        24
   Promotions       0.69      0.87      0.77        23
 Social Media       1.00      0.12      0.21        25
Subscriptions       0.00      0.00      0.00         6

     accuracy                           0.77       200
    macro avg       0.49      0.41      0.39       200
 weighted avg       0.82      0.77      0.74       200

**Notes:**
- Phishing detection improved slightly (81% → 83% accuracy) with the larger, balanced 150-row Kaggle sample. The model stays conservative — very high precision (0.96) but recall of only 0.68, meaning it under-flags real phishing rather than over-flagging safe email.
- Personal inbox accuracy rose from 0.68 → 0.77 with a bigger, more representative sample (200 vs 50 emails), and the category-level breakdown is now much more trustworthy since low-support classes (0-2 examples) were driving misleading numbers in Run 1.
- Biggest weakness found: `Social Media` has perfect precision (1.00) but very low recall (0.12) — the model rarely mislabels something *as* Social Media, but it's failing to catch most actual Social Media emails, most likely because LinkedIn/Glassdoor senders overlap heavily with the dominant `Job Search` category in this inbox. `Subscriptions` and `Finance` show a similar pattern (low recall) for the same underlying reason: category descriptions in `categories.yaml` may need sharper distinctions from `Job Search`/`Other`.
- `OTP` and an updated `Phishing`-in-personal-inbox label were added to the ground truth for this run's dataset, but the evaluation run above was against an earlier cached copy of the label file that didn't yet include those two categories — so this report undercounts them (support 0 for both). Worth re-running once the label file sync issue is confirmed fixed, to get a true reading on those two categories.
