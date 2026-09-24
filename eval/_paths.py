"""Shared paths so the eval scripts work no matter which folder you run them from."""
import os
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(EVAL_DIR)
DATA_DIR = os.path.join(EVAL_DIR, "data")

PHISHING_RAW = os.path.join(DATA_DIR, "Phishing_Email.csv")
PHISHING_TEST = os.path.join(DATA_DIR, "phishing_test.csv")
PERSONAL_TEST = os.path.join(DATA_DIR, "personal_test.csv")
PERSONAL_PREDICTIONS = os.path.join(DATA_DIR, "personal_predictions.csv")
PHISHING_PREDICTIONS = os.path.join(DATA_DIR, "phishing_predictions.csv")

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
