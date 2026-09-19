# Agentic Email Classifier

An agent that reads your Gmail inbox, classifies each email into a category using a locally-hosted, 4-bit quantized LLM (Qwen3-8B), and takes a configurable action (label or archive) via the Gmail API — no cloud LLM API calls, no data leaving your machine.

## Why this project

Most "email classifier" projects call a hosted LLM API and stop there. This one runs the model entirely locally (4-bit quantized to fit consumer GPU VRAM), integrates directly with the Gmail API to actually act on the inbox (not just print a label), and includes a real accuracy evaluation pipeline against both a public benchmark and the user's own hand-labeled inbox data — with documented, honest results including known failure modes.

## How it works

1. **Authenticate** with Gmail via OAuth2 (`auth/gmail_auth.py`).
2. **Fetch** recent inbox emails via the Gmail API, with rate-limit-aware retry/backoff (`auth/gmail_client.py`).
3. **Classify** each email by prompting a locally-loaded, 4-bit quantized Qwen3-8B model with the email content plus category descriptions, requesting structured JSON output (`classifier/classifier.py`).
4. **Act** on the result — apply a Gmail label, archive, or do nothing — based on a per-category config (`actions/actions.py`, `config/actions.yaml`).

Everything is orchestrated by `main.py`, which runs in `dry_run` mode by default (see `config/settings.yaml`) so nothing is changed in your inbox until you're ready.

## Categories

Defined in `config/categories.yaml` and fully configurable — currently includes Phishing, Job Search, Applications, Orders, Subscriptions, Personal, Finance, Promotions, Other, Job Result, Social Media, and OTP. Each category maps to an action (`label`, `archive`, or `none`) in `config/actions.yaml`.

## Evaluation

This project includes a real accuracy evaluation, not just anecdotal testing — see `eval/results.md` for full results. Two benchmarks are used:

- **Phishing detection**: a balanced 150-email sample from a public Kaggle phishing dataset (`eval/prepare_phishing_data.py`). Latest result: **83% accuracy** (96% precision / 68% recall on the Phishing class — the model is conservative, rarely false-flagging but missing some real phishing).
- **Personal inbox classification**: a hand-reviewed sample of the user's own inbox (`eval/build_personal_template.py` fetches emails; labels are manually verified before evaluation). Latest result: **77% accuracy** on 200 emails.

Run the evaluation yourself with:

```
python eval/evaluate.py
```

For a per-row breakdown of misclassifications:

```
python eval/error_analysis.py
```

### Known limitation

The model shows strong precision but weak recall (0.12) on the `Social Media` category — specifically, it systematically misclassifies LinkedIn networking notifications (connection requests, "X accepted your invitation," "you may know") as `Job Search`, likely because it over-weights the sender domain rather than the actual content type. This is documented in `eval/results.md` along with other category-boundary ambiguities found during error analysis (e.g. Facebook notifications overlapping with the `Personal` category, and financial newsletters overlapping with `Promotions`).

## Setup

1. Create a Google Cloud project and enable the Gmail API; download OAuth credentials as `auth/credentials.json` (not included in this repo — see `.gitignore`).
2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```
3. Run `python main.py` — the first run will open a browser for Gmail OAuth consent and cache the token locally (`auth/token.pickle`, also gitignored).
4. Review `config/settings.yaml` — `dry_run: true` by default. Flip to `false` only once you've reviewed the classification behavior.

## Project structure

```
main.py                        # orchestration entrypoint
auth/                          # OAuth flow + Gmail API client
classifier/                    # model loading + classification logic
actions/                       # decide-and-act logic
config/                        # categories.yaml, actions.yaml, settings.yaml
eval/                          # accuracy evaluation pipeline + results.md
```

## Tech stack

Python, Hugging Face Transformers + BitsAndBytes (4-bit quantization), Gmail API, scikit-learn (evaluation metrics).

## Roadmap

- Confidence gating and retry-on-malformed-JSON for more robust agentic behavior
- QLoRA fine-tuning on the evaluation data to improve accuracy on ambiguous categories
- Sharper category descriptions to reduce Social Media / Job Search confusion
