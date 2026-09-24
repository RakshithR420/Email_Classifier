# Agentic Email Classifier

An agent that reads your Gmail inbox, classifies each email into a category using a locally-hosted, 4-bit quantized LLM (Qwen3-8B), and takes a configurable action (label or archive) via the Gmail API — no cloud LLM API calls, no data leaving your machine.

## Why this project

Most "email classifier" projects call a hosted LLM API and stop there. This one runs the model entirely locally (4-bit quantized to fit consumer GPU VRAM), integrates directly with the Gmail API to actually act on the inbox (not just print a label), and includes a real accuracy evaluation pipeline against both a public benchmark and the user's own hand-labeled inbox data — with documented, honest results including known failure modes.

## How it works

1. **Authenticate** with Gmail via OAuth2 (`auth/gmail_auth.py`). Expired or revoked tokens trigger a fresh consent flow automatically.
2. **Fetch** inbox emails via the Gmail API, with pagination and rate-limit-aware retry/backoff, and parse nested MIME emails (plain text preferred, HTML stripped) (`auth/gmail_client.py`).
3. **Classify** each email by prompting a locally-loaded, 4-bit quantized Qwen3-8B model with the email plus category descriptions and disambiguation rules, requesting structured JSON (`classifier/classifier.py`).
   - **Retry on malformed output**: if the reply is not valid JSON or names an unknown category, the model is shown its mistake and asked again (`classify.max_retries`). JSON wrapped in code fences or extra text is still recovered.
   - If every attempt fails, the email falls back to `Other` with `valid=False`, and the agent does not act on it.
4. **Decide & act** (`actions/actions.py`, `config/actions.yaml`):
   - **Confidence gating**: below `classify.confidence_threshold` the category's action is *not* performed. The email only gets the `AI/Needs-Review` label, so nothing is archived on a guess.
   - Otherwise apply the category's action: `label`, `archive` (label + remove from inbox) or `none`.
   - Handled emails get an `AI/Processed` label and the default query skips them, so re-running never processes the same email twice.
5. **Log** every decision to `logs/run_<timestamp>.csv` and print a summary.

Everything is orchestrated by `main.py`. It runs in **dry-run mode by default** (`run.dry_run: true` in `config/settings.yaml`): it prints what it *would* do and changes nothing in Gmail.

## Usage

```
python main.py                     # dry run with settings.yaml
python main.py --max 10            # only 10 emails
python main.py --live              # actually label / archive in Gmail
python main.py --query "in:inbox newer_than:2d"
```

Key settings in `config/settings.yaml`:

| Setting | What it does |
|---|---|
| `fetch.query` / `fetch.max_results` | Which emails to look at |
| `fetch.body_char_limit` | How much of the body the model sees |
| `classify.max_retries` | Extra attempts on malformed output |
| `classify.confidence_threshold` | Minimum confidence before acting |
| `run.dry_run` | `true` = change nothing (override with `--live`) |
| `model.adapter_path` | Load a fine-tuned QLoRA adapter |

## Categories

Defined in `config/categories.yaml` and fully configurable — currently includes Phishing, Job Search, Applications, Orders, Subscriptions, Personal, Finance, Promotions, Other, Job Result, Social Media, and OTP. Each category maps to an action (`label`, `archive`, or `none`) in `config/actions.yaml`.

## Evaluation

This project includes a real accuracy evaluation, not just anecdotal testing — see `eval/results.md` for full results. Two benchmarks are used:

- **Phishing detection**: a balanced 150-email sample from a public Kaggle phishing dataset (`eval/prepare_phishing_data.py`). Latest result: **83% accuracy** (96% precision / 68% recall on the Phishing class — the model is conservative, rarely false-flagging but missing some real phishing).
- **Personal inbox classification**: a hand-reviewed sample of the user's own inbox (`eval/build_personal_template.py` fetches emails; labels are manually verified before evaluation). Latest result: **77% accuracy** on 200 emails.

Run the evaluation yourself with:

```
python eval/evaluate.py                    # both benchmarks
python eval/evaluate.py --only personal    # just your inbox
```

It prints a classification report and a confusion matrix for each benchmark and saves per-row predictions to `eval/data/`. For a breakdown of the misclassifications (most common true→predicted mix-ups first):

```
python eval/error_analysis.py
```

> The numbers above were measured **before** the sharper category descriptions, disambiguation rules and retry logic were added. Re-run `eval/evaluate.py` and add a "Run 3" section to `eval/results.md` to measure the improvement.

### Known limitation

The model shows strong precision but weak recall (0.12) on the `Social Media` category — specifically, it systematically misclassifies LinkedIn networking notifications (connection requests, "X accepted your invitation," "you may know") as `Job Search`, likely because it over-weights the sender domain rather than the actual content type. This is documented in `eval/results.md` along with other category-boundary ambiguities found during error analysis (e.g. Facebook notifications overlapping with the `Personal` category, and financial newsletters overlapping with `Promotions`).

The category descriptions in `config/categories.yaml` and the rules in the prompt now address these directly: classify by *content* rather than sender, LinkedIn networking notifications are `Social Media`, and financial marketing is `Promotions`. Whether this closes the gap still needs to be measured.

## Fine-tuning (QLoRA)

`finetune/train_qlora.py` fine-tunes small LoRA adapters on top of the 4-bit base model using your hand-labeled emails:

1. Splits `eval/data/personal_test.csv` into train and a **held-out test set** (stratified, saved to `eval/data/personal_holdout.csv`), so the fine-tuned model is always measured on emails it never saw.
2. Builds training examples with the **same prompt** the classifier uses at inference time; only the JSON answer tokens are trained.
3. Trains with QLoRA (4-bit NF4 base + LoRA r=16 on all attention/MLP projections) and saves the adapter to `finetune/adapters/qwen3-8b-email/`.

```
python finetune/train_qlora.py                 # needs a CUDA GPU (~10-12 GB VRAM)
# then set  model.adapter_path: "finetune/adapters/qwen3-8b-email"  in settings.yaml
python eval/evaluate.py --only personal --personal-file eval/data/personal_holdout.csv
```

To compare fairly, evaluate the base model on the same holdout file first (with `adapter_path: ""`).

**No local GPU?** Open `notebooks/colab_eval_finetune.ipynb` in Google Colab (free T4 GPU). It runs the full evaluation, the QLoRA training and the base-vs-fine-tuned comparison, then downloads the results.

Note: training targets use a fixed confidence of 0.9, so after fine-tuning the model's confidence is less informative. Invalid answers are still gated, but consider a lower `confidence_threshold` for the fine-tuned model.

## Setup

1. Create a Google Cloud project and enable the Gmail API; download OAuth credentials as `auth/credentials.json` (not included in this repo — see `.gitignore`).
2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```
3. Run `python main.py` — the first run will open a browser for Gmail OAuth consent and cache the token locally (`auth/token.pickle`, also gitignored).
4. Review the dry-run output and the CSV in `logs/`. When you're happy, run `python main.py --live` (or set `run.dry_run: false`).

## Tests

Unit tests cover MIME parsing, pagination, JSON parsing/retry, confidence gating, dry-run safety and fine-tuning data prep. They run without a GPU or Gmail access:

```
pytest
```

## Project structure

```
main.py                        # orchestration entrypoint (CLI)
auth/                          # OAuth flow + Gmail API client
classifier/                    # model loading, prompt, JSON parsing + retry
actions/                       # confidence gating + decide-and-act logic
config/                        # categories.yaml, actions.yaml, settings.yaml
eval/                          # accuracy evaluation pipeline + results.md
finetune/                      # QLoRA fine-tuning on labeled emails
tests/                         # unit tests (pytest)
logs/                          # per-run decision logs (gitignored)
```

## Tech stack

Python, Hugging Face Transformers + BitsAndBytes (4-bit quantization), PEFT (QLoRA), Gmail API, scikit-learn (evaluation metrics), pytest.

## Roadmap

- [x] Confidence gating and retry-on-malformed-JSON for more robust agentic behavior
- [x] QLoRA fine-tuning on the evaluation data to improve accuracy on ambiguous categories
- [x] Sharper category descriptions to reduce Social Media / Job Search confusion
- [ ] Re-run the evaluation (base vs. fine-tuned on the held-out set) and record it as Run 3 in `eval/results.md`
- [ ] Schedule `main.py --live` to run automatically (e.g. Windows Task Scheduler)
