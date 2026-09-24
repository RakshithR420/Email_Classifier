"""Email classification with a local, 4-bit quantized LLM.

Flow for one email:
  1. build the prompt (category descriptions + email)       -> build_messages()
  2. run the model                                          -> _generate()
  3. pull the JSON out of the reply and validate it         -> parse_response()
  4. if the reply was broken, tell the model what was wrong
     and ask again (up to `classify.max_retries` times)
  5. if it still fails, fall back to "Other" with confidence 0
"""

import json
import os
import re

import yaml

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")

_model = None
_tokenizer = None
_categories = None
_settings = None

FALLBACK_CATEGORY = "Other"

SYSTEM_PROMPT = "You are an email classification agent. You always answer with a single JSON object and nothing else."

# General rules that fix the confusions found in eval/results.md.
CLASSIFICATION_RULES = """Rules:
- Judge by WHAT the email is about, not only by WHO sent it. One sender (for example LinkedIn) can send Job Search, Social Media and Promotions emails.
- If the email contains a one-time password or verification code, choose OTP.
- If it is the outcome of an application or interview (offer, rejection, "next steps", assessment result), choose Job Result, not Job Search or Applications.
- If it confirms or updates an application the user submitted, choose Applications.
- Only choose Phishing when there are real warning signs (urgent threats, requests for passwords or payment details, mismatched or odd sender domains, suspicious links). Normal marketing is Promotions, not Phishing.
- Use Other only when nothing else fits."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_yaml(filename):
    path = os.path.join(_CONFIG_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_categories():
    global _categories
    if _categories is None:
        _categories = _load_yaml("categories.yaml")
    return _categories


def get_settings():
    global _settings
    if _settings is None:
        _settings = _load_yaml("settings.yaml")
    return _settings


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(email_subject, email_sender, email_body, categories=None, body_limit=None):
    categories = categories or get_categories()
    if body_limit is None:
        body_limit = get_settings().get("fetch", {}).get("body_char_limit", 1000)

    trimmed_body = " ".join((email_body or "").split())[:body_limit]
    category_list_str = "\n".join(f"- {name}: {desc}" for name, desc in categories.items())

    return f"""Classify the email below into exactly one category.

Categories:
{category_list_str}

{CLASSIFICATION_RULES}

Email:
From: {email_sender or "(unknown)"}
Subject: {email_subject or "(no subject)"}
Body: {trimmed_body or "(empty)"}

Respond ONLY with valid JSON in this exact format, nothing else:
{{"category": "<one of the categories>", "confidence": <float 0 to 1>, "reasoning": "<one short sentence>"}}"""


def build_messages(email_subject, email_sender, email_body, categories=None, body_limit=None):
    """Chat messages for one email. Also used by finetune/ so training and
    inference see exactly the same prompt."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(email_subject, email_sender, email_body, categories, body_limit)},
    ]


# ---------------------------------------------------------------------------
# Parsing the model's reply
# ---------------------------------------------------------------------------

def _match_category(raw, categories):
    """Maps the model's category text onto a real category name,
    ignoring case, extra spaces, underscores and dashes."""
    if not isinstance(raw, str):
        return None

    def norm(s):
        return re.sub(r"[\s_\-]+", " ", s).strip().lower()

    wanted = norm(raw)
    for name in categories:
        if norm(name) == wanted:
            return name
    return None


def parse_response(text, categories=None):
    """Extracts {"category", "confidence", "reasoning"} from the model reply.

    Returns (result_dict, None) on success or (None, error_message) on failure.
    Handles <think> blocks, ```json fences and text around the JSON."""
    categories = categories or get_categories()
    if not text or not text.strip():
        return None, "The reply was empty."

    cleaned = re.sub(r"(?s)<think>.*?</think>", "", text)
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()

    data = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to the first {...} block in the text.
        match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None

    if not isinstance(data, dict):
        return None, "The reply was not valid JSON."

    category = _match_category(data.get("category"), categories)
    if category is None:
        return None, f"'{data.get('category')}' is not one of the allowed categories."

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence > 1.0 and confidence <= 100.0:  # model answered in percent
        confidence /= 100.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "category": category,
        "confidence": round(confidence, 3),
        "reasoning": str(data.get("reasoning", "")).strip(),
    }, None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model():
    """Loads the model + tokenizer once and caches them in memory.
    Call this ONE time at startup (e.g. in main.py) — never per email."""
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_cfg = get_settings()["model"]
    use_4bit = model_cfg.get("load_in_4bit", True)
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
        )

    _tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=quantization_config,
        device_map="auto",
    )

    adapter_path = (model_cfg.get("adapter_path") or "").strip()
    if adapter_path:
        from peft import PeftModel

        if not os.path.isabs(adapter_path):
            adapter_path = os.path.join(os.path.dirname(_CONFIG_DIR), adapter_path)
        print(f"Loading fine-tuned adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    _model = model
    return _model, _tokenizer


def _generate(messages, temperature=0.0):
    """Runs the model on chat `messages` and returns the decoded reply."""
    import torch

    model_cfg = get_settings()["model"]
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)

    gen_kwargs = {
        "max_new_tokens": model_cfg.get("max_new_tokens", 150),
        "pad_token_id": _tokenizer.pad_token_id or _tokenizer.eos_token_id,
    }
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)

    with torch.no_grad():
        outputs = _model.generate(**inputs, **gen_kwargs)

    return _tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()


def classify_email(email_subject, email_sender, email_body, generate_fn=None):
    """Classifies an email into one of the categories in categories.yaml.

    Returns a dict with keys: category, confidence, reasoning, attempts, valid.
    `valid` is False when every attempt failed and we fell back to "Other".
    `generate_fn(messages, temperature) -> str` can be passed in for testing."""
    if generate_fn is None:
        if _model is None or _tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded first. Call load_model() at startup.")
        generate_fn = _generate

    categories = get_categories()
    settings = get_settings()
    base_temp = settings.get("model", {}).get("temperature", 0.0) or 0.0
    max_retries = int(settings.get("classify", {}).get("max_retries", 2))

    messages = build_messages(email_subject, email_sender, email_body, categories)
    last_error = None

    for attempt in range(1 + max_retries):
        # First try is deterministic; retries add a little randomness so the
        # model does not just repeat the same broken answer.
        temperature = base_temp if attempt == 0 else max(base_temp, 0.3)
        reply = generate_fn(messages, temperature)
        result, last_error = parse_response(reply, categories)
        if result:
            result.update(attempts=attempt + 1, valid=True)
            return result

        # Show the model its mistake and ask again.
        messages = messages + [
            {"role": "assistant", "content": reply or ""},
            {
                "role": "user",
                "content": f"{last_error} Answer again with ONLY the JSON object. "
                f"The category must be exactly one of: {', '.join(categories)}.",
            },
        ]

    return {
        "category": FALLBACK_CATEGORY,
        "confidence": 0.0,
        "reasoning": f"Model failed after {1 + max_retries} attempts: {last_error}",
        "attempts": 1 + max_retries,
        "valid": False,
    }
