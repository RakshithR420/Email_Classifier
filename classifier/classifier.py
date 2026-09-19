
import os
import json

import torch
import yaml

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")

_model = None
_tokenizer = None
_categories = None
_settings = None 

def _load_yaml(filename):
    path = os.path.join(_CONFIG_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
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

def load_model():
    """Loads the model + tokenizer once and caches them in memory.
    Call this ONE time at startup (e.g. in main.py) — never per email."""
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    model_cfg = get_settings()["model"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit= model_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type= model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype= torch.float16,
        bnb_4bit_use_double_quant= model_cfg.get("bnb_4bit_use_double_quant", True)
    )
    
    _tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    _model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    return _model, _tokenizer

def classify_email(email_subject, email_sender, email_body):
    """Classifies an email into one of the categories defined in categories.yaml.
    Returns a dict with keys: category, confidence, reasoning."""
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model and tokenizer must be loaded first. Call load_model() at startup.")
    
    categories = get_categories()
    model_cfg = get_settings()["model"]
    body_limit = get_settings().get("fetch",{}).get("body_char_limit", 1000)
    trimmed_body = (email_body or "")[:body_limit]
    
    category_list_str = "\n".join(f"- {name}: {desc}" for name, desc in categories.items())
    
    prompt = f"""You are an email classification agent. Classify the email below into exactly one category.

Categories:
{category_list_str}

Email:
From: {email_sender}
Subject: {email_subject}
Body: {trimmed_body}

Respond ONLY with valid JSON in this exact format, nothing else:
{{"category": "<one of the categories>", "confidence": <float 0 to 1>, "reasoning": "<one short sentence>"}}
    """
    
    messages = [{"role": "user", "content": prompt}]
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = _tokenizer(text , return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = _model.generate(**inputs, max_new_tokens=150, temperature=model_cfg.get("temperature", 0.1))  
    
    response = _tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    
    try: 
        result = json.loads(response)
    except json.JSONDecodeError:
        result = None
    if not result or result.get("category") not in categories:
        result = {"category": "Other", "confidence": 0.0, "reasoning": "Invalid or unrecognized response from model."}
    
    return result
    