import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3-8B"

# 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
)

print("Model loaded!")
print("GPU:", torch.cuda.get_device_name(0))

# messages = [
#     {
#         "role": "user",
#         "content": "Explain in one sentence what an email classification system does."
#     }
# ]

# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=False,
# )
# inputs = tokenizer(text, return_tensors="pt").to("cuda")

# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=100,
#     )

# response = tokenizer.decode(
#     outputs[0][inputs["input_ids"].shape[-1]:],
#     skip_special_tokens=True,
# )

# print("\nQwen:")
# print(response)

EMAIL_SUBJECT = "Your Amazon order has shipped"
EMAIL_SENDER = "shipment-tracking@amazon.com"
EMAIL_BODY = "Your package is on its way and will arrive Thursday..."

CATEGORIES = [ "Phishing",
    "Job Search",
    "Applications",
    "Orders",
    "Subscriptions",
    "Personal",
    "Finance",
    "Promotions",
    "Other"]

prompt = f"""You are an email classification agent. Classify the email below into exactly one category.

Categories: {', '.join(CATEGORIES)}

Email:
From: {EMAIL_SENDER}
Subject: {EMAIL_SUBJECT}
Body: {EMAIL_BODY}

Respond ONLY with valid JSON in this exact format, nothing else:
{{"category": "<one of the categories>", "confidence": <float 0 to 1>, "reasoning": "<one short sentence>"}}
"""

messages = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

inputs = tokenizer(text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.1,
    )

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
).strip()

print("\nRaw model output:")
print(response)

try:
    result = json.loads(response)
    print("\nParsed result:")
    print("Category:", result["category"])
    print("Confidence:", result["confidence"])
    print("Reasoning:", result["reasoning"])
except json.JSONDecodeError:
    print("\nModel didn't return valid JSON.")