"""QLoRA fine-tuning of the classifier on your hand-labeled emails.

What it does, step by step:
  1. Reads labeled emails (default: eval/data/personal_test.csv).
  2. Splits them into train / held-out test (stratified, fixed seed) and saves
     the test part to eval/data/personal_holdout.csv so you can measure the
     fine-tuned model on emails it has NEVER seen.
  3. Turns every training email into (prompt -> correct JSON answer), using the
     exact same prompt as classifier.py.
  4. Loads Qwen3-8B in 4-bit and trains small LoRA adapters on top (QLoRA).
     Only the answer tokens count towards the loss.
  5. Saves the adapter to finetune/adapters/<name>/.

Then set `model.adapter_path` in config/settings.yaml to that folder and run:
    python eval/evaluate.py --only personal --personal-file eval/data/personal_holdout.csv

Usage:
    python finetune/train_qlora.py
    python finetune/train_qlora.py --epochs 3 --extra-data my_more_labels.csv

Needs a CUDA GPU (8B in 4-bit + LoRA fits in roughly 10-12 GB VRAM with the
defaults). Everything it needs is already in requirements.txt.
"""

import argparse
import json
import os
import sys

FINETUNE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(FINETUNE_DIR)
sys.path.insert(0, PROJECT_DIR)

from finetune.data import build_examples, load_labeled, split_train_test  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(PROJECT_DIR, "eval", "data", "personal_test.csv"))
    p.add_argument("--extra-data", nargs="*", default=[], help="more labeled CSVs (train only)")
    p.add_argument("--holdout-out", default=os.path.join(PROJECT_DIR, "eval", "data", "personal_holdout.csv"))
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--output", default=os.path.join(FINETUNE_DIR, "adapters", "qwen3-8b-email"))
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainingArguments)

    from classifier.classifier import get_categories, get_settings

    categories = get_categories()
    model_cfg = get_settings()["model"]

    # 1-2. data + split ------------------------------------------------------
    df = load_labeled(args.data, categories)
    train_df, test_df = split_train_test(df, test_size=args.test_size, seed=args.seed)
    for extra in args.extra_data:
        import pandas as pd
        train_df = pd.concat([train_df, load_labeled(extra, categories)], ignore_index=True)

    test_df.to_csv(args.holdout_out, index=False)
    print(f"Train: {len(train_df)} emails | held-out test: {len(test_df)} emails -> {args.holdout_out}")
    print("Train label counts:\n" + train_df["true_category"].value_counts().to_string())

    # 3. tokenize -----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = build_examples(train_df, tokenizer, categories, max_length=args.max_length)
    print(f"Built {len(examples)} training examples")

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    def collate(batch):
        longest = max(len(b["input_ids"]) for b in batch)
        pad = tokenizer.pad_token_id
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            n = longest - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [pad] * n)
            out["attention_mask"].append([1] * len(b["input_ids"]) + [0] * n)
            out["labels"].append(b["labels"] + [-100] * n)
        return {k: torch.tensor(v) for k, v in out.items()}

    # 4. model + LoRA ---------------------------------------------------------
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="no",
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=ListDataset(examples), data_collator=collate)
    trainer.train()

    # 5. save -----------------------------------------------------------------
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    with open(os.path.join(args.output, "training_info.json"), "w", encoding="utf-8") as f:
        json.dump({
            "base_model": model_cfg["name"],
            "train_rows": len(train_df),
            "holdout_rows": len(test_df),
            "holdout_file": os.path.relpath(args.holdout_out, PROJECT_DIR),
            "categories": list(categories),
            "args": vars(args),
        }, f, indent=2)

    rel = os.path.relpath(args.output, PROJECT_DIR).replace("\\", "/")
    print(f"\nSaved adapter to {args.output}")
    print(f'Next: set  adapter_path: "{rel}"  in config/settings.yaml, then run')
    print(f"  python eval/evaluate.py --only personal --personal-file {os.path.relpath(args.holdout_out, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
