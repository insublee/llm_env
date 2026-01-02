from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.trainer import SFTTrainer
from transformers import TrainingArguments

def format_chat(example):
    text = ""
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            text += f"<|user|>\n{content}\n"
        elif role == "assistant":
            text += f"<|assistant|>\n{content}\n"
    return {"text": text}

import yaml
import sys

def main():
    # Load configuration
    try:
        with open("/workspace/configs/train.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ Config file not found at /workspace/configs/train.yaml")
        sys.exit(1)

    OUTPUT_DIR = config["output"]["dir"]

    print("🚀 Loading dataset...")
    ds = load_dataset(
        config["dataset"]["name"],
        split=config["dataset"]["split"],
    )
    ds = ds.map(format_chat, remove_columns=ds.column_names)

    print("🚀 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"]["name"],
        load_in_4bit=config["model"]["load_in_4bit"],
    )

    print("🚀 Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        target_modules=config["lora"]["target_modules"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]["bias"],
        use_gradient_checkpointing=True,
        random_state=config["lora"]["random_state"],
    )

    print("🚀 Preparing trainer...")
    training_args = config["training"]
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        max_seq_length=2048,
        gradient_checkpointing = "unsloth",
        args=TrainingArguments(
            per_device_train_batch_size=training_args["per_device_train_batch_size"],
            gradient_accumulation_steps=training_args["gradient_accumulation_steps"],
            warmup_steps=training_args["warmup_steps"],
            max_steps=training_args["max_steps"],
            learning_rate=training_args["learning_rate"],
            bf16=training_args["bf16"],
            logging_steps=training_args["logging_steps"],
            output_dir=OUTPUT_DIR,
            save_steps=training_args["save_steps"],
            save_total_limit=training_args["save_total_limit"],
            report_to=training_args["report_to"],
        ),
    )

    print("🚀 Training...")
    trainer.train()

    print("🎉 Training done!")

    print("💾 Saving LoRA adapter...")
    trainer.save_model(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("✅ LoRA adapter saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
