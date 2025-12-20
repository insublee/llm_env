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

def main():
    OUTPUT_DIR = "/models/lora/my_model"

    print("🚀 Loading dataset...")
    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft[:1000]",
    )
    ds = ds.map(format_chat, remove_columns=ds.column_names)

    print("🚀 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-v0.3",
        load_in_4bit=True,
    )

    print("🚀 Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )

    print("🚀 Preparing trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        max_seq_length=2048,
        gradient_checkpointing = "unsloth",
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=1,
            max_steps=10,
            learning_rate=2e-4,
            bf16=True,
            logging_steps=5,
            output_dir=OUTPUT_DIR,
            save_steps=5,
            save_total_limit=2,
            report_to="none",
            
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
