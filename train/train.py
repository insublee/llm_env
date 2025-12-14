from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer

def main():
    print("🚀 Loading dataset...")
    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft[:1000]"
    )

    print("🚀 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/mistral-7b-v0.3",
        load_in_4bit = True,
        max_seq_length = 2048,
    )

    print("🚀 Preparing trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = ds,
        dataset_text_field = "messages",
        max_seq_length = 2048,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        output_dir = "/models/my_model",
        logging_steps = 10,
        save_steps = 100,
    )

    print("🚀 Training...")
    trainer.train()

    print("💾 Saving model...")
    trainer.model.save_pretrained("/models/my_model")
    tokenizer.save_pretrained("/models/my_model")

    print("🎉 Done training!")

if __name__ == "__main__":
    main()
