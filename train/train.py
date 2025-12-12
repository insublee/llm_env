from datasets import load_dataset
from unsloth import FastLanguageModel

def main():
    print("🚀 Loading dataset...")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:1000]")

    print("🚀 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-v0.3",
        load_in_4bit=True,
    )

    print("🚀 Preparing trainer...")
    trainer = model.get_trainer(
        train_dataset=ds,
        tokenizer=tokenizer,
        dataset_text_field="messages",
        max_seq_length=2048,
        batch_size=1,
    )

    print("🚀 Training...")
    trainer.train()

    print("💾 Saving trained model...")
    model.save_pretrained("/models/my_model")
    tokenizer.save_pretrained("/models/my_model")

    print("🎉 Done training!")

if __name__ == "__main__":
    main()
