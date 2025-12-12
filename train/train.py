from datasets import load_dataset
from unsloth import FastLanguageModel

def main():
    print("🚀 Loading dataset...")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train[:1000]")

    print("🚀 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-bnb-4bit",
        max_seq_length=2048,
    )

    print("🚀 Training (tiny demo)...")
    model.train(
        dataset=ds,
        text_column="text",
        output_dir="/workspace/models/my_model",
        epochs=1,
        lr=2e-5,
        batch_size=1,
    )

    print("🎉 Training complete!")

if __name__ == "__main__":
    main()
