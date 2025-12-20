import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "unsloth/mistral-7b-v0.3"
LORA_PATH = "/models/lora/my_model"
OUTPUT_PATH = "/models/merged/my_model"

assert os.path.exists(f"{LORA_PATH}/adapter_config.json"), "❌ LoRA adapter not found"

print("📦 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("🔗 Loading LoRA adapter via PeftModel...")
model = PeftModel.from_pretrained(
    model,
    LORA_PATH,
)

print("🧬 Merging LoRA into base model...")
model = model.merge_and_unload()

print("💾 Saving merged model...")
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print("🎉 Merge complete!")
