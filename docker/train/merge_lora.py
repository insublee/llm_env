import os
import torch
import yaml
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    # Load configuration
    CONFIG_PATH = "/workspace/configs/train.yaml"
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Config file not found at {CONFIG_PATH}")
        sys.exit(1)

    BASE_MODEL = config["model"]["name"]
    LORA_PATH = config["output"]["dir"]
    OUTPUT_PATH = config["output"].get("merged_dir", "/models/merged/my_model")

    # Use template from config or fallback to a default
    CHAT_TEMPLATE = config["model"].get("chat_template", """{% for message in messages %}
{% if message['role'] == 'system' %}
<s>[SYSTEM] {{ message['content'] }}</s>
{% elif message['role'] == 'user' %}
[INST] {{ message['content'] }} [/INST]
{% elif message['role'] == 'assistant' %}
{{ message['content'] }}
{% endif %}
{% endfor %}""")

    if not os.path.exists(f"{LORA_PATH}/adapter_config.json"):
        print(f"❌ LoRA adapter not found at {LORA_PATH}")
        sys.exit(1)

    print(f"📦 Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"🔗 Loading LoRA adapter from {LORA_PATH}...")
    model = PeftModel.from_pretrained(
        model,
        LORA_PATH,
    )

    print("🧬 Merging LoRA into base model...")
    model = model.merge_and_unload()

    print("📝 Setting chat template...")
    tokenizer.chat_template = CHAT_TEMPLATE

    print(f"💾 Saving merged model to {OUTPUT_PATH}...")
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    print("🎉 Merge complete!")

if __name__ == "__main__":
    main()
