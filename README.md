# LLM Training & Serving Environment (WSL + Docker + GPU)

Windows + WSL2 환경에서
**LLM 학습(Unsloth)과 서빙(vLLM)을 Docker로 분리** 운영하기 위한 개발 환경입니다.

RTX 4090 기준으로 테스트되었으며,
재현 가능한 환경 구성을 목표로 합니다.

## 🧱 Architecture Overview
``` scss
Windows
 └─ WSL2 (Ubuntu)
     └─ Docker Desktop (WSL backend, data on E:)
         ├─ llm-train  (Unsloth + GPU fine-tuning)
         ├─ llm-vllm   (vLLM inference server)
         └─ llm-api    (FastAPI, optional)
```

**Training / Serving 완전 분리**

Docker image rebuild 최소화

대용량 캐시(HuggingFace, pip) → 외부 볼륨 마운트

## 💻 Requirements
### Hardware
- NVIDIA GPU (tested: RTX 4090, 24GB VRAM)
- SSD 권장 (Docker + HF cache)

### Software
- Windows 11
- WSL2 (Ubuntu 22.04)
- Docker Desktop (WSL backend)
- NVIDIA GPU Driver (Windows)
- NVIDIA Container Toolkit (Docker Desktop 포함)

## 📁 Project Structure
``` csharp
llm_env/
├─ docker/
│  └─ train/
│     ├─ Dockerfile.base   # heavy deps (torch, unsloth)
│     ├─ Dockerfile        # lightweight training image
│     └─ requirements.txt
├─ docker-compose.train.yml
├─ train.py
├─ models/                 # trained models output
└─ README.md
```

## 🚀 Training (Unsloth)
### 1️⃣ Build base image (1회만)
``` bash
cd docker/train

DOCKER_BUILDKIT=1 docker build \
  -f Dockerfile.base \
  -t llm-train-base .
```

⚠️ 이 단계는 오래 걸릴 수 있음 (torch, unsloth, triton)

---

### 2️⃣ Run training container
``` bash
cd /mnt/e/llm/llm_env

docker compose -f docker-compose.train.yml up --build
```
- GPU 자동 인식
- HuggingFace cache 외부 마운트
- 모델 출력: ```./models/```

---

### 🧠 Training Details

Model: unsloth/mistral-7b-v0.3
Dataset: HuggingFaceH4/ultrachat_200k (train_sft[:1000])
Precision: bf16
Fine-tuning: LoRA (PEFT)
Gradient checkpointing: unsloth

### 핵심 설정 (train.py)
``` python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=100,
        learning_rate=2e-4,
        bf16=True,
        gradient_checkpointing="unsloth",
        output_dir="/models/my_model",
        save_steps=50,
        save_total_limit=2,
        logging_steps=5,
        report_to="none",
    ),
)
```

## 📦 Volume & Cache Strategy
### docker-compose.train.yml
``` yaml
volumes:
  - ./models:/models
  - /mnt/e/hf_cache:/root/.cache/huggingface

environment:
  - HF_HOME=/root/.cache/huggingface
```
- Docker rebuild 시에도 HF 모델 재다운로드 방지
- SSD(E:) 사용 권장

## 🧯 Known Pitfalls & Fixes
❌ ```cannot find -lcuda```

✔ 해결:
nvidia/cuda:*runtime* ❌
nvidia/cuda:*devel* ✅
libcuda.so WSL symlink 필요
``` dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
RUN ln -s /usr/lib/wsl/lib/libcuda.so /usr/lib/libcuda.so || true
```
---
❌ **Unsloth dependency conflict**

✔ 해결:
- torch, trl 버전 직접 고정하지 말 것
- unsloth가 요구하는 버전에 맡기기
---
❌ **Quantized model cannot be fine-tuned**

✔ 해결:
- LoRA adapters 반드시 활성화
- pure 4bit 모델 단독 학습 ❌
---

## ✅ Current Status

- DONE WSL + Docker + GPU 정상 인식
- DONE  Unsloth fine-tuning 성공
- DONE 모델 저장 확인 (/models)
- TODO vLLM 서빙 연결
- TODO HuggingFace 자동 업로드
- TODO FastAPI 인증 / 로그

## 🔜 Next Steps

1. vLLM 컨테이너에서 /models 로컬 모델 로딩
2. OpenAI-compatible API 테스트
3. HuggingFace Hub 자동 push
4. 실사용용 config 분리 (dev / prod)
---

**🧠 Notes**

이 레포는 **“한 번 세팅하면 다시 안 깨지는 LLM 실험 환경”**을 목표로 합니다.
Windows + GPU + Docker + LLM 조합에서 삽질을 줄이기 위한 기록입니다.
