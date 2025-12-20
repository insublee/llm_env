# 🧠 LLM Fine-tuning & vLLM Serving (Docker 기반)

이 레포는 LoRA 기반 LLM 파인튜닝 → 병합 → vLLM 서빙까지를
Docker + GPU 환경에서 한 번에 실행할 수 있도록 구성되어 있습니다.

✅ GPU 1장 (예: RTX 4090)
✅ Docker Desktop + WSL2
✅ NVIDIA 드라이버 설치 완료
이 3가지만 되어 있으면 됩니다.

# 📁 프로젝트 구조
```
llm_env/
├── docker/
│   ├── train/
│   │   ├── Dockerfile.base
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── train.py
│   │   └── merge_lora.py
│   └── vllm/
│       └── Dockerfile
├── docker-compose.train.yml
├── docker-compose.vllm.yml
├── models/
│   ├── lora/
│   │   └── my_model/        # LoRA 학습 결과
│   └── merged/
│       └── my_model/        # LoRA 병합 완료 모델 (vLLM용)
└── README.md
```
# 1️⃣ 사전 준비 (한 번만)
## 1. Docker & GPU 확인
```
docker --version
nvidia-smi
```

Docker Desktop 설정:
Settings → Resources → Advanced
Docker data location을 **여유 있는 디스크 (예: E:)**로 설정 권장

# 2️⃣ 학습 (LoRA Fine-tuning)
## 2-1. 이전 컨테이너 / 볼륨 정리 (중요)
```
docker compose -f docker-compose.train.yml down -v
rm -rf models/*
```
## 2-2. 학습 실행
```
docker compose -f docker-compose.train.yml up --build
```

정상적으로 돌면 마지막에 다음 로그가 보입니다:
```
🎉 Training done!
llm_train exited with code 0
```
## 2-3. LoRA 결과 확인
```
ls models/lora/my_model
```

아래 파일들이 있으면 정상입니다:

adapter_config.json
adapter_model.safetensors
tokenizer.json
tokenizer.model

# 3️⃣ LoRA → Base 모델 병합 (필수)

vLLM은 LoRA 상태의 모델을 직접 서빙할 수 없습니다.
반드시 merge가 필요합니다.

## 3-1. 병합 실행
```
docker compose -f docker-compose.train.yml run --rm llm_train python merge_lora.py
```

정상 로그:
```
🔗 Loading LoRA adapter...
🧬 Merging LoRA into base model...
💾 Saving merged model...
🎉 Merge complete!
```
## 3-2. 병합 결과 확인
```
ls models/merged/my_model
```

아래 파일들이 있어야 합니다:
```
config.json
model.safetensors (또는 shard 파일들)
tokenizer.json
tokenizer.model
generation_config.json
```

❗ adapter_* 파일이 없어야 정상입니다.

# 4️⃣ vLLM 서빙 실행
## 4-1. vLLM 컨테이너 실행
```
docker compose -f docker-compose.vllm.yml up --build
```

정상 로그 예시:
```
vLLM API server version 0.12.0
Listening on http://0.0.0.0:8000
```

## 4-2. API 테스트
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my_model",
    "prompt": "Explain LoRA fine-tuning in simple terms.",
    "max_tokens": 200
  }'
---

# ⚠️ 자주 발생하는 문제
## ❌ 모델이 저장되지 않는 경우

TrainingArguments.output_dir 와

trainer.save_model() 경로가 컨테이너 기준 경로인지 확인
```
output_dir="/models/lora/my_model"
trainer.save_model("/models/lora/my_model")
```
## ❌ vLLM에서 /models/my_model 에러

config.json 없는 디렉토리를 가리키고 있는 경우

반드시 merged 모델 경로 사용

command: vllm serve /models/merged/my_model
---
# 🎯 전체 파이프라인 요약
```
train.py
  ↓
LoRA adapters 생성
  ↓
merge_lora.py
  ↓
순수 HF 모델 생성
  ↓
vLLM 서빙
```
