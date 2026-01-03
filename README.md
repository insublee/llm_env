# 🧠 LLM Fine-tuning & vLLM Serving (Docker + Configurable)

이 레포는 **설정 기반(Config-driven)**으로 디자인된 LLM 파이프라인입니다.
복잡한 코드 수정 없이 **YAML 파일**과 **.env** 파일만으로 학습과 서빙 설정을 제어할 수 있습니다.

## 🚀 주요 변경 사항 (2026 Updated)
- **⚡ Scripts**: 복잡한 docker 명령어를 기억할 필요 없이 `scripts/` 폴더의 스크립트로 실행
- **⚙️ Configs**: `configs/train.yaml`에서 학습 파라미터 제어
- **🌍 Env**: `.env` 파일로 경로 및 포트 설정 관리

## 📁 프로젝트 구조
```
llm_env/
├── configs/             # ⚙️ [NEW] 학습 설정 (YAML)
│   └── train.yaml
├── scripts/             # ⚡ [NEW] 실행 스크립트 (build, train, serve)
├── docker/              # Dockerfile 모음
├── .env                 # 🌍 [NEW] 환경 변수 (경로, 포트 등)
├── docker-compose.yml   # 통합된 docker-compose
└── README.md
```

## 1️⃣ 사전 준비
1. **.env 생성**
   ```bash
   cp .env.example .env
   # .env 파일을 열어서 경로(HF_CACHE_DIR 등)를 본인 환경에 맞게 수정하세요!
   ```

2. **베이스 이미지 빌드 (최초 1회)**
     이 작업은 시간이 꽤 걸립니다 (Pytorch, CUDA 등 설치).
   - **Windows**: `.\scripts\build_base.ps1`
   - **Linux/WSL**: `./scripts/build_base.sh`

## 2️⃣ 학습 (Train)
`configs/train.yaml` 파일을 수정하여 원하는 모델과 파라미터를 설정한 후 실행합니다.

- **Windows**: `.\scripts\run_train.ps1`
- **Linux/WSL**: `./scripts/run_train.sh`

학습이 완료되면 `models/` 폴더에 LoRA 어댑터가 저장됩니다.

## 3️⃣ 서빙 (Serve)
vLLM 엔진과 API 서버를 동시에 실행합니다.

- **Windows**: `.\scripts\run_serve.ps1`
- **Linux/WSL**: `./scripts/run_serve.sh`

### API 테스트
```bash
curl http://localhost:9000/chat?q="Hello! Who are you?"
```

## 🛠️ 고급 설정
### `configs/train.yaml`
```yaml
model:
  name: "unsloth/mistral-7b-v0.3"
  load_in_4bit: true
  chat_template: |
    {% for message in messages %}
    ...
    {% endfor %}

lora:
  r: 16
  lora_alpha: 16
```

### `.env`
```bash
VLLM_PORT=8000
API_PORT=9000
MODEL_NAME=my_model
```
