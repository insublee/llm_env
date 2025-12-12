# 🚀 LLM Environment (WSL2 + Docker + GPU + VSCode)

이 프로젝트는 **Windows + WSL2 + Docker + NVIDIA GPU** 환경에서  
LLM 학습 & 추론을 위한 **완전한 개발 환경**을 구축하는 템플릿입니다.

구성된 기능:

- WSL2 기반의 Linux 개발 환경
- Docker Desktop + GPU passthrough
- `docker compose` 기반 멀티 컨테이너 구성  
  - `train` : Unsloth + LoRA 학습 컨테이너  
  - `vllm` : vLLM inference 서버  
  - `api` : FastAPI + LLM API 서버
- HuggingFace CLI + private token 자동 설정
- VSCode WSL 개발환경
- Tiny dataset을 활용한 LLM fine-tuning

---

# 📌 1. Requirements

- Windows 10/11
- NVIDIA GPU (예: RTX 4090)
- WSL2 설치
- Docker Desktop 설치
- VSCode + WSL extension

---

# 📌 2. Install WSL2 (Ubuntu)

PowerShell (관리자):

```bash
wsl --install -d Ubuntu
```
재부팅 후 사용자 생성.

# 📌 3. Docker Desktop 설치 & WSL 통합
Docker Desktop 설치 후:

Settings → Resources → WSL integration
Ubuntu ON

"Enable integration…" 체크

이 옵션은 WSL에서 docker CLI를 사용하도록
Windows Docker Engine을 연결하는 기능입니다.

# 📌 4. GPU Passthrough 설정
WSL에서 확인:

```bash
nvidia-smi
```
정상 출력되면 GPU 연결 성공.

Docker에서 GPU 사용 가능 여부 확인:

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi
```
# 📌 5. 프로젝트 구조
```kotlin
llm_env/
│
├─ docker-compose.yml
├─ train/
│   ├─ Dockerfile
│   ├─ train.py
│   ├─ requirements.txt
│
├─ serve/
│   ├─ Dockerfile
│   ├─ app.py   ← FastAPI
│
├─ vllm/
│   ├─ Dockerfile
│
├─ data/
│   ├─ dataset/    ← HF tiny dataset 저장 위치
│   └─ outputs/    ← 모델 체크포인트 저장
```
# 📌 6. HuggingFace 로그인
Weights 다운로드 / 업로드를 위해 필수:

```bash
huggingface-cli login
```
단, Docker 컨테이너 내부에서도 로그인 필요
(토큰은 환경변수나 volume으로 전달 예정)

# 📌 7. Docker Compose 실행
최초 빌드
```bash
docker compose up --build
```
컨테이너:
train : 학습 컨테이너
vllm : 추론 서버
api : FastAPI 서버

# 📌 8. Training 실행 방법
train 컨테이너 안에서 실행:

```bash
python train.py
```
학습 결과는:
```bash
data/outputs/
```
에 저장됨.

# 📌 9. Inference (vLLM)
서버 자동 실행 후 다음 주소에서 사용 가능:

```bash
http://localhost:8000/generate
```
# 📌 10. FastAPI API 서버
엔드포인트 예시:

```bash
POST /generate
{
  "prompt": "Hello!"
}
```
# 📌 11. HF Tiny Dataset 사용 방법
예시 (1k 샘플):
```bash
from datasets import load_dataset
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train[:1%]")
```
이 데이터는 train.py에서 자동으로 다운로드되거나
로컬 data/dataset/을 volume으로 사용할 수 있습니다.

# 📌 12. VSCode WSL 환경 구성
```bash
code .
```
필수 확장:
Python
Pylance
Docker
YAML
WSL
Dev Containers (optional)

Interpreter 선택:
Ctrl + Shift + P →
Python: Select Interpreter → /usr/bin/python3

# 📌 13. Troubleshooting
❗ WSL에서 docker 명령이 안 되는 경우
```kotlin
The command 'docker' could not be found in this WSL 2 distro.
→ Docker Desktop → Settings → WSL integration → Ubuntu ON
```
❗ GPU가 안 잡히는 경우
```vbnet
docker: Error: no GPU detected
```
NVIDIA 드라이버 업데이트

Docker Desktop - Enable NVIDIA runtime 체크
