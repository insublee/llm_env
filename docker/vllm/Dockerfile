FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt update && apt install -y \
    python3 python3-pip \
    git \
 && apt clean \
 && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

RUN pip install --upgrade pip \
 && pip install vllm==0.12.0 \
    transformers \
    accelerate

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
