FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt update && apt install -y python3 python3-pip && apt clean
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install vllm

WORKDIR /workspace
