FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt update && apt install -y \
    python3 python3-pip python3-venv git wget \
 && apt clean

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY train.py .
