FROM llm-train-base

WORKDIR /workspace

COPY train.py .
COPY merge_lora.py .

CMD ["python", "train.py"]