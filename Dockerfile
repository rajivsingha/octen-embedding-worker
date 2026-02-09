FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model during build (avoids slow cold starts)
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
import torch; \
SentenceTransformer('Octen/Octen-Embedding-8B', model_kwargs={'torch_dtype': torch.float16}, trust_remote_code=True)"

COPY handler.py .

CMD ["python", "-u", "handler.py"]