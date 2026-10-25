# The API of the pipeline visualizer (app/server.py), CPU only. See DEPLOY.md.
#   docker build -t vip-api .
#   docker run -p 8000:8000 -e MODELS_REPO=you/vip-models -e HF_TOKEN=hf_... vip-api
# Needs ~3 GB of RAM at runtime (wav2vec2-large, ViT, ResNet-50, two text models, Whisper).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface TORCH_HOME=/app/.cache/torch \
    NUMBA_CACHE_DIR=/tmp/numba MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-server.txt .
RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-server.txt

COPY config.py ./
COPY src ./src
COPY app ./app
# Bake the public pretrained networks into the image (no downloads when the server starts).
RUN python app/models_hub.py pretrained

EXPOSE 8000
CMD ["./app/start.sh"]
