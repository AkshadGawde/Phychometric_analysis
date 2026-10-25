#!/usr/bin/env bash
# Container entry point (Dockerfile): fetch the trained models once, then serve the API.
# Render sets PORT; Hugging Face Spaces uses 7860; locally 8000.
set -e
cd "$(dirname "$0")/.."
if [ ! -f results/models/bundle.json ]; then
  python app/models_hub.py download
fi
# Every pretrained network was baked into the image at build time: never reach for the network.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec uvicorn app.server:app --host 0.0.0.0 --port "${PORT:-8000}" --log-level warning
