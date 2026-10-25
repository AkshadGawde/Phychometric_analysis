#!/usr/bin/env bash
# Development mode of the pipeline visualizer (ROADMAP_WEB.md):
#   API     uvicorn on http://127.0.0.1:8000  (loads + warms up the models: ~1 min, ~3 GB RAM)
#   web     Vite on   http://localhost:5173   (hot reload; /api is proxied to the API)  <- open this
#
#   ./dev.sh               both servers, Ctrl+C stops both
#   RELOAD=1 ./dev.sh      also restart the API when app/ or src/ changes (reloads the models)
#   WHISPER_MODEL=base.en ./dev.sh   faster, less accurate speech recognition
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

[ -d web/node_modules ] || (cd web && npm install)
[ -f results/models/bundle.json ] || python src/10_export_bundle.py

RELOAD_ARGS=()
if [ "${RELOAD:-0}" = "1" ]; then RELOAD_ARGS=(--reload --reload-dir app --reload-dir src); fi

trap 'kill 0' EXIT INT TERM
uvicorn app.server:app --host 127.0.0.1 --port 8000 "${RELOAD_ARGS[@]}" &
(cd web && npx vite --port 5173 --strictPort) &
echo "Open http://localhost:5173  (the status dot turns green when the models are ready)"
wait
