# Live pipeline visualizer (ROADMAP_WEB.md). Research pipeline: ./run_all.sh
PY := .venv/bin/python

.PHONY: install bundle dev demo test test-browser samples

install:            ## Python + web dependencies, headless browser for the end-to-end test
	$(PY) -m pip install -r requirements.txt
	cd web && npm install && npx playwright install chromium

bundle:             ## results/models/bundle.json from the trained models (Phase 10)
	$(PY) src/10_export_bundle.py

dev:                ## API :8000 + Vite :5173 with hot reload -> open http://localhost:5173
	./dev.sh

demo: bundle        ## built frontend served by the API -> open http://localhost:8000
	cd web && npm run build
	$(PY) -m uvicorn app.server:app --host 127.0.0.1 --port 8000

test:               ## parity, input and fusion-maths tests (no servers needed)
	$(PY) tests/test_parity.py 20
	$(PY) tests/test_inputs.py
	$(PY) tests/test_fusion_maths.py

test-browser:       ## end-to-end in headless Chromium (needs `make dev` running)
	cd web && node scripts/e2e.mjs

samples:            ## re-record the demo replays from macOS text-to-speech voices
	./app/make_samples.sh
