#!/usr/bin/env bash
# Runs the whole pipeline, Phase 1 -> Phase 9.
#   ./run_all.sh                                       values from config.py
#   MAX_CLIPS=50 USE_HEAVY_MODELS=0 ./run_all.sh       fast smoke test
#   MAX_CLIPS=2000 USE_HEAVY_MODELS=1 ./run_all.sh     full run
set -e
cd "$(dirname "$0")"
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

python src/01_prepare_data.py
python src/02_preprocess.py
python src/03_features_visual.py
python src/04_features_audio.py
python src/05_features_text.py
python src/06_unimodal.py
python src/07_fusion.py
python src/08_ablation.py
python src/08b_diagnostics.py
python src/09_analysis.py
echo "Done. See results/REPORT.md and results/summary.md"
