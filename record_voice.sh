#!/usr/bin/env bash
# Record your voice from the Mac's microphone and run the trained pipeline on it.
#
#   ./record_voice.sh                    15 s (as long as the training clips), then analyse
#   ./record_voice.sh -t 25              record 25 s (5-180 s)
#   ./record_voice.sh -n                 only record, do not analyse
#   ./record_voice.sh -r me -T "My voice"   also save it as a demo sample for the web app
#   ./record_voice.sh -l                 list microphones;  -d 1  record from microphone [1]
#   ./record_voice.sh -o take1.wav       choose the output file
#
# The recording is saved as 16 kHz mono wav in recordings/ (git-ignored: it is your voice).
# The first time, macOS asks to allow the terminal (or VS Code) to use the microphone.
set -e
cd "$(dirname "$0")"

SECONDS_TO_RECORD=15
DEVICE=0
OUT=""
ANALYSE=1
REPLAY=""
TITLE=""

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while getopts "t:d:o:nr:T:lh" opt; do
  case $opt in
    t) SECONDS_TO_RECORD=$OPTARG ;;
    d) DEVICE=$OPTARG ;;
    o) OUT=$OPTARG ;;
    n) ANALYSE=0 ;;
    r) REPLAY=$OPTARG ;;
    T) TITLE=$OPTARG ;;
    l) ffmpeg -hide_banner -f avfoundation -list_devices true -i "" 2>&1 \
         | sed -n '/audio devices/,$p' | grep -E '^\[AVFoundation' | sed 's/^\[AVFoundation[^]]*\] //'
       exit 0 ;;
    h) usage ;;
    *) usage 1 ;;
  esac
done

if ! [[ $SECONDS_TO_RECORD =~ ^[0-9]+$ ]] || [ "$SECONDS_TO_RECORD" -lt 5 ] || [ "$SECONDS_TO_RECORD" -gt 180 ]; then
  echo "Duration must be 5-180 seconds (got '$SECONDS_TO_RECORD')." >&2; exit 1
fi
command -v ffmpeg >/dev/null || { echo "ffmpeg not found: brew install ffmpeg" >&2; exit 1; }

mkdir -p recordings
[ -n "$OUT" ] || OUT="recordings/voice-$(date +%Y%m%d-%H%M%S).wav"

echo "Microphone [$DEVICE], $SECONDS_TO_RECORD s -> $OUT"
echo "Talk about anything: your weekend, a hobby, what you had for lunch."
for n in 3 2 1; do printf "\rStarting in %s… " "$n"; sleep 1; done
printf "\r● Recording — speak now (Ctrl+C stops early)          \n"

# avfoundation ":N" = audio device N, no video. 16 kHz mono = what the pipeline uses anyway.
# Ctrl+C ends the recording cleanly (ffmpeg finalises the file) instead of killing this script.
trap ':' INT
ffmpeg -hide_banner -loglevel error -stats -f avfoundation -i ":$DEVICE" \
  -t "$SECONDS_TO_RECORD" -ac 1 -ar 16000 -y "$OUT" || true
trap - INT

if [ ! -s "$OUT" ]; then
  echo "Nothing was recorded. Is the microphone allowed? System Settings → Privacy & Security → Microphone." >&2
  exit 1
fi

# Quick level check: a muted / blocked microphone records pure silence.
LEVEL=$(ffmpeg -hide_banner -i "$OUT" -af volumedetect -f null - 2>&1 | sed -n 's/.*max_volume: \(-*[0-9.]*\) dB.*/\1/p')
LENGTH=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$OUT")
printf "Saved %s (%.1f s, loudest point %s dB)\n" "$OUT" "$LENGTH" "${LEVEL:-?}"
if [ -n "$LEVEL" ] && awk "BEGIN { exit !($LEVEL < -50) }"; then
  echo "Warning: the recording is almost silent - check the microphone permission or pick another device (-l, -d)." >&2
fi

[ "$ANALYSE" = "1" ] || exit 0

echo
echo "Running the pipeline (loads the models first: about a minute)…"
source .venv/bin/activate
ARGS=("$OUT" --json "${OUT%.*}.json")
if [ -n "$REPLAY" ]; then
  ARGS+=(--record "$REPLAY" --title "${TITLE:-$REPLAY}" --description "Recorded with record_voice.sh on $(date +%Y-%m-%d).")
fi
python src/predict.py "${ARGS[@]}" 2>&1 | grep -vE "Loading weights|objc\[|MallocStackLogging"
echo
echo "Every stage's output: ${OUT%.*}.json"
[ -z "$REPLAY" ] || echo "Demo sample '$REPLAY' saved: pick it under 'Demo samples' at http://localhost:5173"
