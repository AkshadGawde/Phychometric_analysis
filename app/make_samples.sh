#!/usr/bin/env bash
# Re-record the demo replays (app/replays/) from macOS text-to-speech voices: synthetic, so no
# real person's voice is shipped with the app. Your own recording works the same way:
#   python src/predict.py my-voice.m4a --record my-voice --title "..." --description "..."
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
OOD="Out of distribution: the model was trained on real YouTube speakers and has never heard a synthetic voice, so expect over-confident, extreme scores."

say -v Samantha -r 175 -o "$tmp/samantha.aiff" "Hi everyone, so today I want to tell you about my weekend. I finally went hiking up the trail near my house, and honestly it was amazing. The view from the top was so beautiful, and I think I'm going to go back next week with some friends."
say -v Daniel -r 165 -o "$tmp/daniel.aiff" "Right, so. I've been asked a lot about how I organise my week. I make a list every Sunday evening, and I stick to it. It's not complicated, but it does take discipline, and I think that's what most people are missing."
say -v Rishi -r 170 -o "$tmp/rishi.aiff" "Um, okay, hello. So this is my first video, and I'm a little nervous, to be honest. I wanted to talk about cooking, because I really love trying new recipes. Yesterday I made a curry from scratch, and it came out, well, pretty good actually."

python src/predict.py "$tmp/samantha.aiff" --record tts-samantha --title "Synthetic voice · US English" \
  --description "macOS text-to-speech voice 'Samantha' talking about a hiking weekend. Not a real person. $OOD"
python src/predict.py "$tmp/daniel.aiff" --record tts-daniel --title "Synthetic voice · British English" \
  --description "macOS text-to-speech voice 'Daniel' on planning his week. Not a real person. $OOD"
python src/predict.py "$tmp/rishi.aiff" --record tts-rishi --title "Synthetic voice · Indian English" \
  --description "macOS text-to-speech voice 'Rishi' on his first video and cooking. Not a real person. $OOD"
