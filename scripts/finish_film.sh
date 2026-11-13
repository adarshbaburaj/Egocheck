#!/usr/bin/env bash
# Render the story film and apply pacing: everything before the end card at 1.2x, the end card at 1.0x
# (so the tagline lands slowly). Usage: scripts/finish_film.sh <out.mp4>
set -eo pipefail
cd "$(dirname "$0")/../video/egocheck"
export PATH="$PWD/../bin:$PATH"
OUT="${1:-../egocheck-film.mp4}"
END_CARD=$(grep -o 'id="s13"[^>]*data-start="[0-9.]*"' index.html | grep -o '[0-9.]*"$' | tr -d '"')
SPEED=1.2
npx hyperframes@0.9.42 check
npx hyperframes@0.9.42 render -o ../draft.mp4
ffmpeg -y -v error -i ../draft.mp4 -filter_complex "\
[0:v]trim=0:${END_CARD},setpts=(PTS-STARTPTS)/${SPEED}[v1];[0:v]trim=${END_CARD},setpts=PTS-STARTPTS[v2];\
[0:a]atrim=0:${END_CARD},asetpts=PTS-STARTPTS,atempo=${SPEED}[a1];[0:a]atrim=${END_CARD},asetpts=PTS-STARTPTS[a2];\
[v1][a1][v2][a2]concat=n=2:v=1:a=1[v][a]" -map "[v]" -map "[a]" -r 30 -c:v libx264 -crf 16 -preset slow \
  -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart "$OUT"
rm ../draft.mp4
ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT"
