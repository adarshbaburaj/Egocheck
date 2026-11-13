#!/usr/bin/env bash
# One command from "clips recorded" to "demo video rendered".
#
#   scripts/refresh_demo.sh            # pipeline + sims + dashboard recording + video assets + render
#   scripts/refresh_demo.sh --no-render
#
# Video slots (video/demo/assets/):
#   bad_1/2/3.mp4  three rejected clips (first three non-keep clips, by name)
#   dashboard.mp4  scripted dashboard tour (scripts/record_dashboard.py)
#   sim.mp4        MuJoCo replay of the first kept clip
#   shots/*.png    3840x2160 stills (scripts/capture_shots.py), described in shots/index.json
#   live.mp4       record yourself: .venv/bin/python -m judge.live --record video/demo/assets/live.mp4
set -eo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
A=video/demo/assets
export PATH="$PWD/video/bin:$PATH"

echo "== pipeline (OpenCV + Gemini + TypeSafe + sim replays)"
$PY -m judge.pipeline clips/

echo "== video assets"
$PY - <<'EOF'
import json, shutil
r = json.load(open("out/results.json"))
clips = r["clips"]
# one rejected clip per distinct reason (blurry, unfinished, privacy, ...), most confident first
bad, seen = [], set()
for c in sorted((c for c in clips if c["verdict"]["routed_to"] != "keep"), key=lambda c: -c["verdict"]["confidence"]):
    if c["verdict"]["main_reason"] not in seen:
        bad.append(c); seen.add(c["verdict"]["main_reason"])
bad = bad[:3]
for i, c in enumerate(bad, 1):
    shutil.copy(c["src"], f"video/demo/assets/bad_{i}.mp4")
    print(f"bad_{i}.mp4 <- {c['file']} ({c['verdict']['routed_to']}, {c['verdict']['main_reason']})")
kept = [c for c in clips if c.get("sim")]
if kept:
    shutil.copy(kept[0]["sim"]["video"], "video/demo/assets/sim.mp4")
    print("sim.mp4 <-", kept[0]["sim"]["video"])
else:
    print("no kept clip with a sim replay; keeping existing sim.mp4")
EOF

echo "== dashboard recording (1920x1080)"
$PY scripts/record_dashboard.py --out $A/dashboard.mp4

echo "== 4K stills (dashboard light/dark, clip panels, sim phases) -> $A/shots/"
$PY scripts/capture_shots.py --out $A/shots

if [[ "${1:-}" != "--no-render" ]]; then
  echo "== render"
  cd video/demo
  npx -y hyperframes@0.9.42 check
  npx -y hyperframes@0.9.42 render --output ../clip-judge-demo.mp4
  echo "done: video/clip-judge-demo.mp4"
fi
