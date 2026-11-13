# Clip Judge (egocheck)

**A quality gate for egocentric robot-training video.** Every clip is measured, watched,
judged and documented before it reaches a robot. Each one is routed to **keep**, **fix** or
**discard**. If the system isn't sure, a **human** decides.

> Robots learn manipulation from first-person video of human hands. Much of that video is
> unusable: it's blurry, dark or shaky, the hands leave the frame, the object is hidden, or
> the task is never finished. Data vendors estimate 15–30% is rejected. Today that junk is
> found after training, or by people watching hours of footage. Clip Judge checks every clip
> in about a second for about $0.003, and writes a data card for each one.

Built at the AIE × MLH Code Summit hackathon (13 Nov 2026) with **TypeSafe Jev (System One)**,
**Gemini 4 Hydrogen**, **MuJoCo**, **Sentry** and **HyperFrames**.

---

## How it works

```
clip.mp4
  │
  ├─ 1. Measure   OpenCV (local, free)        blur · brightness · dark frames · shake
  ├─ 2. Watch     Gemini 4 Hydrogen (Vertex)  task label · hands/object visible · completed?
  │                                           single task? · faces/screens? · step timeline
  ├─ 3. Judge     TypeSafe Jev (System One)   7 yes/no questions on 3 keyframes → probabilities
  ├─ 4. Decide    plain code (truth table)    keep / fix / discard + main reason + probabilities
  ├─ 5. Route     top probability ≥ 0.80 → follow it, else → human_review
  ├─ 6. Document  out/cards/<clip>.md|json · out/report.md · out/results.json
  └─ 7. Replay    kept clips → MuJoCo Shadow Hand replays the steps → out/sim/<clip>.mp4
```

**Why split "judge" and "decide".** We follow TypeSafe's guidance: the model makes small,
observable judgments ("Are the hands in view in `frames.middle`?"), and code makes the
decision. That keeps the decision visible, testable and easy to tune.

### The questions Jev answers ([judge/system_one.py](judge/system_one.py))

| Failure mode | Question (one Noul each) | Fixable? |
|---|---|---|
| blurry | Is `frames.middle` too blurry to make out the edges of the object in the person's hand? | no |
| dark | Is `frames.middle` too dark to tell what the hands are holding? | yes (relight) |
| hands_out_of_frame | Are the person's hands inside the picture in `frames.middle`? | no |
| object_hidden | Is the object named in `task` clearly visible in `frames.middle`? | no |
| task_failed | Does `frames.end` show the task in `task` finished? | no |
| multiple_tasks | Do `frames.start` and `frames.end` show the person working on two different objects? | yes (trim) |
| privacy | Is a human face or a readable screen visible in any of `frames`? | yes (blur) |
| shaky | measured locally (optical flow), not asked | yes (stabilise) |

Measured blur and darkness are combined with Jev's answers, and the stronger signal wins.

### The decision ([judge/system_one.py](judge/system_one.py) `combine()`)

Each failure probability falls into one of three bands, which TypeSafe recommends:
- **pass:** below 0.3
- **unsettled:** between 0.3 and 0.7
- **fail:** above 0.7

| Situation | Verdict | Confidence |
|---|---|---|
| any **unfixable** failure | discard | that failure's probability |
| only **fixable** failures | fix | that failure's probability |
| something **unsettled** | the leaning verdict | ≤ 0.7, so it always routes to **human_review** |
| everything passes | keep | 1 − (weakest pass) |

The main reason is the failure mode with the highest probability, or "good" if there are none.

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then fill in keys (see below)
scripts/fetch_menagerie.sh      # MuJoCo Shadow Hand model for the sim

python -m judge.pipeline clips/            # judge every clip, write out/
python -m http.server 8000                 # then open http://localhost:8000/dashboard/
python -m judge.live                       # live webcam mode (q to quit)
python -m sim.replay <clip>                # render a MuJoCo replay → out/sim/<clip>.mp4
pytest -q                                  # 13 tests, offline
```

### Keys (`.env`, git-ignored)

| Variable | What for |
|---|---|
| `TYPESAFE_API_KEY` | TypeSafe Jev (System One). Without it, a labelled rule-based stand-in decides |
| `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION=global` | Gemini through Vertex AI, billed to Google Cloud. Sign in once with `gcloud auth application-default login` |
| `GEMINI_MODEL` | default `gemini-4-hydrogen` |
| `GEMINI_API_KEY` (`_1`, `_2`) | alternative: AI Studio keys. The first one that works is used |
| `SENTRY_DSN` | errors, tracing, logs, Gemini call monitoring |

Without a Gemini setup, fact sheets are skipped and verdicts use the local checks plus Jev.

### Labels

`labels.csv` (`clip,my_verdict`) holds your own verdicts: keep / fix / discard, or `bad`,
meaning anything but keep. **If there's no labels.csv**, the filename decides: a clip with
`good` in its name is labelled keep, and every other clip is labelled bad. These labels feed
the "agreement with your labels" number.

---

## Outputs

| Path | What |
|---|---|
| `out/results.json` | everything: summary + every clip (local checks, fact sheet, verdict, probabilities, checks, sim) |
| `out/cards/<clip>.md` / `.json` | data card per clip: verdict, probabilities, recording metrics, fact sheet, steps, sha256 |
| `out/report.md` | dataset report: routes, usable minutes, reject reasons, privacy flags, review queue, failures |
| `out/sim/<clip>.mp4` | MuJoCo replay for each kept clip (1920×1080) |
| `out/cache/` | API responses + keyframes keyed by file sha256, so re-runs are instant |

## Dashboard ([dashboard/index.html](dashboard/index.html))

A single static page with no build step. It reads `../out/results.json`.
- **Summary:** a sentence ("N of M clips are ready for training"), a route bar, and a legend that doubles as the filter.
- **Clip views:** grid and list.
- **Clip panel:**
  - verdict in plain words, with a probability bar for each option and the 0.80 line marked
  - video with the step timeline
  - robot replay
  - ✓ / ! / ✗ checks list
  - notes and a link to the data card
- **Theme and motion:** light and dark mode, view-transition animations, and respects reduced motion.

## Live mode ([judge/live.py](judge/live.py))

`python -m judge.live [--record out.mp4] [--window 3]` works like this:
1. Every 3 s, the last frames go through the same local checks.
2. Three frames go to Jev, which answers in about 0.3–0.8 s, in a background thread.
3. The overlay shows the route and the 7 checks.

Gemini is batch-only, at about 8 s per clip. On macOS, allow camera access for your terminal or VS Code.

## Sim ([sim/](sim/))

- **Model:** the Shadow Hand (DeepMind MuJoCo Menagerie, `sim/menagerie/shadow_hand`) on a kinematic forearm, with a table, an apple and a plate.
- **Steps:** the step order and timing come from the clip's Gemini fact sheet (move_to → grasp → lift → place → release).
- **Side grasp:** the hand turns on its side (fingers toward the plate, palm facing the apple), slides in, wraps, lifts, carries and sets the apple down. The orientation was found by searching axis-aligned rotations. The palm offset came from a physics grid search: no table contact, and the apple moves less than 1.5 cm when the fingers close. See `SIDE_*` in `sim/replay.py`. Top-down mode is `Replayer(steps, side=False)`.
- **Honest limits:**
  - **Fixed layout:** the apple and plate positions don't come from the video.
  - **Assisted grasp:** once the fingers close, the apple is locked to the palm.
  - **It's a replay, not a learned policy.**
- **Physics and speed:** fingers are physical position actuators, and rendering runs at about 2× real time on a MacBook.

## Monitoring (Sentry, project `egocheck`)

- **What's captured:** errors (a failed clip is logged and skipped, never crashes the run), a trace for each run with a span for each step, logs, Gemini calls, and a count for each route and reason.
- **Sampling:** trace sample rate 1.0 in development and 0.1 in production (`SENTRY_ENVIRONMENT`).

---

## Numbers (measured 13 Nov 2026 unless marked)

| | Value |
|---|---|
| Jev: 7 questions on 3 frames | **0.15–0.43 s** per clip, **2,706 input tokens** |
| Jev cost | **$0.000114** per clip ($0.042 per 1M input tokens, output free) |
| Gemini 4 Hydrogen fact sheet | **≈ 8.5 s** per clip (Vertex) |
| Gemini cost | ≈ $0.003 per clip *(estimated from published pricing, not measured)* |
| All-in | ≈ **$1 per hour** of video checked |
| Sim replay render | ≈ 2× real time at 1080p |

**Market context.** These are vendor estimates, not audited:
- raw egocentric capture costs $15–40 per hour (DexSet, DataXpower)
- 15–30% is typically rejected
- packaged datasets sell for $90–1,800 per hour

**Worked example (assumptions):**
- Buying 1,000 h at $30/h costs $30,000. At 20% junk, about $6,000 of it is unusable.
- Checking it by hand at $25/h costs $25,000 or more.
- Clip Judge checks it for about $1,000, which is ≈ 3% of the data cost.

**Not yet measured.** Accuracy against human labels comes only from our own handful of
recorded clips. That's a demo, not a benchmark.

---

## Demo video (HyperFrames) — [video/](video/)

```bash
scripts/refresh_demo.sh      # pipeline → sims → dashboard recording → 4K stills → render
```

| Asset (`video/demo/assets/`) | Made by |
|---|---|
| `bad_1..3.mp4` | the first three non-keep clips, copied by the refresh script |
| `dashboard.mp4` | `scripts/record_dashboard.py`: a scripted tour in headless Chromium, 1920×1080 |
| `shots/*.png` + `shots/index.json` | `scripts/capture_shots.py`: 3840×2160 dashboard states (light and dark), clip panels, sim phases |
| `sim.mp4` | the first kept clip's MuJoCo replay |
| `live.mp4` | you: `python -m judge.live --record video/demo/assets/live.mp4` |

- **Composition:** `video/demo/index.html`, which uses GSAP timing as in the HyperFrames contract.
- **Check, preview and render:** `cd video/demo && PATH=../bin:$PATH npx hyperframes@0.9.42 check|preview|render`.
- **FFmpeg / ffprobe:** they live in `video/bin/` (installed through npm under `video/tools/`; no Homebrew needed).

---

### Story film — [video/egocheck/](video/egocheck/)

The 66 s story film: Elon raises $500K, buys data, the dream pops, garbage in / garbage out, then EgoCheck, how it works, System One's speed, the savings, and the robot. Output: `video/egocheck-film-v9.mp4`.

- **Narration:** ElevenLabs, built-in "Jessica" voice (`ELEVENLABS_API_KEY` in `.env`). Run `.venv/bin/python scripts/make_voice.py` to write `assets/vo/l01..l15.mp3`. Set `VO_VOICE=<id>` to use another voice; library voices need a paid plan.
- **Music:** none under the story half. Music enters with EgoCheck: a Lyria-generated theme starts at the logo reveal and runs under the product scenes, then a short hype hit plays under the end card. Both come from Google **Lyria** (`lyria-002`) on Vertex AI, using the same project and credits as Gemini. Run `.venv/bin/python scripts/make_music.py`. To use your own track (for example from the YouTube Audio Library), save it as `assets/music/theme.wav` and rerun the script.
- **SFX:** HyperFrames bundled library, plus two Openverse CC BY sounds (see `assets/sfx/CREDITS-openverse.md`).
- **"Garbage out" sim:** `.venv/bin/python -m sim.chaos` renders `out/sim/chaos.mp4`. A Shadow Hand "trained on bad data" twitches, grabs the apple and throws it at the camera. The throw is real MuJoCo physics: at release the apple gets the ballistic velocity that reaches the lens. The film adds the cracked screen.
- **Build:** `scripts/finish_film.sh ../egocheck-film-v9.mp4` runs the check, renders, then sets the pacing: everything before the end card plays at 1.2×, and the end card at 1.0× so the tagline lands.

## Repository map

```
judge/
  local_checks.py   step 1: OpenCV metrics + keyframes
  gemini_facts.py   step 2: Gemini fact sheet (schema, validation, one retry; AI Studio or Vertex)
  system_one.py     step 3+4: TypeSafe Jev questions + decision code (only file that knows TypeSafe)
  rules_decider.py  stand-in decider when Jev is unavailable (labelled rules_fallback)
  pipeline.py       runs everything, caching, Sentry, cards, report, sim replays
  baseline_llm.py   same decision by Gemini alone (for comparison, --baseline)
  live.py           live webcam mode
  cache.py, env.py  disk cache; .env loader
sim/
  scene.py          MuJoCo scene (Shadow Hand + table + apple + plate)
  replay.py         step-driven replay, grasp assist, mp4 render, --live viewer
  menagerie/        DeepMind MuJoCo Menagerie models (shadow_hand, franka_emika_panda)
dashboard/index.html
scripts/            make_demo_clips.py · record_dashboard.py · capture_shots.py · refresh_demo.sh
video/              HyperFrames project (demo/), local ffmpeg (bin/, tools/)
tests/              acceptance tests 1–5 + spec rules (offline)
examples/synthetic/ the synthetic placeholder clips used before real footage
SPEC.md             original build spec
```

## Known limitations

- **Images in TypeSafe:** a "dreamer-ready" (preview) feature, enabled per request with a header.
- **Jev sees still frames, not video.** Shake is measured locally instead.
- **The sim is a replay:** fixed scene layout and an assisted grasp.
- **Untuned thresholds:** the blur, dark and shake thresholds in `system_one.py` and `rules_decider.py` were set by hand, not tuned on a large labelled set.
- **Vertex sign-in expires.** Re-run `gcloud auth application-default login` if Gemini starts failing.
