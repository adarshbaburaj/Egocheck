# EgoCheck: full technical write-up

**Tagline:** *Clean data today. Better robots tomorrow.*
**Repo:** https://github.com/adarshbaburaj/Egocheck
**Built:** 13 Nov 2026, at the hackathon

---

## 1. The one-minute version

Robotics teams train robot "brains" (manipulation policies, robotics foundation models) on **egocentric video**: first-person footage of people doing tasks, filmed from a head or chest camera, a phone, or a UMI gripper. A lot of that footage is bad. It's blurry, dark or shaky, the hands leave the frame, the object is hidden, the task is never finished, or a face or screen leaks private data. Today most of it goes into training unchecked. That's garbage in, garbage out.

**EgoCheck is the quality gate before training.** For every clip it:

1. **Measures** it locally with OpenCV: sharpness, brightness, shake, length.
2. **Watches** it with **Gemini 4 Hydrogen** (Vertex AI), which writes a structured *fact sheet*: the task, hand and object visibility, whether the task was finished, privacy, and timestamped steps.
3. **Judges** it with **TypeSafe's System One model, Jev**: seven calibrated yes/no questions over three keyframes, each answered with a real probability.
4. **Decides** in plain code: **keep / fix / discard**, with a confidence. Anything under **80%** goes to **human review**.

It writes a **data card** per clip, a batch **report**, a **results.json**, and a **dashboard**. Kept clips are **replayed on a simulated Shadow Hand in MuJoCo**, using the steps Gemini wrote down. Everything is traced in **Sentry**. The demo film is made in **HyperFrames**.

**Speed and cost:** System One answers in **0.15–0.43 s** for about **$0.0001 per clip** (measured). Gemini takes about 8.5 s per clip. All-in it's **about $1 per hour of video** (Gemini's part is estimated from list price).

---

## 2. The problem, in numbers

| Fact | Value | Source / status |
|---|---|---|
| Raw egocentric capture | **$15–40 per hour** | vendor estimates (DexSet, DataXpower) |
| Packaged datasets | $90–1,800 per hour | vendor price lists |
| Share of footage that's unusable | **15–30%** | vendor estimates |
| Checking by hand | ~$25 per hour of video | estimate (one person watching) |
| EgoCheck | **≈ $1 per hour of video** | System One measured, Gemini estimated |

**The film's example (illustrative):** 5,000 h at ~$30/h = $150,000 of data. At 30% unusable, that's **$45,000 of junk**. Checking all of it with EgoCheck costs about **$5,000**. Bad data costs money, and it also costs **training time** and **model quality**.

---

## 3. Architecture

```
clip.mp4 ──► [1] local_checks.py (OpenCV)        blur, brightness, dark ratio, shake, fps, duration
          │                                      + 3 keyframes (10% / 50% / 90%)
          ├─► [2] gemini_facts.py (Gemini 4 Hydrogen, Vertex AI)
          │                                      fact sheet JSON, validated, 1 retry
          ├─► [3] system_one.py (TypeSafe Jev)   7 yes/no questions, 3 images, one batched request
          │        → failure probabilities
          └─► [4] combine() truth table          keep / fix / discard + confidence + main reason
                     │
                     ▼
               pipeline.route()  conf ≥ 0.80 → the verdict ; else → human_review
                     │
     ┌───────────────┼───────────────┬──────────────────┬───────────────────┐
 results.json   cards/<clip>.md|json   report.md   dashboard/index.html   out/sim/<clip>.mp4 (kept clips)
                                                                     (MuJoCo Shadow Hand replay)
             Sentry: one transaction per run, a span per stage, Gemini traced, errors + metrics
```

**Design rule (from the TypeSafe skill):** *judgments go in the model, decisions go in code.* The models answer narrow factual questions. A short, readable truth table makes the call. That keeps decisions explainable, testable and tunable without re-prompting.

**One file per outside service:**

| File | Knows about |
|---|---|
| `judge/local_checks.py` | OpenCV only |
| `judge/gemini_facts.py` | Gemini / Vertex only |
| `judge/system_one.py` | TypeSafe only (the only file that does) |
| `judge/pipeline.py` | orchestration, caching, routing, outputs, Sentry |
| `judge/rules_decider.py` | offline fallback when there's no System One key |
| `judge/baseline_llm.py` | "Gemini alone decides", for comparison (`--baseline`) |
| `judge/live.py` | webcam mode |
| `judge/env.py` | tiny `.env` loader |
| `sim/scene.py`, `sim/replay.py`, `sim/chaos.py` | MuJoCo |

---

## 4. Stage 1: local checks (OpenCV)

These run on the laptop. They're fast, free and deterministic.

- **Sampling:** every **10th frame**, downscaled to **640 px** wide.
- **Sharpness (`blur_score`)** is the **variance of the Laplacian** (`cv2.Laplacian(gray, CV_64F).var()`), the median over sampled frames. The Laplacian is a second-derivative edge filter: sharp images have strong edges and high variance; blurry images score low.
  - Real examples: blurry clip **8.1**, sharp kept clip **619.8**.
- **Brightness** is the mean gray level. A frame with mean below **40** counts as *dark*, and `dark_frame_ratio` is the share of dark frames.
- **Shake (`shake_score`)** is the mean magnitude of **Farneback dense optical flow** between consecutive sampled frames, computed at 320 px for speed. More motion between frames means more camera shake.
- **fps, duration, frame count** are also recorded.
- **Keyframes** are three JPEGs at **10% / 50% / 90%** of the clip (start / middle / end), sent to System One.

**Turning measurements into probabilities** uses a linear ramp between a "good" and a "bad" value (`_ramp`):

| Signal | good → bad |
|---|---|
| shake | 6 → 15 |
| blur | 120 → 30 (lower is worse) |
| brightness | 70 → 25 (lower is worse) |

---

## 5. Stage 2: Gemini fact sheet

- **Model:** `gemini-4-hydrogen` via **Vertex AI** (`google-genai` SDK, `GOOGLE_GENAI_USE_VERTEXAI=true`, location `global`), on the team's Google Cloud project and credits. Auth is Application Default Credentials (`gcloud auth application-default login`).
- **Input:** the **whole video, inline as bytes**, plus a prompt that asks for JSON only.
- **Output (fact sheet):**
  ```json
  {"task_label": "pick up apple and place on plate",
   "hands_visible_ratio": 0.67, "object_visible_ratio": 1.0,
   "task_completed": true, "single_task": true, "faces_or_screens_visible": false,
   "steps": [{"t_start": 0.0, "t_end": 1.0, "action": "move_to", "object": "apple"}, ...],
   "notes": "Clear, well-lit view of the single-hand manipulation..."}
  ```
- **Validation:** types, ranges (ratios in 0–1) and a known action vocabulary. If the JSON is invalid, it **retries once**, then fails that clip gracefully. Both paths are tested.
- **Why Gemini:** it's a strong video understanding model. It "watches" and **describes**, and it **does not decide**. The `steps` also drive the robot replay.
- **Fallback keys:** AI Studio keys are tried in order if Vertex isn't configured.
- ~**8.5 s per clip**.

---

## 6. Stage 3: TypeSafe System One (Jev)

### What a "System One" model is
In the "System 1 / System 2" sense, it's a **fast, intuitive decision model**, not a slow reasoning chat LLM. Jev (`jev-latest`, which resolved to **jev-1.14.0**) takes structured state (JSON plus images) and answers **Noul questions**: yes/no questions that come back as a **calibrated probability**, not free text.

### The request
- **State:** the fact sheet, the local measurements, the task label, and **three keyframes** (`frames.start/middle/end`). Images are enabled with the header `TypeSafe-Danger-Zone-Images: true`.
- **Seven questions**, each with explicit true/false **criteria** and a `fail_if` direction:

| Reason | Question (abridged) | Fails when |
|---|---|---|
| blurry | Is `frames.middle` too blurry to see the object's edges? | yes |
| dark | Is `frames.middle` too dark to tell what the hands hold? | yes |
| hands_out_of_frame | Are the hands inside the picture in `frames.middle`? | no |
| object_hidden | **Before the person touches it**, is the object visible in `frames.start`? | no |
| task_failed | Does `frames.end` show the task finished? | no |
| multiple_tasks | Do start and end show two different objects? | yes |
| privacy | Is a face or readable screen visible in any frame? | yes |

- **Measured:** **2,706 input tokens** per clip and **0.15–0.43 s**. Price is **$0.042 per 1M input tokens** (output free), so **$0.000114 per clip**.

### The decision: `combine()`, a truth table in code
1. Turn each answer into a **failure probability** `p`.
2. **Merge** model and measurement by taking the max of each pair: `blurry = max(Jev, measured blur)` and `dark = max(Jev, measured dark)`. `shaky` comes from optical flow only. If either signal is confident, trust it.
3. Apply TypeSafe's recommended **bands**: `p < 0.3` = pass, `p > 0.7` = fail, in between = *unsettled*.
4. Truth table:

| Condition | Verdict | Confidence |
|---|---|---|
| any **unfixable** failure (blurry, hands out, object hidden, task failed) | **discard** | p of the worst failure |
| only **fixable** failures (dark, shaky, multiple_tasks, privacy) | **fix** | p of the worst failure |
| something unsettled | leans keep / fix / discard | max(p, 1−p) ≤ 0.7, so it always goes to **human review** |
| everything passes | **keep** | 1 − max p ("as sure as the weakest pass") |

5. **Routing:** `confidence ≥ 0.80` → the verdict; otherwise → **human_review**.

"Fixable" means repairable after the fact: brighten, stabilise, blur a screen or face, or trim a second task out.

### Two calibration lessons from the build
- **We replaced the product of probabilities with the truth table.** First we multiplied pass-probabilities together. With seven questions, small doubts compound, and clean clips only reached ~0.81 confidence. The truth table rates a clean clip by its **weakest** check, which is easier to explain and better behaved.
- **We moved the object_hidden question to the start frame.** Asked on the middle frame, it penalised the person's own hand naturally covering the apple while grasping. We fixed the *question*, not the threshold. That follows the TypeSafe guidance: fix the state or wording before touching numbers.

### Fallback and caching
- **No key → `rules_decider.py`**, an offline decider on the same signals. A `_FellBack` guard keeps fallback results **out of the System One cache**.
- **Cache key:** the clip's **sha256** plus the state, questions and model. Re-runs skip paid API calls (tested), and changing a question invalidates the cache.

---

## 7. Outputs

- **`out/results.json`:** every clip's local metrics, fact sheet, checks, verdict probabilities, confidence, main reason, route and sha256.
- **`out/cards/<clip>.md` / `.json`:** a **data card** per clip, in the spirit of "datasheets for datasets": verdict, why, probabilities, recording metrics, fact sheet, steps.
- **`out/report.md`:** batch summary with counts per route, reasons, usable minutes, and agreement with labels if `labels.csv` (or the `good`/`bad` filename convention) is given.
- **Dashboard (`dashboard/index.html`):** a static page that reads `results.json`.
  - A one-sentence hero ("2 of 12 clips are ready for training").
  - A route bar whose legend doubles as a filter.
  - Grid and list views with view transitions.
  - A drawer per clip with the verdict explained, probability bars, video plus step timeline, the **robot replay**, and the checks list.
  - Light and dark themes.
- **Live mode (`judge/live.py`):** near-real-time webcam judging. Every **3 s window** (`--window`) gets the same local checks plus a **System One request on 3 frames**, answered in a **background thread** so the video stays smooth. The overlay shows ok / unsure / problem per check, the verdict and its route (same `combine()` and 0.80 rule), and the latency in ms. Gemini isn't used live (~8 s per clip). `--rules` runs offline, and `--record` saves the annotated stream.

**CLI:**
```bash
.venv/bin/python -m judge.pipeline clips/ [--decider auto|system_one|rules] [--threshold 0.8] [--baseline] [--no-sim]
```

---

## 8. Results on our 12 real clips

The clips were recorded on our phones: apple to plate, transcoded from HEVC to H.264.

| Route | Count | Clips (confidence) |
|---|---|---|
| **keep** | 2 | 130610 (0.86), 130630 (0.81) |
| **discard** | 4 | 125233 blurry (1.00), 125309 blurry (0.89), 125319 blurry (0.90), 130621 task not finished (0.80) |
| **human_review** | 6 | 125344 (0.72), 130640 (0.52), 130656 task_failed (0.77), demo01 privacy (0.63), demo02 privacy (0.58), demo03 (0.56) |

**Talking points:**
- **All three blurry clips were discarded.** That's required by a test: a blurry clip is never kept.
- **The `demo0X_good` clips went to a human, correctly.** We labelled them "good", but a **laptop screen is visible**, which is a real privacy issue (fixable by blurring it). EgoCheck caught something we missed when labelling.
- **130656 (task_failed 0.77) went to review, not discard,** because 0.77 < 0.80. That's the threshold working as designed: when the model isn't sure, a person decides.

---

## 9. Robot replay (MuJoCo)

- **Engine:** **MuJoCo 4.0.1**, with **mjpython** for the live viewer on macOS.
- **Robot:** the **Shadow Dexterous Hand** from DeepMind's **MuJoCo Menagerie**, attached through `MjSpec` to a **mocap body**. Mocap is a kinematically driven forearm, so no arm inverse kinematics is needed.
- **Fingers:** **physical position actuators**. The grasp "closes" by ramping the actuator targets (a `CLOSED` pose table).
- **Physics settings:** timestep 0.002 s, `implicitfast` integrator, elliptic friction cones, `impratio=10`.
- **Scene:** a table, an apple (free joint, 0.18 kg, r = 4 cm) and a plate.
- **Steps from Gemini:** move_to → grasp → lift → place → release, timed by the fact sheet and turned into keyframes with **smoothstep** interpolation.
- **Grasp assist (honest):** a **kinematic attach**. Once the fingers close (grip > 0.8), the apple keeps its pose relative to the palm. Real learned grasping is out of scope; this is a *replay*.
- **Side grasp with the thumb up**, like the human in the clip:
  - The **hand orientation** came from **searching rotations**, scoring finger direction, curl direction and thumb direction.
  - The **palm offset** came from a **physics grid search**: close the fingers at each candidate, and reject any with table contact or apple displacement.
  - Final values: quat `[-0.5, -0.5, 0.5, -0.5]`, palm offset (6, −7.5, 0) cm from the apple. Closing makes 3 finger contacts, no table contact, and the apple moves only 1.1 cm.
- **Camera:** **first-person** (azimuth 180°, elevation −50°, slight head sway), so the replay looks like the source clip.
- **Output:** 1920×1080 at 30 fps. `apple_on_plate = true` for both kept clips.

### The "garbage out" sim (`sim/chaos.py`), made for the film
A hand "trained on bad data": it twitches on the approach (sinusoidal noise on the target), grabs the apple, winds up and **throws it at the camera**. The throw is real ballistics. At release, the apple gets the velocity that reaches the lens in T = 0.30 s:

**v = (lens − p₀) / T + (0, 0, ½·g·T)**

MuJoCo then integrates the flight, with collisions off for the apple after release. The clip ends just before impact; the film adds the cracked screen.

---

## 10. Monitoring (Sentry)

- **Project:** org `loopers-09`, project `egocheck`. The DSN lives in `.env`.
- **Tracing:** one **transaction per batch run**, with a **span per stage** (local checks, Gemini, System One, decide), so slow stages are visible.
- **GoogleGenAIIntegration** traces each Gemini call, including tokens.
- **Errors:** API failures and invalid JSON are captured with the clip name. A failed clip is skipped, not fatal (tested).
- **Logs:** forwarded through `LoggingIntegration`.
- **Metrics:** verdict counts per run.

---

## 11. Tests (`tests/test_pipeline.py`, 13 tests, pytest)

1. The pipeline writes all outputs.
2. Every clip has a verdict, probabilities and a route.
3. Low confidence goes to human review.
4. **A blurry clip is never kept.**
5. The dashboard is served with every clip; 5b: it renders in a real browser (Playwright, Chromium).
6. **System One is not invented.** It raises `NotImplementedError` until real docs and a key exist; no made-up endpoints.
7. A failed clip is skipped, not fatal.
8. The cache makes re-runs skip the API.
9. Fact-sheet validation.
10. Gemini retries once, then succeeds.
11. Gemini gives up after one retry.
12. The rules decider uses the fact sheet.

---

## 12. The demo film (HyperFrames)

- **Format:** 1920×1080, 30 fps, **66 s**. File: `video/egocheck-film-v9.mp4`. Project: `video/egocheck/`.
- **HyperFrames 0.9.42:** video as **HTML + GSAP**. Each scene is a sub-composition (`<template>`, `#root`) with one **paused GSAP timeline** registered on `window.__timelines`. The renderer **seeks** the timeline frame by frame, so renders are **deterministic** (no `Math.random`, no timers). Timed elements use `data-start` / `data-duration`.
- **13 scenes:** Elon → $500K → buying data → dream bubble → POP → garbage in / garbage out (real clips, then the MuJoCo throw, then the cracked screen) → what was lost (dream, money, **time**) → logo → real dashboard → Measure / Watch / Judge / Decide on a real clip → System One speed race → savings → robot replay → end card.
- **Look:** chalk-white stick figures on #0A0A0A with one accent colour (logo green #8FD45E). A **"boil" filter** gives the hand-drawn wobble: SVG `feTurbulence` plus `feDisplacementMap`, with the seed stepped 12×/s.
- **Audio:**
  - **Narration:** ElevenLabs TTS, `eleven_multilingual_v2`, built-in "Jessica" voice (`scripts/make_voice.py`).
  - **Music:** Google **Lyria** (`lyria-002`) on Vertex AI (`scripts/make_music.py`). A theme comes in at the logo reveal and a hype hit plays under the end card. Lyria's "recitation check" blocks prompts that sound too close to existing music, so we reworded those prompts.
  - **SFX:** the HyperFrames bundled library, plus Openverse CC BY sounds (credited in `assets/sfx/CREDITS-openverse.md`).
- **Pacing:** `scripts/finish_film.sh` renders, then plays everything before the end card at **1.2×** (ffmpeg `setpts` / `atempo`) and the end card at **1.0×**, so the tagline lands.
- **Workflow:** `hyperframes check` (lint, runtime, layout, contrast), `snapshot` frames, review, then `render`.

---

## 13. How it was built ("software factory")

- **Spec first:** a written build spec with a fixed `results.json` shape, file list, rules and acceptance tests. Coding agents (Claude Code) then built it one file at a time against that spec.
- **Strict boundaries:** one file per outside service. Keys only from environment variables and `.env` (git-ignored). No invented endpoints.
- **Reproducible:** results are cached by sha256 and model calls are deterministic where possible. The film renders deterministically. Everything regenerates from scripts: `refresh_demo.sh`, `make_voice.py`, `make_music.py`, `finish_film.sh`, `sim.replay`, `sim.chaos`.
- **Observable:** Sentry on every run.

---

## 14. Honest limitations

- **The sim is a replay, not learned control.** The layout is fixed, the grasp is assisted, and the motion follows Gemini's steps.
- **Gemini cost per clip is estimated** from list price. System One numbers are measured.
- **Market figures are vendor estimates** and are labelled "illustrative" in the film.
- **The test set is small:** 12 real clips of one task (apple to plate). The thresholds (0.3 / 0.7 bands, 0.80 routing) are sensible defaults, not tuned on a large labelled set.
- **Live mode skips Gemini** (~8 s per clip), so live verdicts have no fact sheet: no steps and no full-video view of whether the task was finished. They come from local checks plus System One on 3 frames per window.
- **"Fix" is a label, not an action.** EgoCheck says *what* to fix, but doesn't yet brighten, stabilise or blur automatically.

---

## 15. Questions judges might ask, with answers

**Q: Why not just ask a big LLM "is this clip good?"**
A: Three reasons. **Cost and speed:** a big video model takes ~8.5 s per clip; System One takes 0.15–0.43 s and ~$0.0001. **Calibration:** a chat model's "I'm 90% sure" is text, not a probability, while Noul answers come back as real probabilities we can threshold. **Explainability:** we get seven separate answers and a readable truth table, not one opaque yes/no. We also kept a `--baseline` mode where Gemini alone decides, for comparison.

**Q: Then why use Gemini at all?**
A: Gemini **watches the whole video** and describes it: task, steps, whether the task was finished, privacy. That's perception. System One makes the **decision** from that plus the keyframes and measurements. Judgments sit in the model, decisions in code.

**Q: What does "calibrated" mean here?**
A: When the model says 0.8, it should be right about 80% of the time. That's what lets us set one threshold (0.80) for what goes to a human.

**Q: How did you pick 0.80, 0.3 and 0.7?**
A: 0.3 and 0.7 are TypeSafe's recommended pass/fail bands for Noul answers. 0.80 is a routing policy: a team can raise it to send more to humans (safer) or lower it to automate more (cheaper). It's a CLI flag (`--threshold`). With more labelled data we'd tune it on a precision/recall curve.

**Q: What's the difference between "fix" and "discard"?**
A: Fix covers problems you can repair after recording: too dark (brighten), shaky (stabilise), a second task (trim), or a face or screen (blur). Discard covers what you can't recover: blur, hands out of frame, object hidden, task not finished.

**Q: Your own "good" clips went to human review. Isn't that a failure?**
A: No, it's the system working. A laptop screen was visible in them, which is a privacy issue we missed when labelling. Confidence was below 0.80, so a human decides. That's the point of the gate.

**Q: How do you combine the OpenCV numbers with the model?**
A: For blur and darkness we take the **max** of the measured probability and System One's answer. If either is confident something's wrong, we trust it. Shake comes only from optical flow, because three still frames can't show motion.

**Q: Why three keyframes and not the whole video for System One?**
A: Cost and speed. Start, middle and end answer the questions we ask (object visible before touching, hands in frame mid-task, task finished at the end), while Gemini covers the whole video. Three images cost 2,706 tokens.

**Q: How does it scale to thousands of hours?**
A: Clips are independent, so the work parallelises. Local checks are CPU-cheap, System One is sub-second, and Gemini is the long pole (~8.5 s) but parallelises across requests. Results are cached by file hash, so re-runs are free. At ≈$1 per hour of video, 5,000 hours costs about $5,000.

**Q: What happens when an API fails?**
A: Gemini retries once and the clip is then skipped with an error recorded. With no System One key we fall back to the offline rules decider, and fallback results are never cached as System One results. Every failure goes to Sentry with the clip's name. One bad clip never stops the batch (tested).

**Q: Is the robot actually learning from the clip?**
A: No, and we say so. It's a **replay**: Gemini's timestamped steps drive a scripted Shadow Hand in MuJoCo with a grasp assist. It shows the clip contains a complete, usable demonstration. Learned policies are the next step.

**Q: How did you get the side grasp to look right?**
A: We searched hand rotations for "fingers forward, thumb up, palm facing the apple", then ran a physics grid search over palm positions. We rejected any with table contact or apple movement. The winner makes 3 finger contacts and moves the apple only 1.1 cm when closing.

**Q: Is the apple throw real physics?**
A: Yes. At release we give the apple the ballistic velocity that reaches the camera in 0.3 s (v = Δp/T + ½gT upward), then MuJoCo simulates the flight.

**Q: What does Sentry show you?**
A: A trace per run with a span per stage, so you can see where the time goes. Gemini calls with token usage, errors tagged with the clip, logs, and verdict metrics.

**Q: Privacy: do you store the videos?**
A: Everything runs on our machine. Clips go to Gemini (Vertex, our project) and three keyframes go to TypeSafe for judging. Output cards hold metrics and text, not video. Privacy is also a check: faces and screens get flagged.

**Q: How accurate is it?**
A: On our 12 clips, every blurry and unfinished clip was caught, the 2 clean ones were kept, and uncertain cases went to a human. That's a small set of one task. A real evaluation needs a larger labelled dataset; the report already computes agreement with `labels.csv` when one is given.

**Q: Can it run in real time?**
A: Mostly, yes. Live mode already judges a webcam every 3 s: OpenCV on every frame, then System One on 3 frames per window in a background thread (0.15–0.43 s per call). Gemini is the only slow part (~8 s), so live mode skips it. The full real-time design: OpenCV per frame for instant "too dark / shaky / hands out" warnings, System One on a rolling window every ~1 s, and Gemini's fact sheet once at the end of each clip (or Gemini's streaming Live API). The recorder gets warnings while filming, and a verdict within about a second of the clip ending.

**Q: What would you build next?**
A: Auto-fix (brighten, stabilise, blur faces and screens). More task types and multi-step tasks. Threshold tuning on a large labelled set. Streaming checks during capture (warn the recorder live). An API or upload service for data vendors. Turning data cards into dataset-level datasheets.

**Q: Who pays for this?**
A: Robotics teams buying or collecting egocentric and UMI data, and the data vendors selling it, who could certify their data before delivery.

**Q: How long did it take, and what did AI do?**
A: One hackathon day. We wrote the spec. Claude Code (Claude Opus 5.5) built the code file by file against it and helped make the film. The models in the product are Gemini (perception) and System One (decision).

**Q: Why HyperFrames for the video?**
A: The film is code: HTML + GSAP, rendered deterministically, so we could drop in real product footage (dashboard recording, real clips, MuJoCo renders), iterate with check/snapshot, and re-render in about a minute after every change.

---

## 16. Glossary

| Term | Meaning |
|---|---|
| **Egocentric video** | First-person footage, filmed from the person's point of view |
| **UMI** | Universal Manipulation Interface: a handheld gripper with a camera, used to collect robot demonstrations |
| **Robotics foundation model / policy** | A model that maps observations to robot actions, trained on demonstrations |
| **System One model** | A fast decision model that returns calibrated probabilities (here TypeSafe's Jev) |
| **Noul question** | TypeSafe's yes/no question format, answered with a probability |
| **Calibration** | Predicted probability matches how often it's right |
| **Truth table** | Explicit rules mapping conditions to the decision |
| **Laplacian variance** | A sharpness measure: variance of the second-derivative edge response |
| **Farneback optical flow** | Dense per-pixel motion estimate between frames; here a shake measure |
| **Keyframe** | A still frame sampled at a set point (10%, 50%, 90%) |
| **Fact sheet** | Gemini's structured JSON description of a clip |
| **Data card** | A per-clip summary of quality, provenance and verdict |
| **Human-in-the-loop** | Uncertain cases routed to a person |
| **Vertex AI** | Google Cloud's AI platform, hosting Gemini and Lyria |
| **ADC** | Google Application Default Credentials |
| **MuJoCo** | Physics engine for robotics (Multi-Joint dynamics with Contact) |
| **Menagerie** | DeepMind's collection of curated MuJoCo robot models |
| **Shadow Hand** | A 24-DoF dexterous robot hand |
| **Mocap body** | A kinematically driven body in MuJoCo (moved directly, not by forces) |
| **Kinematic attach** | Locking the object's pose to the palm while held (grasp assist) |
| **Smoothstep** | Ease curve 3k² − 2k³ for smooth motion between keyframes |
| **Sentry span / transaction** | Timed units of work in a trace |
| **HyperFrames** | HTML-to-video framework: seekable GSAP timelines rendered frame by frame |
| **GSAP** | JavaScript animation library |
| **Lyria** | Google's music generation model |
| **sha256 cache** | Results keyed by the file's hash, so the same clip is never paid for twice |

---

## 17. Commands cheat sheet

```bash
# judge a folder of clips (also renders robot replays for kept clips)
.venv/bin/python -m judge.pipeline clips/
# offline / baseline
.venv/bin/python -m judge.pipeline clips/ --decider rules
.venv/bin/python -m judge.pipeline clips/ --baseline
# live webcam
.venv/bin/python -m judge.live --record out/live.mp4
# robot replay of a clip, or live viewer
.venv/bin/python -m sim.replay 20261113_130610
mjpython -m sim.replay 20261113_130610 --live
# the "trained on bad data" throw
.venv/bin/python -m sim.chaos
# tests
.venv/bin/python -m pytest -q
# film: narration, music, render + pacing
.venv/bin/python scripts/make_voice.py
.venv/bin/python scripts/make_music.py
scripts/finish_film.sh ../egocheck-film-v9.mp4
```
