# Clip Judge — build spec

Quality gate for egocentric robot-training video. Clips go in; each one is checked, judged
keep / fix / discard with a confidence score, and documented. Only good data reaches the robot.

## Why
Robotics companies are collecting lots of egocentric video, but much of it is unusable:
blurry, hands out of frame, object hidden, task not finished. Nobody checks it before training.
Clip Judge is the check, plus the documentation that should travel with every clip.

## Pipeline (one clip)
1. **Local checks** (OpenCV, no AI, runs on the laptop)
   - duration_s, width, height, fps
   - blur_score (variance of Laplacian, sampled every 10th frame, median)
   - brightness_mean (0-255), dark_frame_ratio
   - shake_score (mean optical-flow magnitude between sampled frames)
2. **Gemini fact sheet** (video understanding, returns strict JSON)
   - task_label (short text, e.g. "pick up cup and place on plate")
   - hands_visible_ratio (0-1), object_visible_ratio (0-1)
   - task_completed (true / false / unclear)
   - single_task (true / false)
   - faces_or_screens_visible (true / false)  -> privacy flag
   - steps: [{t_start, t_end, action, object}] where action is one of
     move_to | grasp | lift | place | release
   - notes (one sentence)
3. **System One verdict** (typed decision with probabilities)
   - input: local checks + fact sheet as one state block
   - question 1: verdict in {keep, fix, discard}
   - question 2: main_reason in {good, blurry, dark, shaky, hands_out_of_frame,
     object_hidden, task_failed, multiple_tasks, privacy}
   - output: chosen option + probability for every option
4. **Routing**
   - top probability >= 0.80 -> follow the verdict
   - below 0.80 -> human_review
5. **Documentation**
   - one data card per clip: `out/cards/<clip>.json` and `out/cards/<clip>.md`
   - one dataset report: `out/report.md` (counts per verdict, reject reasons,
     usable minutes, privacy flags)
   - `out/results.json` holds everything the dashboard needs

## Files
```
clip-judge/
  SPEC.md
  clips/                 input .mp4 / .mov files
  labels.csv             clip, my_verdict   (hand labels for the accuracy number)
  judge/
    local_checks.py      step 1
    gemini_facts.py      step 2
    system_one.py        step 3  (one function: decide(state) -> dict)
    pipeline.py          runs 1-5 over clips/, writes out/
    baseline_llm.py      same decision made by Gemini alone, for the comparison
  out/
    results.json
    cards/
    report.md
  dashboard/index.html   static page, reads ../out/results.json
  tests/test_pipeline.py
  README.md
```

## results.json shape
```json
{
  "summary": {"total": 15, "keep": 6, "fix": 3, "discard": 4, "human_review": 2,
              "agreement_with_labels": "13/15",
              "system_one_seconds_total": 0.0, "baseline_seconds_total": 0.0},
  "clips": [{
    "file": "clip_01.mp4",
    "local": {"duration_s": 8.2, "blur_score": 143.0, "brightness_mean": 121.0,
              "shake_score": 2.1},
    "facts": {"task_label": "...", "hands_visible_ratio": 0.9, "steps": []},
    "verdict": {"choice": "keep", "probs": {"keep": 0.91, "fix": 0.07, "discard": 0.02},
                "main_reason": "good", "routed_to": "keep"},
    "label": "keep"
  }]
}
```

## Dashboard (single HTML file, no framework)
- top bar: totals, agreement number, System One time vs baseline time
- grid of clip cards: video player, verdict stamp, confidence bar, main reason
- click a card: fact sheet, step timeline under the video, link to the data card
- filter buttons: keep / fix / discard / human review

## Rules for the coding agent
- API keys only from environment variables: GEMINI_API_KEY, SYSTEM_ONE_API_KEY, SENTRY_DSN
- `system_one.py` is the only file that knows the System One API. Fill it from the sponsor's
  docs. Do not invent endpoints. Until the docs are in, it raises NotImplementedError.
- Gemini must return JSON matching the fact-sheet schema; validate it, retry once on failure
- cache every API response in `out/cache/` keyed by file hash, so reruns are instant
- init Sentry in pipeline.py; a failed clip is logged and skipped, never crashes the run
- no login, no database, no upload form

## Acceptance tests
1. `python -m judge.pipeline clips/` finishes and writes results.json, cards, report.md
2. every clip has a verdict, probabilities that sum to ~1, and a routed_to value
3. a clip with top probability under 0.80 lands in human_review
4. a deliberately blurry clip does not get "keep"
5. dashboard opens from a local server and shows every clip with its verdict

## Stretch (only after tests 1-5 pass)
- replay the steps of a kept clip as a simple arm animation
- live webcam mode
