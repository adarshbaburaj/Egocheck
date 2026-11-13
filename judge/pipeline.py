"""Clip Judge pipeline: local checks -> Gemini facts -> verdict -> routing -> docs.

    python -m judge.pipeline clips/ [--out out] [--labels labels.csv]
                                    [--decider auto|system_one|rules]
                                    [--baseline] [--threshold 0.80]

A clip that fails at any step is logged (and sent to Sentry if SENTRY_DSN is
set) and skipped; the run itself never crashes because of one clip.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from . import gemini_facts, local_checks, rules_decider, system_one
from .cache import cached, file_hash

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
THRESHOLD = 0.80
VERSION = "0.1.0"

log = logging.getLogger("clip_judge")


# ---------------------------------------------------------------- helpers

def init_sentry() -> bool:
    """Errors, tracing, logs and Gemini (AI) monitoring. DSN from SENTRY_DSN (.env)."""
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        return False
    import sentry_sdk
    from sentry_sdk.integrations.google_genai import GoogleGenAIIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration

    env = os.environ.get("SENTRY_ENVIRONMENT", "development")
    default_rate = "1.0" if env == "development" else "0.1"
    sentry_sdk.init(
        dsn=dsn,
        environment=env,
        release=f"clip-judge@{VERSION}",
        traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", default_rate)),
        integrations=[
            LoggingIntegration(capture_sentry_logs=True),   # pipeline log lines -> Sentry Logs
            GoogleGenAIIntegration(),                       # Gemini calls -> AI agent monitoring
        ],
    )
    return True


def report_error(exc: BaseException, clip: str) -> None:
    try:
        import sentry_sdk
        with sentry_sdk.new_scope() as scope:
            scope.set_tag("clip", clip)
            sentry_sdk.capture_exception(exc)
    except Exception:
        pass


def route(probs: dict[str, float], threshold: float = THRESHOLD) -> str:
    choice = max(probs, key=probs.get)
    return choice if probs[choice] >= threshold else "human_review"


def label_agrees(label: str, choice: str) -> bool:
    """Labels are keep / fix / discard, or "bad" = anything but keep."""
    return choice != "keep" if label == "bad" else choice == label


def load_labels(path: Optional[Path], clips: list[Path] | None = None) -> dict[str, str]:
    """labels.csv (clip,my_verdict). Without one, a clip named *good* is labelled keep
    and every other clip "bad" (the recording convention for this demo)."""
    if path and path.exists():
        with open(path, newline="") as f:
            return {r["clip"].strip(): r["my_verdict"].strip().lower()
                    for r in csv.DictReader(f) if r.get("clip") and r.get("my_verdict")}
    if clips is None:
        return {}
    return {p.name: ("keep" if "good" in p.stem.lower() else "bad") for p in clips}


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return p.resolve().as_posix()


# ---------------------------------------------------------------- deciders

class _FellBack(Exception):
    pass


class AutoDecider:
    """Use System One; the first time it raises NotImplementedError, switch to
    the labelled rules fallback for the rest of the run."""

    def __init__(self):
        self.name = "system_one"

    def __call__(self, state: dict) -> dict:
        if self.name == "system_one":
            try:
                return system_one.decide(state)
            except NotImplementedError:
                log.warning("System One not implemented yet -> using rules_fallback decider")
                self.name = "rules_fallback"
        return rules_decider.decide(state)


def pick_decider(name: str):
    """Returns an object/func with the decide signature, plus a name getter."""
    if name == "rules":
        return lambda: "rules_fallback", rules_decider.decide
    if name == "system_one":
        return lambda: "system_one", system_one.decide
    auto = AutoDecider()
    return lambda: auto.name, auto


def check_decision(d: dict) -> None:
    for q, opts in (("verdict", system_one.VERDICTS), ("main_reason", system_one.REASONS)):
        probs = d[q]["probs"]
        if set(probs) != set(opts):
            raise ValueError(f"{q} probs must cover {opts}, got {sorted(probs)}")
        if abs(sum(probs.values()) - 1) > 0.02:
            raise ValueError(f"{q} probs sum to {sum(probs.values()):.3f}")


# ---------------------------------------------------------------- one clip

def default_facts(path: Path, duration_s: float) -> dict:
    return gemini_facts.get_facts(path, duration_s)


def process_clip(path: Path, *, cache_dir: Path, decider_name: Callable[[], str],
                 decide_fn: Callable[[dict], dict],
                 facts_fn: Callable[[Path, float], Optional[dict]],
                 threshold: float, baseline: bool) -> dict:
    import sentry_sdk

    fhash = file_hash(path)
    with sentry_sdk.start_span(op="clip.local_checks", name="OpenCV local checks"):
        local, _ = cached(fhash, "local", lambda: local_checks.run(path), cache_dir=cache_dir)

    facts_status = "ok"
    try:
        with sentry_sdk.start_span(op="clip.gemini_facts", name="Gemini fact sheet"):
            facts, _ = cached(fhash, "gemini_facts",
                              lambda: facts_fn(path, local["duration_s"]), cache_dir=cache_dir)
    except gemini_facts.GeminiUnavailable as e:
        facts, facts_status = None, f"unavailable: {e}"

    state = {"local": local, "facts": facts,
             "frames": local_checks.keyframes(path, cache_dir / "frames" / fhash[:16])}

    def _decide():
        t0 = time.perf_counter()
        with sentry_sdk.start_span(op="clip.decide", name="System One verdict"):
            out = decide_fn(state)
        return {"result": out, "seconds": round(time.perf_counter() - t0, 4),
                "decider": decider_name()}

    if decider_name() == "system_one":
        def _system_one():
            d = _decide()
            if d["decider"] != "system_one":   # auto mode fell back: don't cache as System One
                raise _FellBack(d)
            return d
        try:
            dec, _ = cached(fhash, "system_one", _system_one, cache_dir=cache_dir,
                            extra={"state": state, "questions": system_one.QUESTIONS, "model": system_one.MODEL})
        except _FellBack as fb:
            dec = fb.args[0]
    else:
        dec = _decide()
    used = dec.get("decider") or decider_name()
    decision = dec["result"]
    check_decision(decision)

    vprobs = decision["verdict"]["probs"]
    record = {
        "file": path.name,
        "src": _rel(path),
        "sha256": fhash,
        "local": local,
        "facts": facts,
        "facts_status": facts_status,
        # per-check severity 0 (fine) .. 1 (bad); same scale whichever decider ran
        "checks": None,  # filled below from the decider's failure probabilities
        "verdict": {
            "choice": decision["verdict"]["choice"],
            "probs": vprobs,
            "confidence": max(vprobs.values()),
            "main_reason": decision["main_reason"]["choice"],
            "reason_probs": decision["main_reason"]["probs"],
            "routed_to": route(vprobs, threshold),
            "decider": used,
            "seconds": dec["seconds"],
        },
    }
    if "severities" in decision:
        record["verdict"]["severities"] = decision["severities"]
    # per-check failure likelihood 0 (fine) .. 1 (bad) for the dashboard
    record["checks"] = decision.get("failure_probs") or rules_decider.severities(state)

    if baseline:
        from . import baseline_llm
        try:
            record["baseline"], _ = cached(fhash, "baseline_llm",
                                           lambda: baseline_llm.decide(path), cache_dir=cache_dir)
        except gemini_facts.GeminiUnavailable:
            record["baseline"] = None
    return record


# ---------------------------------------------------------------- documentation

def write_card(rec: dict, cards_dir: Path, threshold: float) -> None:
    stem = Path(rec["file"]).stem
    card = {
        "card_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "pipeline": f"clip-judge {VERSION}",
        "routing_threshold": threshold,
        **rec,
    }
    (cards_dir / f"{stem}.json").write_text(json.dumps(card, indent=2))

    v, loc, f = rec["verdict"], rec["local"], rec["facts"]
    lines = [
        f"# Data card — {rec['file']}",
        "",
        f"**Routed to:** `{v['routed_to']}`  ",
        f"**Verdict:** {v['choice']} ({v['confidence']:.0%})  ·  **Main reason:** {v['main_reason']}  ",
        f"**Decided by:** {v['decider']}" + ("  _(rule-based stand-in; System One not wired yet)_"
                                             if v["decider"] == "rules_fallback" else ""),
        f"**Human label:** {rec.get('label') or '—'}",
        "",
        "| verdict | probability |", "|---|---|",
        *[f"| {k} | {p:.3f} |" for k, p in v["probs"].items()],
        "",
        "## Recording",
        "",
        "| metric | value |", "|---|---|",
        f"| duration | {loc['duration_s']} s |",
        f"| resolution | {loc['width']}×{loc['height']} @ {loc['fps']} fps |",
        f"| blur score (Laplacian var, median) | {loc['blur_score']} |",
        f"| brightness mean (0–255) | {loc['brightness_mean']} |",
        f"| dark frame ratio | {loc['dark_frame_ratio']} |",
        f"| shake score (mean optical flow, px) | {loc['shake_score']} |",
        f"| sha256 | `{rec['sha256']}` |",
        "",
        "## Content (Gemini fact sheet)",
        "",
    ]
    if f:
        lines += [
            f"- **Task:** {f['task_label']}",
            f"- **Hands visible:** {f['hands_visible_ratio']:.0%} · "
            f"**Object visible:** {f['object_visible_ratio']:.0%}",
            f"- **Task completed:** {f['task_completed']} · **Single task:** {f['single_task']}",
            f"- **Privacy (faces/screens):** {'⚠️ yes' if f['faces_or_screens_visible'] else 'no'}",
            f"- **Notes:** {f['notes']}",
            "",
            "| t_start | t_end | action | object |", "|---|---|---|---|",
            *[f"| {s['t_start']} | {s['t_end']} | {s['action']} | {s['object']} |" for s in f["steps"]],
        ]
    else:
        lines.append(f"_Fact sheet {rec['facts_status']}._")
    (cards_dir / f"{stem}.md").write_text("\n".join(lines) + "\n")


def summarise(records: list[dict], failed: list[dict], decider: str) -> dict:
    routed = [r["verdict"]["routed_to"] for r in records]
    labelled = [r for r in records if r.get("label")]
    agree = sum(label_agrees(r["label"], r["verdict"]["choice"]) for r in labelled)
    base = [r["baseline"] for r in records if r.get("baseline")]
    return {
        "total": len(records),
        **{k: routed.count(k) for k in ("keep", "fix", "discard", "human_review")},
        "agreement_with_labels": f"{agree}/{len(labelled)}" if labelled else "n/a",
        "baseline_agreement_with_labels": (
            f"{sum(label_agrees(r['label'], r['baseline']['verdict']['choice']) for r in labelled if r.get('baseline'))}"
            f"/{sum(1 for r in labelled if r.get('baseline'))}") if base and labelled else "n/a",
        "decider": decider,
        "system_one_seconds_total": round(sum(r["verdict"]["seconds"] for r in records
                                              if r["verdict"]["decider"] == "system_one"), 3),
        "decider_seconds_total": round(sum(r["verdict"]["seconds"] for r in records), 4),
        "baseline_seconds_total": round(sum(b["seconds"] for b in base), 3),
        "usable_minutes": round(sum(r["local"]["duration_s"] for r in records
                                    if r["verdict"]["routed_to"] == "keep") / 60, 2),
        "total_minutes": round(sum(r["local"]["duration_s"] for r in records) / 60, 2),
        "failed": failed,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def write_report(summary: dict, records: list[dict], path: Path, threshold: float) -> None:
    s = summary
    reasons: dict[str, int] = {}
    for r in records:
        if r["verdict"]["choice"] != "keep":
            reasons[r["verdict"]["main_reason"]] = reasons.get(r["verdict"]["main_reason"], 0) + 1
    privacy = [r["file"] for r in records if (r["facts"] or {}).get("faces_or_screens_visible")]
    lines = [
        "# Clip Judge — dataset report", "",
        f"Generated {s['generated_at']} · decider: **{s['decider']}** · routing threshold {threshold:.2f}", "",
        "## Verdicts", "",
        "| routed to | clips |", "|---|---|",
        *[f"| {k} | {s[k]} |" for k in ("keep", "fix", "discard", "human_review")],
        f"| **total** | **{s['total']}** |", "",
        f"- Usable minutes (routed to keep): **{s['usable_minutes']}** of {s['total_minutes']}",
        f"- Agreement with hand labels: **{s['agreement_with_labels']}**",
    ]
    if s["baseline_seconds_total"]:
        lines.append(f"- Baseline (Gemini alone) agreement: {s['baseline_agreement_with_labels']}; "
                     f"decision time {s['baseline_seconds_total']} s vs {s['decider_seconds_total']} s")
    lines += ["", "## Reject / fix reasons", ""]
    lines += ([f"- {k}: {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])]
              or ["- none"])
    lines += ["", "## Privacy flags", ""]
    lines += [f"- {p}" for p in privacy] or ["- none flagged"
                                             + (" (no fact sheets available)" if not any(r["facts"] for r in records) else "")]
    lines += ["", "## Human review queue", ""]
    lines += [f"- {r['file']} — {r['verdict']['choice']} at {r['verdict']['confidence']:.0%} "
              f"({r['verdict']['main_reason']})" for r in records
              if r["verdict"]["routed_to"] == "human_review"] or ["- empty"]
    if s["failed"]:
        lines += ["", "## Failed clips (skipped)", ""]
        lines += [f"- {f['file']}: {f['error']}" for f in s["failed"]]
    lines += ["", "## Clips", "", "| clip | routed | verdict | conf | reason | label |", "|---|---|---|---|---|---|"]
    lines += [f"| [{r['file']}](cards/{Path(r['file']).stem}.md) | {r['verdict']['routed_to']} | "
              f"{r['verdict']['choice']} | {r['verdict']['confidence']:.2f} | {r['verdict']['main_reason']} | "
              f"{r.get('label') or '—'} |" for r in records]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- sim replay

def _render_sims(records: list[dict], out_dir: Path) -> None:
    """MuJoCo replay of each kept clip's steps -> out/sim/<clip>.mp4 (skips if already rendered)."""
    try:
        from sim import replay
    except Exception as e:                      # mujoco missing: sim is optional
        log.warning("sim unavailable: %s", e)
        return
    for rec in records:
        if rec["verdict"]["routed_to"] != "keep":
            continue
        steps = (rec.get("facts") or {}).get("steps") or replay.DEFAULT_STEPS
        mp4 = out_dir / "sim" / f"{Path(rec['file']).stem}.mp4"
        meta = out_dir / "sim" / f"{Path(rec['file']).stem}.json"
        key = json.dumps(steps, sort_keys=True)
        if mp4.exists() and meta.exists() and json.loads(meta.read_text()).get("key") == key:
            info = json.loads(meta.read_text())
        else:
            try:
                info = replay.render(steps, mp4, title=f"Sim replay - {rec['file']}")
            except Exception as e:
                log.exception("sim render failed for %s", rec["file"])
                report_error(e, rec["file"])
                continue
            info["key"] = key
            meta.write_text(json.dumps(info))
        rec["sim"] = {"video": _rel(mp4), "seconds": info["seconds"],
                      "apple_on_plate": info["apple_on_plate"],
                      "steps_source": "gemini" if (rec.get("facts") or {}).get("steps") else "default"}
        log.info("sim replay %s -> %s", rec["file"], rec["sim"]["video"])


# ---------------------------------------------------------------- run

def run(clips_dir: str | Path, out_dir: str | Path = "out", labels: str | Path | None = "labels.csv",
        decider: str = "auto", threshold: float = THRESHOLD, baseline: bool = False,
        facts_fn: Optional[Callable[[Path, float], Optional[dict]]] = None,
        decide_fn: Optional[Callable[[dict], dict]] = None, sim: bool = False) -> dict:
    clips_dir, out_dir = Path(clips_dir), Path(out_dir)
    cards_dir, cache_dir = out_dir / "cards", out_dir / "cache"
    cards_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if decide_fn is not None:
        fixed = decider if decider != "auto" else "custom"
        decider_name = lambda: fixed
    else:
        decider_name, decide_fn = pick_decider(decider)
    facts_fn = facts_fn or default_facts
    if facts_fn is default_facts and not os.environ.get("GEMINI_API_KEY"):
        log.warning("GEMINI_API_KEY not set -> fact sheets skipped, verdicts use local checks only")

    clips = sorted(p for p in clips_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    label_map = load_labels(Path(labels) if labels else None, clips if labels else None)
    records, failed = [], []
    import sentry_sdk
    txn = sentry_sdk.start_transaction(op="pipeline.run", name="clip-judge run")
    txn.__enter__()
    for i, path in enumerate(clips, 1):
        try:
            with sentry_sdk.start_span(op="clip", name=path.name):
                    rec = process_clip(path, cache_dir=cache_dir, decider_name=decider_name,
                                   decide_fn=decide_fn, facts_fn=facts_fn,
                                   threshold=threshold, baseline=baseline)
        except NotImplementedError:
            raise  # misconfiguration (forced system_one), not a bad clip
        except Exception as e:
            log.exception("clip %s failed — skipped", path.name)
            report_error(e, path.name)
            failed.append({"file": path.name, "error": f"{type(e).__name__}: {e}"})
            continue
        rec["label"] = label_map.get(path.name)
        rec["card"] = _rel(cards_dir / f"{path.stem}.md")
        write_card(rec, cards_dir, threshold)
        records.append(rec)
        v = rec["verdict"]
        log.info("[%d/%d] %-28s %-12s %s %.2f (%s)", i, len(clips), path.name,
                 v["routed_to"], v["choice"], v["confidence"], v["main_reason"])
        sentry_sdk.metrics.count("clip.routed", 1, attributes={"route": v["routed_to"], "reason": v["main_reason"]})
        sentry_sdk.metrics.distribution("clip.confidence", v["confidence"])
    txn.__exit__(None, None, None)

    if sim:
        _render_sims(records, out_dir)
    summary = summarise(records, failed, decider_name())
    results = {"summary": summary, "clips": records}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_report(summary, records, out_dir / "report.md", threshold)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="judge.pipeline", description=__doc__.split("\n")[0])
    ap.add_argument("clips", nargs="?", default="clips/")
    ap.add_argument("--out", default="out")
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--decider", choices=["auto", "system_one", "rules"], default="auto")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--baseline", action="store_true", help="also run Gemini-only baseline")
    ap.add_argument("--no-sim", action="store_true", help="skip MuJoCo replays of kept clips")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    init_sentry()
    res = run(a.clips, a.out, a.labels, a.decider, a.threshold, a.baseline, sim=not a.no_sim)
    s = res["summary"]
    print(f"\n{s['total']} clips: keep {s['keep']} · fix {s['fix']} · discard {s['discard']} · "
          f"human_review {s['human_review']} · failed {len(s['failed'])} · "
          f"agreement {s['agreement_with_labels']} · decider {s['decider']}")
    print(f"wrote {a.out}/results.json, {a.out}/report.md, {a.out}/cards/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
