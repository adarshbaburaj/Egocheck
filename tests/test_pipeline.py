"""Acceptance tests 1-5 from SPEC.md, plus guards for the rules in the spec.

Runs fully offline: Gemini is replaced by a fake fact sheet where needed and
the decider is the rules fallback (System One is not wired yet).
"""
from __future__ import annotations

import functools
import http.server
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ["CLIP_JUDGE_NO_DOTENV"] = "1"
sys.path.insert(0, str(ROOT))

from judge import gemini_facts, pipeline, rules_decider, system_one  # noqa: E402

_spec = importlib.util.spec_from_file_location("make_demo_clips", ROOT / "scripts/make_demo_clips.py")
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)

GOOD_FACTS = {
    "task_label": "pick up cup and place on plate",
    "hands_visible_ratio": 0.95, "object_visible_ratio": 0.95,
    "task_completed": True, "single_task": True, "faces_or_screens_visible": False,
    "steps": [{"t_start": 0, "t_end": 1, "action": "move_to", "object": "cup"},
              {"t_start": 1, "t_end": 1.4, "action": "grasp", "object": "cup"},
              {"t_start": 1.4, "t_end": 3, "action": "place", "object": "plate"},
              {"t_start": 3, "t_end": 4, "action": "release", "object": "cup"}],
    "notes": "clean single pick-and-place",
}


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    """A throwaway project tree: clips/, labels.csv, dashboard/, out/."""
    root = tmp_path_factory.mktemp("proj")
    clips = root / "clips"
    clips.mkdir()
    demo.make(clips / "good.mp4", "good", 1)
    demo.make(clips / "blurry.mp4", "blurry", 2)
    demo.make(clips / "shaky.mp4", "shaky", 3)
    (root / "labels.csv").write_text("clip,my_verdict\ngood.mp4,keep\nblurry.mp4,discard\nshaky.mp4,fix\n")
    (root / "dashboard").mkdir()
    shutil.copy(ROOT / "dashboard/index.html", root / "dashboard/index.html")
    return root


@pytest.fixture(scope="session")
def cli_run(project):
    """Test 1: the documented command, run as a subprocess, no API keys."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("GEMINI_API_KEY", "SYSTEM_ONE_API_KEY", "SENTRY_DSN")}
    env["PYTHONPATH"] = str(ROOT)
    env["CLIP_JUDGE_NO_DOTENV"] = "1"
    proc = subprocess.run([sys.executable, "-m", "judge.pipeline", "clips/"],
                          cwd=project, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    return json.loads((project / "out/results.json").read_text())


# ---------------------------------------------------------------- 1
def test_1_pipeline_writes_outputs(project, cli_run):
    out = project / "out"
    assert (out / "results.json").exists()
    assert (out / "report.md").read_text().startswith("# Clip Judge")
    for name in ("good", "blurry", "shaky"):
        assert (out / "cards" / f"{name}.json").exists()
        assert (out / "cards" / f"{name}.md").exists()
    s = cli_run["summary"]
    assert s["total"] == 3 and s["agreement_with_labels"].endswith("/3")
    assert s["decider"] == "rules_fallback"     # honest about System One not being wired


# ---------------------------------------------------------------- 2
def test_2_every_clip_has_verdict_probs_and_route(cli_run):
    assert len(cli_run["clips"]) == 3
    for c in cli_run["clips"]:
        v = c["verdict"]
        assert v["choice"] in system_one.VERDICTS
        assert set(v["probs"]) == set(system_one.VERDICTS)
        assert sum(v["probs"].values()) == pytest.approx(1, abs=0.01)
        assert sum(v["reason_probs"].values()) == pytest.approx(1, abs=0.01)
        assert v["main_reason"] in system_one.REASONS
        assert v["routed_to"] in ("keep", "fix", "discard", "human_review")


# ---------------------------------------------------------------- 3
def test_3_low_confidence_goes_to_human_review(project, tmp_path):
    assert pipeline.route({"keep": 0.79, "fix": 0.11, "discard": 0.10}) == "human_review"
    assert pipeline.route({"keep": 0.80, "fix": 0.10, "discard": 0.10}) == "keep"

    def unsure(state):
        return {"verdict": {"choice": "keep", "probs": {"keep": 0.6, "fix": 0.3, "discard": 0.1}},
                "main_reason": {"choice": "good",
                                "probs": {r: (0.92 if r == "good" else 0.01) for r in system_one.REASONS}}}

    res = pipeline.run(project / "clips", tmp_path / "out", labels=None,
                       facts_fn=lambda p, d: GOOD_FACTS, decide_fn=unsure)
    assert {c["verdict"]["routed_to"] for c in res["clips"]} == {"human_review"}
    assert res["summary"]["human_review"] == 3


# ---------------------------------------------------------------- 4
def test_4_blurry_clip_never_kept(project, cli_run, tmp_path):
    blurry = next(c for c in cli_run["clips"] if c["file"] == "blurry.mp4")
    assert blurry["verdict"]["choice"] != "keep"
    assert blurry["verdict"]["routed_to"] != "keep"
    # even when Gemini says the content is perfect
    res = pipeline.run(project / "clips", tmp_path / "out", labels=None,
                       decider="rules", facts_fn=lambda p, d: GOOD_FACTS)
    blurry = next(c for c in res["clips"] if c["file"] == "blurry.mp4")
    assert blurry["verdict"]["choice"] != "keep"
    assert blurry["verdict"]["main_reason"] == "blurry"
    good = next(c for c in res["clips"] if c["file"] == "good.mp4")
    assert good["verdict"]["routed_to"] == "keep"


# ---------------------------------------------------------------- 5
def test_5_dashboard_served_with_every_clip(project, cli_run):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(project))
    handler.log_message = lambda *a: None
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{port}"
        html = urllib.request.urlopen(f"{base}/dashboard/index.html").read().decode()
        assert "../out/results.json" in html
        data = json.loads(urllib.request.urlopen(f"{base}/out/results.json").read())
        assert len(data["clips"]) == 3
        for c in data["clips"]:   # every video and card the page links to is reachable
            assert urllib.request.urlopen(f"{base}/{c['src']}").status == 200
            assert urllib.request.urlopen(f"{base}/{c['card']}").status == 200
    finally:
        srv.shutdown()


def test_5b_dashboard_renders_in_browser(project, cli_run):
    """Real DOM check if Playwright + Chromium are installed; skipped otherwise."""
    pw = pytest.importorskip("playwright.sync_api")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(project))
    handler.log_message = lambda *a: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pw.sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as e:
                pytest.skip(f"chromium not available: {e}")
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{srv.server_address[1]}/dashboard/index.html")
            page.wait_for_selector(".clip-card")
            cards = page.query_selector_all(".clip-card")
            assert len(cards) == 3
            for c in cli_run["clips"]:
                card = page.query_selector(f'.clip-card[data-file="{c["file"]}"]')
                assert card and c["verdict"]["routed_to"].replace("_", " ") in card.inner_text().lower()
            browser.close()
    finally:
        srv.shutdown()


# ---------------------------------------------------------------- spec rules
def test_system_one_not_invented():
    with pytest.raises(NotImplementedError):
        system_one.decide({"local": {}, "facts": {}})


def test_failed_clip_is_skipped_not_fatal(project, tmp_path):
    clips = tmp_path / "clips"
    shutil.copytree(project / "clips", clips)
    (clips / "broken.mp4").write_bytes(b"not a video")
    res = pipeline.run(clips, tmp_path / "out", labels=None, facts_fn=lambda p, d: GOOD_FACTS)
    assert res["summary"]["total"] == 3
    assert [f["file"] for f in res["summary"]["failed"]] == ["broken.mp4"]


def test_cache_makes_rerun_skip_api(project, tmp_path):
    calls = []

    def facts(p, d):
        calls.append(p.name)
        return GOOD_FACTS

    pipeline.run(project / "clips", tmp_path / "out", labels=None, facts_fn=facts)
    pipeline.run(project / "clips", tmp_path / "out", labels=None, facts_fn=facts)
    assert len(calls) == 3   # second run served from out/cache/


def test_fact_sheet_validation():
    assert gemini_facts.validate(GOOD_FACTS, 4.0) == []
    bad = dict(GOOD_FACTS, hands_visible_ratio=1.4,
               steps=[{"t_start": 2, "t_end": 1, "action": "wave", "object": "cup"}])
    errs = gemini_facts.validate(bad, 4.0)
    assert any("hands_visible_ratio" in e for e in errs)
    assert any("action" in e for e in errs)
    assert any("t_start" in e for e in errs)
    assert gemini_facts.normalise({"task_completed": "unclear"})["task_completed"] == "unclear"
    assert gemini_facts.normalise({"task_completed": "true"})["task_completed"] is True


def test_gemini_retries_once_then_succeeds(monkeypatch):
    answers = iter(['{"task_label": ""}', json.dumps(dict(GOOD_FACTS, task_completed="true"))])
    prompts = []

    class FakeFiles:
        def delete(self, name): pass

    class FakeClient:
        files = FakeFiles()

    monkeypatch.setattr(gemini_facts, "_client", lambda: FakeClient())
    monkeypatch.setattr(gemini_facts, "upload_video", lambda c, p: type("F", (), {"name": "f"})())
    monkeypatch.setattr(gemini_facts, "generate_json",
                        lambda c, v, prompt, s: (prompts.append(prompt), next(answers))[1])
    out = gemini_facts.get_facts("x.mp4", 4.0)
    assert out["task_completed"] is True and len(prompts) == 2
    assert "rejected" in prompts[1]


def test_gemini_gives_up_after_one_retry(monkeypatch):
    class FakeClient:
        files = type("FF", (), {"delete": lambda self, name: None})()

    monkeypatch.setattr(gemini_facts, "_client", lambda: FakeClient())
    monkeypatch.setattr(gemini_facts, "upload_video", lambda c, p: type("F", (), {"name": "f"})())
    monkeypatch.setattr(gemini_facts, "generate_json", lambda *a: "{}")
    with pytest.raises(gemini_facts.FactSheetError):
        gemini_facts.get_facts("x.mp4", 4.0)


def test_rules_use_fact_sheet():
    local = {"blur_score": 400, "brightness_mean": 120, "dark_frame_ratio": 0, "shake_score": 1}
    d = rules_decider.decide({"local": local, "facts": dict(GOOD_FACTS, hands_visible_ratio=0.1)})
    assert d["verdict"]["choice"] == "discard" and d["main_reason"]["choice"] == "hands_out_of_frame"
    d = rules_decider.decide({"local": local, "facts": dict(GOOD_FACTS, faces_or_screens_visible=True)})
    assert d["main_reason"]["choice"] == "privacy" and d["verdict"]["choice"] != "keep"
