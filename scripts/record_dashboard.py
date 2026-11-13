"""Screen-record the dashboard (headless Chromium) for the demo video.

    python scripts/record_dashboard.py [--out video/demo/assets/dashboard.mp4] [--clip <file to open>]

Serves the project root, plays a short scripted tour (load, filter, open a clip,
scroll to the robot replay), saves WebM, converts to H.264 MP4 with video/bin/ffmpeg.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = ROOT / "video" / "bin" / "ffmpeg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "video/demo/assets/dashboard.mp4"))
    ap.add_argument("--clip", help="clip file name to open (default: first kept clip with a sim replay)")
    a = ap.parse_args()
    results = json.loads((ROOT / "out/results.json").read_text())
    clips = results["clips"]
    pick = a.clip or next((c["file"] for c in clips if c.get("sim")), None) or next(
        (c["file"] for c in clips if c["verdict"]["routed_to"] == "keep"), clips[0]["file"])
    bad = next((c["file"] for c in clips if c["verdict"]["routed_to"] in ("discard", "human_review")), None)

    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    h.log_message = lambda *x: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/dashboard/index.html"
    tmp = Path(tempfile.mkdtemp())
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1,
                            record_video_dir=str(tmp), record_video_size={"width": 1920, "height": 1080})
        ctx.add_init_script("try{localStorage.clear()}catch(e){}")
        pg = ctx.new_page()
        pg.goto(url)
        pg.wait_for_selector(".clip-card")
        pg.wait_for_timeout(2600)                                  # count-up + bars
        for r in ("discard", "human_review", "keep"):              # tour the routes
            pg.click(f'#legend [data-f="{r}"]'); pg.wait_for_timeout(1300)
        pg.click('#legend [data-f="keep"]'); pg.wait_for_timeout(900)   # back to all
        if bad:
            pg.hover(f'.clip-card[data-file="{bad}"]'); pg.wait_for_timeout(700)
            pg.click(f'.clip-card[data-file="{bad}"]'); pg.wait_for_timeout(2600)
            pg.mouse.wheel(0, 700); pg.wait_for_timeout(1800)
            pg.keyboard.press("Escape"); pg.wait_for_timeout(700)
        pg.hover(f'.clip-card[data-file="{pick}"]'); pg.wait_for_timeout(900)
        pg.click(f'.clip-card[data-file="{pick}"]'); pg.wait_for_timeout(2600)
        body = pg.locator("#dbody")
        for _ in range(6):
            body.evaluate("el => el.scrollBy({top: 260, behavior: 'smooth'})"); pg.wait_for_timeout(900)
        pg.wait_for_timeout(3500)                                  # let the robot replay play
        pg.keyboard.press("Escape"); pg.wait_for_timeout(600)
        pg.click('[data-v="list"]'); pg.wait_for_timeout(2200)
        pg.click('[data-v="grid"]'); pg.wait_for_timeout(1200)
        webm = Path(pg.video.path())
        ctx.close(); b.close()
    srv.shutdown()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", "-i", str(webm), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-preset", "slow", "-crf", "14", "-r", "30", str(out)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", out, "| opened", pick)


if __name__ == "__main__":
    main()
