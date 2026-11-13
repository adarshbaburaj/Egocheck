"""High-resolution stills for the demo video (HyperFrames assets).

    python scripts/capture_shots.py [--out video/demo/assets/shots] [--scale 2]

Dashboard states are captured in headless Chromium at 1920x1080 CSS pixels with
device scale 2 (3840x2160 PNGs). MuJoCo sim stills are rendered at 3840x2160.
Writes shots/index.json describing each file.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dashboard_shots(out: Path, scale: int) -> list[dict]:
    from playwright.sync_api import sync_playwright
    res = json.loads((ROOT / "out/results.json").read_text())
    clips = res["clips"]
    keep = next((c["file"] for c in clips if c.get("sim")), None) or next(
        (c["file"] for c in clips if c["verdict"]["routed_to"] == "keep"), None)
    bad = next((c["file"] for c in clips if c["verdict"]["routed_to"] == "discard"), None)
    review = next((c["file"] for c in clips if c["verdict"]["routed_to"] == "human_review"), None)

    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    h.log_message = lambda *x: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/dashboard/index.html"
    shots = []

    def snap(page, name, what, **kw):
        p = out / f"{name}.png"
        page.screenshot(path=str(p), **kw)
        shots.append({"file": p.name, "what": what})

    with sync_playwright() as pw:
        b = pw.chromium.launch()
        for scheme in ("light", "dark"):
            ctx = b.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=scale,
                                color_scheme=scheme, reduced_motion="reduce")
            ctx.add_init_script("try{localStorage.clear()}catch(e){}")
            pg = ctx.new_page()
            pg.goto(url); pg.wait_for_selector(".clip-card"); pg.wait_for_timeout(1500)
            sfx = "" if scheme == "light" else "_dark"
            snap(pg, f"dashboard_overview{sfx}", f"Dashboard overview ({scheme})")
            if scheme == "dark":
                ctx.close(); continue
            snap(pg, "dashboard_full", "Dashboard, full page", full_page=True)
            for r in ("keep", "discard", "human_review"):
                if pg.locator(f'#legend [data-f="{r}"]').count():
                    pg.click(f'#legend [data-f="{r}"]'); pg.wait_for_timeout(600)
                    snap(pg, f"dashboard_filter_{r}", f"Dashboard filtered to {r}")
                    pg.click(f'#legend [data-f="{r}"]'); pg.wait_for_timeout(400)
            pg.click('[data-v="list"]'); pg.wait_for_timeout(600)
            snap(pg, "dashboard_list", "List view of all clips")
            pg.click('[data-v="grid"]'); pg.wait_for_timeout(400)
            for name, f in (("keep", keep), ("discard", bad), ("review", review)):
                if not f:
                    continue
                pg.click(f'.clip-card[data-file="{f}"]'); pg.wait_for_timeout(1500)
                snap(pg, f"drawer_{name}_top", f"Clip panel: {f} ({name}) - verdict and probabilities")
                drawer = pg.locator("#drawer")
                snap(pg, f"drawer_{name}_panel", f"Clip panel only: {f}", clip=drawer.bounding_box())
                pg.locator("#dbody").evaluate("el => el.scrollTo(0, el.scrollHeight)"); pg.wait_for_timeout(700)
                snap(pg, f"drawer_{name}_checks", f"Clip panel: {f} - checks list", clip=drawer.bounding_box())
                pg.keyboard.press("Escape"); pg.wait_for_timeout(500)
            ctx.close()
        b.close()
    srv.shutdown()
    return shots


def sim_shots(out: Path) -> list[dict]:
    sys.path.insert(0, str(ROOT))
    import cv2
    import mujoco
    from sim import replay
    r = replay.Replayer(replay.DEFAULT_STEPS)
    rend = mujoco.Renderer(r.m, 2160, 3840)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.02, 0.13]; cam.distance = 0.98; cam.azimuth = 205; cam.elevation = -20
    marks = {"move to apple": "sim_reach", "grasp": "sim_grasp", "lift": "sim_lift", "carry": "sim_carry",
             "place": "sim_place", "done": "sim_done"}
    shots, seen = [], set()
    while not r.done:
        r.step()
        lab = r.label
        if lab in marks and lab not in seen and r.d.time > 0.2:
            # capture near the end of each phase
            nxt = [k for k in r.seq if k[3] == lab]
            if nxt and r.d.time >= nxt[-1][0] - 0.05:
                rend.update_scene(r.d, cam)
                p = out / f"{marks[lab]}.png"
                cv2.imwrite(str(p), rend.render()[:, :, ::-1])
                shots.append({"file": p.name, "what": f"MuJoCo Shadow Hand - {lab}"})
                seen.add(lab)
    return shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "video/demo/assets/shots"))
    ap.add_argument("--scale", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    shots = dashboard_shots(out, a.scale) + sim_shots(out)
    (out / "index.json").write_text(json.dumps(shots, indent=1))
    for s in shots:
        print(f"{s['file']:32} {s['what']}")


if __name__ == "__main__":
    main()
