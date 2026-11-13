"""Music for the EgoCheck film with Google Lyria on Vertex AI (same project/credits as Gemini).

No music under the story half by choice (narration + SFX only). Music enters with EgoCheck:
  theme.mp3  the EgoCheck theme, from the logo reveal (38.0 s) under the product scenes
  outro.mp3  a short hype hit under the end card (71.3 s)
Lyria returns ~33 s WAV clips; raw clips are cached as <name>.wav, cuts are written as .mp3.
To use your own track instead (e.g. from the YouTube Audio Library), drop it in as theme.wav
or hype.wav and rerun: only the cut step runs.
Usage: .venv/bin/python scripts/make_music.py
"""
import base64, json, os, pathlib, subprocess, sys, urllib.request
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from judge.env import load
load()
import google.auth, google.auth.transport.requests

PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
MODEL = os.environ.get("LYRIA_MODEL", "lyria-002")
OUT = pathlib.Path(__file__).resolve().parents[1] / "video/egocheck/assets/music"
NEG = "vocals, singing, lyrics, speech, harsh distortion"
# Lyria blocks prompts that sound too close to existing music ("recitation checks"); reword if that happens.
TRACKS = {
    "theme": ("Innovative, optimistic modern tech underscore: bright plucked synth arpeggios, airy pads, light rhythmic "
              "pulse, a sense of discovery and quiet confidence, clean and minimal, 118 bpm, instrumental", 1207),
    "hype": ("Energetic electronic finale: tight punchy kick and snare, pulsing synth bass, shimmering arpeggiated synths, "
             "triumphant and full energy throughout, 126 bpm, instrumental", 314),
}
# name -> (raw clip, seconds kept, fade-in, fade-out, loudness target)
CUTS = {"theme": ("theme", 32.7, 0.25, 1.2, -18), "outro": ("hype", 7.0, 0.12, 2.2, -16)}


def token():
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


def generate(name, path):
    prompt, seed = TRACKS[name]
    url = (f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/us-central1/"
           f"publishers/google/models/{MODEL}:predict")
    body = {"instances": [{"prompt": prompt, "negative_prompt": NEG, "seed": seed}], "parameters": {}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {token()}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        pred = json.load(r)["predictions"][0]
    path.write_bytes(base64.b64decode(pred.get("bytesBase64Encoded") or pred.get("audioContent")))


def cut(src, dst, keep, fin, fout, lufs):
    ff = str(OUT.parents[2] / "bin/ffmpeg")
    af = (f"loudnorm=I={lufs}:TP=-1.5,aresample=48000,atrim=0:{keep},asetpts=N/SR/TB,"
          f"afade=t=in:d={fin},afade=t=out:st={keep - fout}:d={fout}")
    subprocess.run([ff, "-y", "-v", "error", "-i", str(src), "-af", af, "-c:a", "libmp3lame", "-b:a", "192k", str(dst)], check=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for out, (raw, keep, fin, fout, lufs) in CUTS.items():
        src = OUT / f"{raw}.wav"
        if not src.exists():
            generate(raw, src)
        cut(src, OUT / f"{out}.mp3", keep, fin, fout, lufs)
        print(f"{out}.mp3")
