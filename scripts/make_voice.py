"""Narration + music for the EgoCheck film via ElevenLabs. Key: ELEVENLABS_API_KEY in .env."""
import json, os, sys, urllib.request, pathlib, concurrent.futures as cf
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from judge.env import load as load_dotenv  # noqa
load_dotenv()
KEY = os.environ["ELEVENLABS_API_KEY"]
VOICE = os.environ.get("VO_VOICE", "cgSgspJ2msm6clMCkdW9")  # premade "Jessica" (library voices need a paid plan)
OUT = pathlib.Path(__file__).resolve().parents[1] / "video/egocheck/assets/vo"
LINES = {
  "l01": "This is Elon. He wants to build a robot brain.",
  "l02": "He raises five hundred thousand dollars.",
  "l03": "So he buys data. Egocentric video. UMI. Lots of it.",
  "l04": "He can already see it. His robots. His company.",
  "l05": "Then... he actually watches the data.",
  "l06": "Blurry. Unfinished. A laptop screen in the shot.",
  "l07": "Garbage in. Garbage out.",
  "l08": "He lost his dream. His investor's money. And the one thing he can't buy back. Time.",
  "l09": "Meet EgoCheck.",
  "l10": "EgoCheck checks every clip before a robot learns from it. Keep it, fix it, or throw it out.",
  "l11": "OpenCV measures it. Gemini watches it. TypeSafe's System One judges it.",
  "l12": "System One answers in a third of a second, for a hundredth of a cent.",
  "l13": "Five thousand hours, checked for about five thousand dollars. Forty-five thousand dollars of junk, caught before training.",
  "l14": "Only good data reaches the robot.",
  "l15": "EgoCheck. Clean data today. Better robots tomorrow.",
}
def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"xi-api-key": KEY, "Content-Type": "application/json", "Accept": "audio/mpeg"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()
def tts(k):
    p = OUT / f"{k}.mp3"
    if p.exists(): return k, "cached"
    a = post(f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE}?output_format=mp3_44100_128",
             {"text": LINES[k], "model_id": "eleven_multilingual_v2",
              "voice_settings": {"stability": 0.4, "similarity_boost": 0.8, "style": 0.35, "use_speaker_boost": True}})
    p.write_bytes(a); return k, len(a)
if __name__ == "__main__":
    which = sys.argv[1:] or list(LINES)
    if which == ["music"]:
        a = post("https://api.elevenlabs.io/v1/music", {"prompt": "Playful light comedic underscore for an animated stick-figure explainer: pizzicato strings, soft marimba, light percussion; turns hopeful and confident in the second half, warm uplifting finish. Instrumental, no vocals.", "music_length_ms": 75000})
        (OUT.parent / "music.mp3").write_bytes(a); print("music", len(a)); sys.exit()
    with cf.ThreadPoolExecutor(2) as ex:
        for k, n in ex.map(tts, which): print(k, n)
