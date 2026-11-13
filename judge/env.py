"""Load KEY=value lines from the project's .env into os.environ.

Values already set in the real environment win. Set CLIP_JUDGE_NO_DOTENV=1 to skip
(tests do this so they never hit live APIs).
"""
import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load(path: Path = ENV_FILE) -> None:
    if os.environ.get("CLIP_JUDGE_NO_DOTENV") or not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if v and k not in os.environ:
            os.environ[k] = v
