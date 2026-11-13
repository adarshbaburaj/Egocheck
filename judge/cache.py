"""Disk cache for API responses (and local checks), keyed by content hash.

out/cache/<file_sha256[:16]>.<kind>[.<extra_sha[:12]>].json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

CACHE_DIR = Path("out/cache")


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def cached(key: str, kind: str, compute: Callable[[], Any], extra: Any = None,
           cache_dir: Path | None = None) -> tuple[Any, bool]:
    """Return (value, was_cached). Values must be JSON-serialisable."""
    d = Path(cache_dir or CACHE_DIR)
    d.mkdir(parents=True, exist_ok=True)
    name = f"{key[:16]}.{kind}"
    if extra is not None:
        name += "." + text_hash(extra)[:12]
    p = d / f"{name}.json"
    if p.exists():
        try:
            return json.loads(p.read_text()), True
        except json.JSONDecodeError:
            p.unlink()
    value = compute()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=1))
    tmp.replace(p)
    return value, False
