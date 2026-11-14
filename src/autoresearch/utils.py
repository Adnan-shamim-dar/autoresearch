from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Canonical serialisation and hashing
# ---------------------------------------------------------------------------

def json_safe(obj: Any) -> Any:
    """Convert numpy/pandas scalars and non-finite floats into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def short_hash(value: str, n: int = 10) -> str:
    return value[:n]


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(json_safe(obj), indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(canonical_json(obj) + "\n")


# ---------------------------------------------------------------------------
# Hash chains
# ---------------------------------------------------------------------------

def chain_hash(prev_hash: str, payload: Any) -> str:
    return sha256_text(prev_hash + canonical_json(payload))


def verify_chain(entries: Iterable[dict], genesis: str = "GENESIS") -> tuple[bool, str | None]:
    prev = genesis
    for i, entry in enumerate(entries):
        body = dict(entry)
        claimed = body.pop("hash", None)
        if claimed is None:
            return False, f"entry {i} has no hash"
        expected = chain_hash(prev, body)
        if claimed != expected:
            return False, f"entry {i} hash mismatch"
        prev = claimed
    return True, None


# ---------------------------------------------------------------------------
# Seeded randomness and small helpers
# ---------------------------------------------------------------------------

def stable_seed(*parts: Any, base: int = 0) -> int:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(f"{base}|{text}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def rng_for(*parts: Any, base: int = 0) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*parts, base=base))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    clean = {k: float(v) for k, v in weights.items() if abs(float(v)) > 1e-12}
    total = sum(abs(v) for v in clean.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in clean.items()}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_id(prefix: str = "ar") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def fmt_pct(x: float | None, digits: int = 1) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:.{digits}f}%"


def fmt_num(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"
