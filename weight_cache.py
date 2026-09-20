from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def fingerprint(path: Path) -> list:
    stat = path.stat()
    return [str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def verified_digest(path: Path, hasher, directory: Path) -> str:
    path = Path(path)
    if path.suffix != ".safetensors":
        return hasher(path)
    path = path.resolve(strict=True)
    before = fingerprint(path)
    key = hashlib.sha256(str(path).encode()).hexdigest()
    cache = directory / (key + ".json")
    try:
        entry = json.loads(cache.read_text())
        digest = entry.get("sha256", "")
        if entry.get("file") == before and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
            return digest
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    digest = hasher(path)
    if before != fingerprint(path):
        raise RuntimeError("weights changed during integrity verification")
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as temporary:
            json.dump({"file": before, "sha256": digest}, temporary)
        os.replace(temporary.name, cache)
    except OSError:
        pass
    return digest


def install_verified_hash_cache() -> None:
    import yue2.storage as storage

    if getattr(storage.sha256_file, "yue_verified_cache", False):
        return
    original = storage.sha256_file
    directory = Path(os.getenv("YUE_WEIGHT_HASH_CACHE", "~/.cache/yue2/verified-weights")).expanduser()

    def cached(path: Path) -> str:
        return verified_digest(path, original, directory)

    cached.yue_verified_cache = True
    storage.sha256_file = cached
