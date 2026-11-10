#!/usr/bin/env python3
"""Apply patches/yue2-accel.patch to the installed, pinned yue2 package.

Idempotent. Run with the interpreter that serves YuE2:
  .venv/bin/python scripts/apply_yue2_patch.py [--check | --revert]
"""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

PATCH = Path(__file__).resolve().parents[1] / "patches/yue2-accel.patch"
MARKER = ".yue-cog-accel"
PRISTINE = {
    "__init__.py": "e35b3471f497934fe5dbe4aa615fc571b82e075b973fef3eebbe994368148ea8",
    "cuda_graph.py": "9c5bce70ce3eb7e465b4a670f1d7cfc97a997119fc8783d3ee02e51c98799c03",
    "nar.py": "1a13a8e54b40229e008a270804a1fb830dfdf806735be484e8ce10171eb4b8d8",
    "pipeline.py": "48eda878c0af5b101a2f48f62c0d12ffb40db91c902b831bab1082be13c4af44",
    "sampling.py": "b680afd5d6b6694b0e8d430b4c1b6d3b5108107672b06500ecc6645c471bf6c4",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_patch(root: Path, *flags: str) -> None:
    subprocess.run(["patch", "-p1", "--fuzz=0", "--no-backup-if-mismatch", "-s", "-d", str(root), "-i", str(PATCH), *flags],
                   check=True)


def main(argv: list[str]) -> int:
    spec = importlib.util.find_spec("yue2")
    if spec is None or spec.origin is None:
        raise SystemExit("yue2 is not installed in this interpreter")
    package = Path(spec.origin).parent
    marker, wanted = package / MARKER, digest(PATCH)
    applied = marker.read_text().strip() if marker.exists() else ""
    if "--check" in argv:
        print("applied" if applied == wanted else "stale" if applied else "pristine", package)
        return 0 if applied == wanted else 1
    if "--revert" in argv:
        if applied != wanted:
            raise SystemExit("installed package does not carry this patch; reinstall the pinned yue2-infer instead")
        run_patch(package.parent, "-R")
        marker.unlink()
        print("reverted", package)
        return 0
    if applied == wanted:
        print("already applied", package)
        return 0
    if applied:
        raise SystemExit("a different yue-cog patch is applied; run the previous --revert or reinstall the pinned yue2-infer")
    changed = [name for name, value in PRISTINE.items() if digest(package / name) != value]
    if changed or (package / "placement.py").exists():
        raise SystemExit(f"yue2 is not the pinned pristine release (differs: {changed or ['placement.py']}); refusing to patch")
    run_patch(package.parent, "--dry-run")
    run_patch(package.parent)
    marker.write_text(wanted + "\n")
    print("applied", package)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
