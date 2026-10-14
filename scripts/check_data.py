"""Check archived input files without importing Gym, Ray, or PyTorch."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sha256", action="store_true", help="Read entire files and compare SHA256 hashes")
    p.add_argument("--small-only", action="store_true", help="Check only files stored in normal Git")
    args = p.parse_args()
    records = json.loads((ROOT / "manifests/source_files.json").read_text(encoding="utf-8"))
    failures, checked = [], 0
    for rec in records:
        if args.small_only and rec["bytes"] > 100 * 1024 * 1024:
            continue
        path = ROOT / rec["path"]
        if not path.is_file():
            failures.append(f"Missing: {rec['path']}")
        elif path.stat().st_size != rec["bytes"]:
            failures.append(f"Size mismatch: {rec['path']}")
        elif args.sha256 and digest(path) != rec["sha256"]:
            failures.append(f"SHA256 mismatch: {rec['path']}")
        checked += 1
    print(f"Checked {checked} archived files; {len(failures)} failures.")
    for error in failures:
        print(error)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
