"""Download and verify large environment files from a published GitHub Release."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", required=True, help="GitHub OWNER/REPOSITORY")
    p.add_argument("--site", choices=("site01", "site06", "site07"))
    p.add_argument("--tag", help="Override the release tag in the manifest")
    args = p.parse_args()
    components = args.repository.split("/")
    if len(components) != 2 or not all(s and all(c.isalnum() or c in "-_." for c in s) for s in components):
        p.error("--repository must be OWNER/REPOSITORY")
    manifest = json.loads((ROOT / "manifests/release_assets.json").read_text(encoding="utf-8"))
    tag = args.tag or manifest["release_tag"]
    for asset in manifest["assets"]:
        if args.site and Path(asset["path"]).parent.name != args.site:
            continue
        target = ROOT / asset["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size == asset["bytes"] and sha(target) == asset["sha256"]:
                print(f"Already verified: {asset['path']}")
                continue
            raise RuntimeError(f"Existing file differs from manifest; move it aside before retrying: {target}")
        url = f"https://github.com/{args.repository}/releases/download/{tag}/{asset['asset_name']}"
        compressed = asset.get("compression") == "gzip"
        downloaded = target.with_suffix(target.suffix + (".gz.part" if compressed else ".part"))
        print(f"Downloading {asset['asset_name']} ({asset['bytes']:,} bytes)", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=120) as response, downloaded.open("wb") as out:
                for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    out.write(block)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Release asset unavailable: {url}. Confirm repository/tag and publication status.") from exc
        if compressed:
            if downloaded.stat().st_size != asset["asset_bytes"] or sha(downloaded) != asset["asset_sha256"]:
                raise RuntimeError(f"Downloaded archive failed verification: {downloaded}")
            temporary = target.with_suffix(target.suffix + ".part")
            with gzip.open(downloaded, "rb") as src, temporary.open("wb") as out:
                shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
            downloaded.unlink()
        else:
            temporary = downloaded
        if temporary.stat().st_size != asset["bytes"] or sha(temporary) != asset["sha256"]:
            raise RuntimeError(f"Downloaded file failed verification: {temporary}")
        temporary.replace(target)
        print(f"Verified: {asset['path']}")


if __name__ == "__main__":
    main()
