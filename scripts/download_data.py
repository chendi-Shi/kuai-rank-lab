"""Download the official archive, verify its published MD5 and safely extract it."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request

URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
CONTENT_URL = "https://zenodo.org/api/records/10439422/files/KuaiRand-Pure.tar.gz/content"
MD5 = "0820331067a3784d9691136f772b35a7"
EXPECTED_BYTES = 47432272


def digest(path):
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch_range(start, end):
    request = urllib.request.Request(CONTENT_URL, headers={
        "User-Agent": "KuaiRankLab/0.1", "Range": f"bytes={start}-{end}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                expected_range = f"bytes {start}-{end}/{EXPECTED_BYTES}"
                if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                    raise ValueError("Server did not return the requested byte range")
                try:
                    block = response.read()
                except http.client.IncompleteRead as error:
                    block = error.partial
                if not 0 < len(block) <= end - start + 1:
                    raise IOError("Invalid range response length")
            return block
        except (OSError, ValueError) as error:
            print(f"Range {start} retry {attempt + 1}: {type(error).__name__}", flush=True)
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def download(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "KuaiRand-Pure.tar.gz"
    if not archive.exists() or digest(archive) != MD5:
        temporary = archive.with_suffix(".part")
        count = temporary.stat().st_size if temporary.exists() else 0
        if count > EXPECTED_BYTES:
            raise ValueError("Oversized partial file; inspect it before retrying")
        # Requests are independent reads; archive writes remain strictly ordered.
        with ThreadPoolExecutor(max_workers=4) as pool:
            while count < EXPECTED_BYTES:
                step = 512 * 1024
                ranges = [(start, min(start + step - 1, EXPECTED_BYTES - 1))
                          for start in range(count, min(count + 4 * step, EXPECTED_BYTES), step)]
                futures = [pool.submit(fetch_range, start, end) for start, end in ranges]
                for (start, end), future in zip(ranges, futures):
                    block = future.result()
                    if start != count:
                        raise ValueError("Noncontiguous archive write")
                    with temporary.open("ab") as f:
                        f.write(block)
                    count += len(block)
                    print(f"Downloaded {count / 1024**2:.1f}/{EXPECTED_BYTES / 1024**2:.1f} MiB", flush=True)
                    if len(block) != end - start + 1:
                        # A short response is a contiguous prefix. Drain/discard
                        # other ranges, then resume at the exact missing byte.
                        for pending in futures:
                            pending.result()
                        break
        if digest(temporary) != MD5:
            raise ValueError("Official MD5 mismatch; refusing extraction")
        temporary.replace(archive)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            if not target.is_relative_to(root) or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe archive member: {member.name}")
        # Do not import Unix archive ownership/modes into Windows ACLs. Files
        # inherit the destination permissions, which also works in sandboxes.
        for member in members:
            target = (root / member.name).resolve()
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
    manifest = {"url": URL, "download_endpoint": CONTENT_URL, "md5": digest(archive), "bytes": archive.stat().st_size,
                "source": "https://github.com/chongminggao/KuaiRand", "license": "CC BY-SA 4.0"}
    (root / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw")
    download(parser.parse_args().out)
