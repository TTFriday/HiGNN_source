"""Record local input hashes and optionally check AtMAD HTTP metadata.

No raw files are downloaded or redistributed. The audit date is not the original
acquisition date. File-size equality is not proof of byte-level identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

FILES = (
    "ciseqtl.xls", "transeqtl.xls", "environmenteqtl.xlsx", "gwaseqtl.csv",
    "emqtl.csv", "epathway.csv", "phenotypepathway.csv", "gwas.csv",
    "twas.csv", "ewas.csv",
)
BASE = "http://119.3.41.228/atmad/downloads/"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(raw_dir, check_remote=False):
    raw_dir = Path(raw_dir)
    entries = []
    for name in FILES:
        path = raw_dir / name
        item = {"filename": name, "source_url": BASE + name,
                "local_present": path.is_file()}
        if path.is_file():
            item.update(local_bytes=path.stat().st_size, local_sha256=sha256(path))
        if check_remote:
            try:
                with urlopen(Request(BASE + name, method="HEAD"), timeout=30) as response:
                    length = response.headers.get("Content-Length")
                    item.update(http_status=response.status,
                                remote_bytes=int(length) if length else None,
                                remote_last_modified=response.headers.get("Last-Modified"))
                    if path.is_file() and length:
                        item["same_byte_length"] = path.stat().st_size == int(length)
            except Exception as exc:
                item["remote_check_error"] = type(exc).__name__ + ": " + str(exc)
        entries.append(item)
    return {
        "source": "AtMAD", "source_doi": "10.1093/nar/gkaa1042",
        "download_page": "http://119.3.41.228/atmad/download.php",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_download_date": None,
        "repository_summary_label": "June 2020 (website summary, not verified local snapshot)",
        "redistribution_permission": "Not verified; obtain source files directly from AtMAD",
        "exact_remote_snapshot_match": "Not established; remote content hashes not compared",
        "files": entries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-remote", action="store_true")
    args = parser.parse_args()
    if not args.raw_dir.is_dir():
        parser.error("--raw-dir must be an existing directory")
    if not args.output.parent.is_dir():
        parser.error("output parent directory must already exist")
    result = audit(args.raw_dir, args.check_remote)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Input manifest written:", args.output)
    missing = [item["filename"] for item in result["files"] if not item["local_present"]]
    if missing:
        print("Missing inputs:", ", ".join(missing))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
