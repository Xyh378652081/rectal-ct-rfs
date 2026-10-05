from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    excluded = {"release_manifest.json", "synthetic_prediction.csv"}
    files = {}
    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or path.name in excluded
            or "__pycache__" in path.parts
            or ".git" in path.parts
        ):
            continue
        files[path.relative_to(ROOT).as_posix()] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    payload = {
        "release": "v1.1.0",
        "analysis_id": "RECTAL_LATEST_CLINICAL_V3_20261001",
        "patient_level_data_included": False,
        "files": files,
    }
    (ROOT / "release_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
