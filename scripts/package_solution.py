"""Package source, original data, measured results and fitted models."""

import hashlib
import json
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".venv", ".git", "__pycache__", ".pytest_cache"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files = []
    for directory, folders, names in os.walk(ROOT):
        folders[:] = [name for name in folders if name not in EXCLUDED]
        for name in names:
            path = Path(directory) / name
            if name.endswith(".pyc") or name == ".DS_Store":
                continue
            files.append(path)
    files.sort()
    manifest_path = ROOT / "reports/file_manifest.json"
    manifest = {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": digest(path)}
                for path in files if path != manifest_path}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if manifest_path not in files:
        files.append(manifest_path)
    archive = ROOT.parent / "phishing-hackathon-solution.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for path in files:
            target.write(path, Path(ROOT.name) / path.relative_to(ROOT))
    with zipfile.ZipFile(archive) as target:
        assert target.testzip() is None, "Archive CRC failed"
        assert f"{ROOT.name}/submissions/submission_primary.csv" in target.namelist()
        assert f"{ROOT.name}/artifacts/primary.joblib" in target.namelist()
        assert not any("/.venv/" in name for name in target.namelist())
    archive.with_suffix(".zip.sha256").write_text(f"{digest(archive)}  {archive.name}\n", encoding="utf-8")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "files": len(files), "crc_ok": True}))


if __name__ == "__main__":
    main()
