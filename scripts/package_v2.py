"""Package the reproducible v2 solution without research pages or model caches."""

import hashlib
import json
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def included(path):
    relative=path.relative_to(ROOT)
    if any(part in {".venv",".git","__pycache__",".pytest_cache"} for part in relative.parts):
        return False
    if path.name in {".DS_Store"} or path.suffix==".pyc":
        return False
    if relative.parts[0]=="external":
        return path.name in {"SOURCE.md","mendeley_index.sql","mendeley_files.json"}
    if relative.parts[:2]==("experiments","boosting") and path.suffix==".joblib":
        return False
    if relative.parts[:2]==("experiments","astra") and path.suffix==".joblib":
        return path.name=="development_model.joblib"
    if relative.as_posix()=="reports/file_manifest.json":
        return False
    return True


def main():
    files=[]
    for directory,folders,names in os.walk(ROOT):
        folders[:]=[n for n in folders if n not in {".venv",".git","__pycache__",".pytest_cache"}]
        for name in names:
            path=Path(directory)/name
            if included(path):
                files.append(path)
    files.sort()
    manifest_path=ROOT/"reports/v2/file_manifest.json"
    manifest={str(p.relative_to(ROOT)):{"bytes":p.stat().st_size,"sha256":digest(p)} for p in files if p!=manifest_path}
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    if manifest_path not in files:
        files.append(manifest_path)
    archive=ROOT.parent/"phishing-hackathon-solution-v2.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as target:
        for path in files:
            target.write(path,Path(ROOT.name)/path.relative_to(ROOT))
    with zipfile.ZipFile(archive) as target:
        assert target.testzip() is None
        assert f"{ROOT.name}/external/mendeley_index.sql" in target.namelist()
        assert f"{ROOT.name}/artifacts/v2/reference.joblib" in target.namelist()
        assert f"{ROOT.name}/submissions/submission_reference.csv" in target.namelist()
        assert not any(name.endswith("mendeley_bundle.js") for name in target.namelist())
    archive.with_suffix(".zip.sha256").write_text(f"{digest(archive)}  {archive.name}\n",encoding="utf-8")
    print(json.dumps({"archive":str(archive),"bytes":archive.stat().st_size,"files":len(files),"crc_ok":True}))


if __name__=="__main__":
    main()
