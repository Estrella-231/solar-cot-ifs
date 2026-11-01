"""Install only reviewed entry points; preserve each remote predecessor."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(sys.argv[1]).resolve(strict=True)
    bundle = Path(sys.argv[2])
    archive = root / "docs/execution_simplification_20261030/remote_before"
    rows = []
    with zipfile.ZipFile(bundle) as z:
        manifest = json.loads(z.read("manifest.json"))
        for item in manifest:
            relative = Path(item["path"])
            target = (root / relative).resolve()
            if relative.is_absolute() or root not in target.parents:
                raise ValueError("Target outside project")
            data = z.read("files/" + item["path"])
            if sha(data) != item["after_sha256"]:
                raise ValueError("Bundle content mismatch: " + item["path"])
            old = target.read_bytes() if target.is_file() else None
            current = sha(old) if old is not None else None
            if current != item["after_sha256"] and current != item["expected_remote_sha256"]:
                raise ValueError("Remote source changed; no files written: " + item["path"])
            backup = archive / relative
            if current != item["after_sha256"] and old is not None and backup.exists() and backup.read_bytes() != old:
                raise ValueError("Backup conflict; no files written: " + item["path"])
            rows.append((item, target, data, old, current, backup))
        changes = []
        for item, target, data, old, current, backup in rows:
            if current != item["after_sha256"]:
                if old is not None:
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    if not backup.exists():
                        backup.write_bytes(old)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".execution_fix_20261030.tmp")
                with temporary.open("xb") as f:
                    f.write(data)
                os.replace(temporary, target)
            if sha(target.read_bytes()) != item["after_sha256"]:
                raise ValueError("Installed content mismatch: " + item["path"])
            changes.append({"path": item["path"], "before_sha256": current,
                            "after_sha256": item["after_sha256"]})
    receipt = {"state": "CURRENT_ENTRY_POINTS_SYNCED", "root": str(root), "files": changes,
               "historical_run_scripts_replaced": False, "qsub_submitted": 0,
               "jobs_cancelled": 0, "test_payloads_read": False}
    (archive.parent / "REMOTE_SYNC.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
