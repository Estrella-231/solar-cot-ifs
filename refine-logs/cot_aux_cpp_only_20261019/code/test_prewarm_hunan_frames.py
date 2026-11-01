"""Synthetic witness: only unique train/validation frames are cached."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="cot_cache_witness_") as td:
        root = Path(td)
        source = root / "source"
        source.mkdir()
        (source / "hunan_data.py").write_text(
            "import numpy as np\n"
            "def load_frame(root, rel):\n"
            "    return (np.full((2,2),1,dtype=np.float32),\n"
            "            np.ones((2,2),dtype=bool),\n"
            "            np.zeros((3,2,2),dtype=np.float32))\n",
            encoding="utf-8")
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["split", "data_relpaths"])
            w.writeheader()
            w.writerows([
                {"split": "train", "data_relpaths": "a.npz|b.npz"},
                {"split": "val", "data_relpaths": "b.npz|c.npz"},
                {"split": "test", "data_relpaths": "never_read.npz"},
                {"split": "buffer", "data_relpaths": "also_never_read.npz"},
            ])
        cache = root / "cache"
        out = root / "witness.json"
        subprocess.run([
            sys.executable, str(here / "prewarm_hunan_frames.py"),
            "--source", str(source), "--root", str(root),
            "--manifest", str(manifest), "--cache", str(cache),
            "--output", str(out), "--workers", "2",
        ], check=True, capture_output=True, text=True)
        result = json.loads(out.read_text(encoding="utf-8"))
        assert result["status"] == "PASS_TRAINVAL_FRAME_CACHE_WARM"
        assert result["unique_train_val_frames"] == 3
        assert result["created"] == 3 and result["missing_after"] == 0
        assert result["test_payloads_read"] == 0
        assert (cache / "a.npz.npz").is_file()
        assert (cache / "b.npz.npz").is_file()
        assert (cache / "c.npz.npz").is_file()
        assert not (cache / "never_read.npz.npz").exists()
        assert not (cache / "also_never_read.npz.npz").exists()
    print("PASS synthetic train/val-only prewarm witness")


if __name__ == "__main__":
    main()
