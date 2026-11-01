"""Verify a prebuilt fold cache from train-frame names only; never opens arrays."""
from __future__ import annotations

import sys
from pathlib import Path


def verify(cache_dir: Path, train_frames: Path) -> int:
    cache_dir = cache_dir.resolve()
    expected = {f"{line}.npz" for line in train_frames.read_text().splitlines() if line}
    found = {path.relative_to(cache_dir).as_posix() for path in cache_dir.rglob("*.npz")}
    missing, extra = expected - found, found - expected
    assert not missing and not extra, (
        f"cache inventory mismatch: expected={len(expected)} found={len(found)} "
        f"missing={len(missing)} extra={len(extra)}"
    )
    assert all((cache_dir / rel).stat().st_size > 0 for rel in expected), "empty cached frame"
    assert not list(cache_dir.rglob("*.tmp")), "unfinished cache write present"
    return len(expected)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_future_ghi_oof_cache.py CACHE TRAIN_FRAMES")
    count = verify(Path(sys.argv[1]), Path(sys.argv[2]))
    print("OOF_FOLD0_COMPLETE_TRAIN_CACHE_INVENTORY_PASS", count)


if __name__ == "__main__":
    main()
