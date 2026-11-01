"""Confirm a failed partial fold cache is redundant before local cleanup."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--complete-cache", type=Path, required=True)
    parser.add_argument("--partial-cache", type=Path, required=True)
    parser.add_argument("--train-frames", type=Path, required=True)
    args = parser.parse_args()

    complete = args.complete_cache.resolve(strict=True)
    partial = args.partial_cache.resolve(strict=True)
    assert complete != partial and complete.is_dir() and partial.is_dir()
    assert complete.parent == partial.parent == Path("/tmp")
    expected = {f"{line}.npz" for line in args.train_frames.read_text().splitlines() if line}
    complete_files = {p.relative_to(complete).as_posix(): p for p in complete.rglob("*.npz")}
    partial_files = {p.relative_to(partial).as_posix(): p for p in partial.rglob("*.npz")}
    assert set(complete_files) == expected, (
        f"complete cache mismatch: expected={len(expected)} found={len(complete_files)}"
    )
    assert partial_files and set(partial_files) <= expected, "partial cache contains unknown frame path"
    missing_from_complete = set(partial_files) - set(complete_files)
    size_mismatch = [
        rel for rel, part in partial_files.items()
        if rel in complete_files and part.stat().st_size != complete_files[rel].stat().st_size
    ]
    assert not missing_from_complete and not size_mismatch, (
        f"partial cache not covered by complete cache: missing={len(missing_from_complete)} "
        f"size_mismatch={len(size_mismatch)}"
    )
    assert not list(partial.rglob("*.tmp")), "unfinished partial-cache write present"
    partial_bytes = sum(p.stat().st_size for p in partial_files.values())
    print(
        "REDUNDANT_FOLD0_PARTIAL_CACHE_VERIFIED",
        f"partial_frames={len(partial_files)}",
        f"complete_frames={len(complete_files)}",
        f"partial_bytes={partial_bytes}",
    )


if __name__ == "__main__":
    main()
