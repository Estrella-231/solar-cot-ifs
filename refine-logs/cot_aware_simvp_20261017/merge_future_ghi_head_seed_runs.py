"""Merge independently trained, seed-isolated future-GHI head runs.

Each child run uses the unchanged trainer with one seed and writes to a unique
output directory. This merger preserves each child artifact while rebuilding
the aggregate manifest expected by the existing paired audit.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


VARIABLE_MANIFEST_KEYS = {"state", "seeds", "runs", "source_sha256"}


def read_manifest(root: Path) -> dict:
    path = root / "run_manifest.json"
    if not path.is_file():
        raise ValueError(f"missing run manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True,
                   help="completed seed-42 aggregate run to extend")
    p.add_argument("--child", type=Path, nargs="+", required=True,
                   help="completed isolated seed runs")
    p.add_argument("--complete-state", required=True,
                   help="expected completed state string in all run manifests")
    p.add_argument("--required-arms", required=True,
                   help="comma-separated arms that must exist for every station/seed")
    a = p.parse_args()
    base = a.base.resolve()
    children = [x.resolve() for x in a.child]
    if len(set(children + [base])) != len(children) + 1:
        raise ValueError("base and child output roots must be distinct")
    arms = tuple(x for x in a.required_arms.split(",") if x)
    if not arms:
        raise ValueError("required arms cannot be empty")
    bm = read_manifest(base)
    if bm.get("state") != a.complete_state:
        raise ValueError("base run is not complete")
    if bm.get("seeds") != [42]:
        raise ValueError("base run must contain exactly seed 42")
    stations = tuple(bm.get("stations", []))
    if not stations:
        raise ValueError("base run has no declared stations")
    combined_runs = list(bm.get("runs", []))
    if len(combined_runs) != len(stations) * len(arms):
        raise ValueError("seed-42 base has incomplete station/arm coverage")
    combined_seeds = {42}
    prepared = []
    for child in children:
        cm = read_manifest(child)
        if cm.get("state") != a.complete_state:
            raise ValueError(f"child run is not complete: {child}")
        seeds = cm.get("seeds", [])
        if len(seeds) != 1 or seeds[0] in combined_seeds:
            raise ValueError(f"child must contain one new seed, got {seeds}")
        stable_base = {k: v for k, v in bm.items() if k not in VARIABLE_MANIFEST_KEYS}
        stable_child = {k: v for k, v in cm.items() if k not in VARIABLE_MANIFEST_KEYS}
        if stable_child != stable_base:
            raise ValueError(f"incompatible child manifest contract: {child}")
        if len(cm.get("runs", [])) != len(stations) * len(arms):
            raise ValueError(f"incomplete child station/arm coverage: {child}")
        seed = seeds[0]
        if seed < 0 or seed in combined_seeds:
            raise ValueError(f"unexpected/duplicate child seed {seed}")
        copies = []
        for station in stations:
            for arm in arms:
                source = child / station / f"seed{seed}" / arm
                required = ("best.pt", "metrics.json", "training_curve.json",
                            "validation_predictions.npz")
                if any(not (source / name).is_file() for name in required):
                    raise ValueError(f"incomplete run artifacts: {source}")
                target = base / station / f"seed{seed}" / arm
                if target.exists():
                    raise FileExistsError(target)
                copies.append((source, target))
        prepared.append((seed, cm, copies))

    for seed, cm, copies in prepared:
        for source, target in copies:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
        combined_runs.extend(cm["runs"])
        combined_seeds.add(seed)

    expected_seeds = {42, *(m["seeds"][0] for _, m, _ in prepared)}
    if combined_seeds != expected_seeds or len(expected_seeds) != len(children) + 1:
        raise ValueError(f"child seed coverage mismatch: {sorted(combined_seeds)}")
    for seed in sorted(expected_seeds):
        for station in stations:
            for arm in arms:
                if not (base / station / f"seed{seed}" / arm / "metrics.json").is_file():
                    raise ValueError(f"missing merged output for {station}/seed{seed}/{arm}")
    bm["seeds"] = sorted(expected_seeds)
    bm["runs"] = sorted(combined_runs, key=lambda x: (x["station"], x["seed"], x["arm"]))
    # The source run and the attenuated-COT follow-up have distinct completion
    # states. Preserve the state requested by the caller after validating that
    # every base/child manifest has that same completed state.
    bm["state"] = a.complete_state
    (base / "run_manifest.json").write_text(json.dumps(bm, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": bm["state"], "seeds": bm["seeds"],
                      "runs": len(bm["runs"]), "output": str(base)}, sort_keys=True))


if __name__ == "__main__":
    main()
