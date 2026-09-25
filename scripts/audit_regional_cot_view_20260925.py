"""Measure upstream source support for candidate Hunan station view sizes.

This is a geometry audit of fixed historical-C13 motion records, not a cloud
forecast or a claim that the estimated motion is correct. No test data are read.
"""
import argparse
import hashlib
import json
from pathlib import Path


STATIONS = {"sili": (155, 171, 140, 156), "zhujia": (65, 81, 142, 158)}
GRID_SIZE = 256
SIZES = (16, 32, 64, 128, 256)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounds(station, size):
    r0, r1, c0, c1 = STATIONS[station]
    if size == GRID_SIZE:
        return (0, GRID_SIZE, 0, GRID_SIZE)
    cy, cx = r0 + 8, c0 + 8
    top, left = cy - size // 2, cx - size // 2
    if not (0 <= top and top + size <= GRID_SIZE and 0 <= left and left + size <= GRID_SIZE):
        raise ValueError(f"{station} {size} crop crosses Hunan grid")
    return top, top + size, left, left + size


def fraction_in_view(box, source_ys, source_xs):
    top, bottom, left, right = box
    count = sum(top <= y < bottom and left <= x < right for y in source_ys for x in source_xs)
    return count / (len(source_ys) * len(source_xs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-pilot", type=Path, required=True)
    p.add_argument("--val-pilot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise RuntimeError("refuse to overwrite regional-view audit")
    report = {"state": "REGIONAL_VIEW_GEOMETRY_AUDIT", "test_used": False,
              "source": "historical C13 phase-correlation pilot; small fixed train/val samples",
              "crop_definition": "station pixel centered; 256 uses full Hunan grid",
              "source_convention": "backward sample = target pixel minus round(motion per 15 min * lead)",
              "limits": "source coverage is geometric only; velocity and COT retrieval accuracy remain unverified",
              "views": {}}
    for split, path in (("train", a.train_pilot), ("val", a.val_pilot)):
        pilot = json.loads(path.read_text(encoding="utf-8"))
        records = pilot["records"]
        if (pilot.get("state") != "COMPLETE_P16_SUPPORT_PILOT" or
                pilot.get("test_used") is not False or
                {r["station"] for r in records} != set(STATIONS) or
                len(records) != 2 * pilot["sequences"]):
            raise RuntimeError(f"invalid {split} pilot contract")
        result = {"pilot_sha256": digest(path), "sequences": pilot["sequences"],
                  "station_records": len(records), "per_view": {}}
        for size in SIZES:
            lead_rows = []
            for lead in range(1, 17):
                whole, center = [], []
                for rec in records:
                    r0, r1, c0, c1 = STATIONS[rec["station"]]
                    sy = round(rec["dy_pixels_per_15min"] * lead)
                    sx = round(rec["dx_pixels_per_15min"] * lead)
                    box = bounds(rec["station"], size)
                    whole.append(fraction_in_view(box, range(r0-sy, r1-sy), range(c0-sx, c1-sx)))
                    center.append(fraction_in_view(box, range(r0+6-sy, r0+11-sy),
                                                   range(c0+6-sx, c0+11-sx)))
                lead_rows.append({"lead_minutes": lead*15,
                                  "source_support_full16": sum(whole)/len(whole),
                                  "source_support_station5": sum(center)/len(center),
                                  "fraction_records_full_station5_support":
                                      sum(v == 1 for v in center)/len(center)})
            result["per_view"][str(size)] = lead_rows
        report["views"][split] = result
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({s: {n: report["views"][s]["per_view"][n][-1]["source_support_station5"]
                          for n in map(str, SIZES)} for s in ("train", "val")}))


if __name__ == "__main__":
    main()
