"""Attach the project's deterministic NOAA solar-cosine daylight mask."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

STATIONS = {
    "Niushou": (32.0800, 112.0400),
    "Wangzhai": (32.0400, 112.1200),
    "Mizhuang": (32.1200, 112.2000),
}


def solar_cosine_noaa(t, lat_deg, lon_deg):
    """Same NOAA solar-cosine function used by the Hunan M0 label builder."""
    doy = t.timetuple().tm_yday
    hour = t.hour + t.minute / 60.0 + t.second / 3600.0
    gamma = 2.0 * math.pi / 365.0 * (doy - 1 + (hour - 12.0) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))
    true_solar_minutes = (hour * 60.0 + eqtime + 4.0 * lon_deg - 480.0) % 1440.0
    hour_angle = math.radians(true_solar_minutes / 4.0 - 180.0)
    lat = math.radians(lat_deg)
    return max(-1.0, min(1.0, math.sin(lat) * math.sin(decl)
        + math.cos(lat) * math.cos(decl) * math.cos(hour_angle)))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    inp, out = Path(a.labels), Path(a.output)
    counts = Counter()
    with inp.open(encoding="utf-8-sig", newline="") as src, out.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        fields = reader.fieldnames + ["solar_cosine", "day_mask_sza_lt_90", "valid_daylight_label"]
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            station = row["station"]
            if station not in STATIONS:
                raise ValueError(f"Unknown station {station}")
            t = datetime.fromisoformat(row["target_time_BJT"])
            cosine = solar_cosine_noaa(t, *STATIONS[station])
            day = cosine > 0.0
            valid_day = int(day and int(row["mask"]) == 1)
            row["solar_cosine"] = f"{cosine:.12g}"
            row["day_mask_sza_lt_90"] = str(day)
            row["valid_daylight_label"] = valid_day
            writer.writerow(row)
            counts[(station, row["split"], "slots")] += 1
            counts[(station, row["split"], "day")] += int(day)
            counts[(station, row["split"], "valid_label")] += int(row["mask"])
            counts[(station, row["split"], "valid_daylight_label")] += valid_day
    report = {
        "status": "TRAIN_VAL_NOAA_DAYLIGHT_MASK_ATTACHED_CLEAR_SKY_PENDING",
        "source_label_sha256": digest(inp),
        "output_sha256": digest(out),
        "test_values_read": 0,
        "day_mask": "NOAA solar cosine > 0 (SZA < 90 deg), same function as Hunan M0 builder",
        "coordinates_source": "B0_SOURCE_AUDIT_20261014.md; station table coordinates rounded to 4 decimals",
        "coordinates": {k: {"latitude": v[0], "longitude": v[1]} for k, v in STATIONS.items()},
        "counts_by_station_split": {
            s: {sp: {metric: counts[(s, sp, metric)] for metric in ("slots", "day", "valid_label", "valid_daylight_label")}
                for sp in ("train", "val")}
            for s in STATIONS
        },
        "clear_sky_filter": "not applied; Ineichen interval mean remains to be computed",
        "normalization": "not fit",
        "unit": "provisional legacy W/m2 convention; source instrument metadata unresolved",
    }
    Path(a.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts_by_station_split"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
