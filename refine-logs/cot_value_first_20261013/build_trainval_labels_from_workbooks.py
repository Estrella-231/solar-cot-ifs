#!/usr/bin/env python3
"""Read only requested train/val GHI cells from the existing station workbooks.

The candidate file is the authority for the allowed (station, target-time) keys.
Workbook XML is streamed by row; on each row only DATA_DATE is decoded unless
the date is needed by a train/val candidate, in which case only the three
requested 5-minute cells are decoded.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timedelta, date
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
STATIONS = {"sili": (25.9478611111, 112.347916667),
            "zhujia": (29.566666667, 112.45)}


def col_num(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def excel_date(raw: str) -> date:
    raw = raw.strip()
    try:
        return datetime.fromisoformat(raw[:10]).date()
    except ValueError:
        # Excel 1900 date system, matching pandas/openpyxl conversion.
        return (datetime(1899, 12, 30) + timedelta(days=float(raw))).date()


def cell_payload(cell: ET.Element, shared: list[str]) -> str | None:
    typ = cell.attrib.get("t", "")
    if typ == "inlineStr":
        return "".join(x.text or "" for x in cell.findall(f".//{{{MAIN}}}t"))
    v = cell.find(f"{{{MAIN}}}v")
    if v is None or v.text is None:
        return None
    if typ == "s":
        return shared[int(v.text)]
    return v.text


def solar_cosine_noaa(t: datetime, lat_deg: float, lon_deg: float) -> float:
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


def shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        stream = zf.open("xl/sharedStrings.xml")
    except KeyError:
        return []
    vals: list[str] = []
    with stream:
        for _, elem in ET.iterparse(stream, events=("end",)):
            if elem.tag == f"{{{MAIN}}}si":
                vals.append("".join(x.text or "" for x in elem.findall(f".//{{{MAIN}}}t")))
                elem.clear()
    return vals


def sheet_path(zf: zipfile.ZipFile, sheet_name: str) -> str:
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rid = None
    for sh in wb.findall(f".//{{{MAIN}}}sheet"):
        if sh.attrib.get("name") == sheet_name:
            rid = sh.attrib[f"{{{REL}}}id"]
            break
    if rid is None:
        raise ValueError(f"sheet not found: {sheet_name}")
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    target = next(x.attrib["Target"] for x in rels.findall(f"{{{PKG_REL}}}Relationship")
                  if x.attrib.get("Id") == rid)
    return str(PurePosixPath("xl") / target) if not target.startswith("/") else target.lstrip("/")


def extract_requested(path: Path, needed: dict[date, dict[str, datetime]]) -> tuple[dict[datetime, float | None], dict[str, object]]:
    with zipfile.ZipFile(path) as zf:
        shared = shared_strings(zf)
        xml_path = sheet_path(zf, "结果集")
        values: dict[datetime, float | None] = {}
        date_col: int | None = None
        cols: dict[str, int] = {}
        min_date: date | None = None
        max_date: date | None = None
        rows_seen = 0
        with zf.open(xml_path) as stream:
            for _, elem in ET.iterparse(stream, events=("end",)):
                if elem.tag != f"{{{MAIN}}}row":
                    continue
                rows_seen += 1
                cells = {col_num(c.attrib["r"]): c for c in elem.findall(f"{{{MAIN}}}c") if "r" in c.attrib}
                if date_col is None:
                    headers: dict[str, int] = {}
                    for idx, c in cells.items():
                        val = cell_payload(c, shared)
                        if val is not None:
                            headers[val.strip()] = idx
                    date_col = headers["DATA_DATE"]
                    names = ["VAL2400"] + [f"VAL{m//60:02d}{m%60:02d}" for m in range(5, 1440, 5)]
                    if len(names) != 288 or any(name not in headers for name in names):
                        raise ValueError("expected the 288 five-minute columns VAL0005..VAL2400")
                    cols = {name: headers[name] for name in names}
                    continue
                cdate = cells.get(date_col)
                if cdate is None:
                    elem.clear()
                    continue
                raw_date = cell_payload(cdate, shared)
                if raw_date is None:
                    elem.clear()
                    continue
                day = excel_date(raw_date)
                if min_date is None or day < min_date: min_date = day
                if max_date is None or day > max_date: max_date = day
                requested = needed.get(day)
                if requested:
                    for hhmm, ts in requested.items():
                        c = cells.get(cols[f"VAL{hhmm}"])
                        raw = cell_payload(c, shared) if c is not None else None
                        try:
                            value = float(raw) if raw not in (None, "") else None
                        except ValueError:
                            value = None
                        if value is not None and not math.isfinite(value):
                            value = None
                        values[ts] = value
                elem.clear()
        return values, {"source_file": path.name, "sheet": "结果集", "daily_rows_scanned": rows_seen-1,
                        "source_date_min": min_date.isoformat() if min_date else None,
                        "source_date_max": max_date.isoformat() if max_date else None,
                        "requested_dates": len(needed), "requested_5min_timestamps": sum(map(len, needed.values()))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--sili-workbook", type=Path, required=True)
    p.add_argument("--zhujia-workbook", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    candidates: list[dict[str, str]] = []
    keys = set()
    with a.candidates.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["split"] not in {"train", "val"}:
                raise ValueError("candidate list includes non-train/val row")
            target = datetime.fromisoformat(row["target_time_bjt"])
            key = (row["station"], target)
            if key in keys: raise ValueError(f"duplicate candidate key: {key}")
            keys.add(key)
            row["_target"] = target.isoformat(sep=" ")
            candidates.append(row)

    outputs = {}
    source_meta = {}
    station_files = {"sili": a.sili_workbook, "zhujia": a.zhujia_workbook}
    for station, workbook in station_files.items():
        targets = [datetime.fromisoformat(r["_target"]) for r in candidates if r["station"] == station]
        # A 00:00 record is represented by the previous day's VAL2400 cell.
        needed: dict[date, dict[str, datetime]] = defaultdict(dict)
        for target in targets:
            for ts in (target-timedelta(minutes=10), target-timedelta(minutes=5), target):
                if ts.hour == 0 and ts.minute == 0:
                    day, hhmm = ts.date()-timedelta(days=1), "2400"
                else:
                    day, hhmm = ts.date(), ts.strftime("%H%M")
                needed[day][hhmm] = ts
        outputs[station], source_meta[station] = extract_requested(workbook, needed)

    a.output_dir.mkdir(parents=True)
    fields = ["new_split", "station", "target_time_bjt", "target_time_utc", "agri_frame_relpath",
              "label_record_times_bjt", "raw_ghi_5min_wm2", "observed_ghi_15min_wm2",
              "day_mask_sza_lt_90", "label_status", "label_source"]
    counts = Counter()
    out = a.output_dir / "unique_target_labels_train_val.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in candidates:
            station = row["station"]; target = datetime.fromisoformat(row["_target"])
            times = [target-timedelta(minutes=m) for m in (10, 5, 0)]
            vals = [outputs[station].get(ts) for ts in times]
            complete = all(v is not None and math.isfinite(v) and v >= 0 for v in vals)
            day = solar_cosine_noaa(target, *STATIONS[station]) > 0.0
            state = "complete_ghi" if complete else "incomplete_ghi_source"
            counts[(row["split"], state)] += 1
            if day:
                counts[(row["split"], "station_day_sza_lt90")] += 1
                counts[(row["split"], "day_complete_ghi" if complete else "day_incomplete_ghi")] += 1
            writer.writerow({"new_split": row["split"], "station": station,
                "target_time_bjt": row["target_time_bjt"], "target_time_utc": row["target_time_utc"],
                "agri_frame_relpath": row["agri_frame_relpath"],
                "label_record_times_bjt": "|".join(t.isoformat(sep=" ") for t in times),
                "raw_ghi_5min_wm2": "|".join("" if v is None else f"{v:.12g}" for v in vals),
                "observed_ghi_15min_wm2": f"{sum(vals)/3:.12g}" if complete else "",
                "day_mask_sza_lt_90": str(day),
                "label_status": state, "label_source": station_files[station].name})
    report = {"state": "TRAIN_VAL_LABELS_EXTRACTED_FROM_EXISTING_WORKBOOKS",
        "split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1", "candidate_keys": len(candidates),
        "candidate_manifest_sha256": sha256(a.candidates),
        "builder_sha256": sha256(Path(__file__)),
        "counts": {"/".join(k): v for k,v in sorted(counts.items())},
        "source_workbooks": {s: {**source_meta[s], "path": str(station_files[s]),
                                  "sha256": sha256(station_files[s])} for s in station_files},
        "label_window": "(target-15min,target]; values at target-10/-5/target",
        "only_candidate_train_val_keys_requested": True,
        "noncandidate_label_cells_decoded": False,
        "day_mask": "NOAA solar cosine at target station/time > 0; same formula as M0 production",
        "source_payload_scope": "train/val candidate dates and exact requested five-minute cells only",
        "output": out.name, "output_sha256": sha256(out)}
    (a.output_dir/"workbook_label_extraction_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
