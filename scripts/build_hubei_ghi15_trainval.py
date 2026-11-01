"""Build provisional Hubei GHI15 labels without reading test radiation values.

The XLSX worksheet XML is streamed twice: pass 1 reads only headers and the
time column; pass 2 captures the radiation cell only for train/validation row
indices identified in pass 1. Test rows' radiation cell text is ignored.
"""
import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import html
import json
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

from hubei_ghi15_contract import aggregate_window

STATIONS = {
    "牛首供电所": "Niushou",
    "王寨供电所": "Wangzhai",
    "米庄供电所": "Mizhuang",
}
NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def col_index(ref):
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n


def worksheet_paths(zf):
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    relmap = {x.attrib["Id"]: x.attrib["Target"] for x in rels}
    result = {}
    for sh in wb.find("m:sheets", NS):
        target = relmap[sh.attrib["{" + NS["r"] + "}id"]]
        path = target.lstrip("/") if target.startswith("/") else "xl/" + target
        if not path.startswith("xl/"):
            path = "xl/" + path
        result[sh.attrib["name"]] = path
    return result


ROW_OPEN = re.compile(rb"<row(?:\s|>)")
ROW_CLOSE = b"</row>"
CELL = re.compile(rb'<c\b(?P<attrs>[^>]*?)(?:/>|>(?P<body>.*?)</c>)', re.S)
ATTR_REF = re.compile(rb'\br="([A-Z]+[0-9]+)"')
ATTR_TYPE = re.compile(rb'\bt="([^"]+)"')
TEXT_NODE = re.compile(rb'<t\b[^>]*>(.*?)</t>', re.S)
VALUE_NODE = re.compile(rb'<v\b[^>]*>(.*?)</v>', re.S)


def iter_xml_rows(zf, sheet_path):
    """Stream row bytes; cell payloads are not decoded until explicitly selected."""
    with zf.open(sheet_path) as stream:
        buf = b""
        in_row = False
        while True:
            chunk = stream.read(1024 * 1024)
            if chunk:
                buf += chunk
            eof = not chunk
            while True:
                if not in_row:
                    match = ROW_OPEN.search(buf)
                    if not match:
                        if eof:
                            return
                        buf = buf[-16:]
                        break
                    buf = buf[match.start():]
                    in_row = True
                end = buf.find(ROW_CLOSE)
                if end < 0:
                    if eof:
                        raise ValueError("Truncated worksheet row XML")
                    break
                stop = end + len(ROW_CLOSE)
                yield buf[:stop]
                buf = buf[stop:]
                in_row = False
            if eof:
                return


def attr_value(attrs, pattern, default=b""):
    match = pattern.search(attrs)
    return match.group(1) if match else default


def decode_cell(cell_type, body):
    if body is None:
        return ""
    if cell_type == "inlineStr":
        pieces = [html.unescape(x.decode("utf-8")) for x in TEXT_NODE.findall(body)]
        return "".join(pieces)
    if cell_type == "s":
        return VALUE_NODE.search(body).group(1).decode("ascii") if VALUE_NODE.search(body) else ""
    match = VALUE_NODE.search(body)
    return html.unescape(match.group(1).decode("utf-8")) if match else ""


def stream_cells(zf, sheet_path, capture_fn):
    """Yield selected cells. Unselected cell payload bytes are never decoded."""
    for row_xml in iter_xml_rows(zf, sheet_path):
        row_match = re.search(rb"<row\b[^>]*\br=\"([0-9]+)\"", row_xml)
        if not row_match:
            continue
        row = int(row_match.group(1))
        for match in CELL.finditer(row_xml):
            attrs = match.group("attrs")
            ref_bytes = attr_value(attrs, ATTR_REF)
            if not ref_bytes:
                continue
            ref = ref_bytes.decode("ascii")
            col = col_index(ref)
            if not capture_fn(row, col, ref):
                continue
            cell_type = attr_value(attrs, ATTR_TYPE, b"n").decode("ascii")
            yield row, ref, cell_type, decode_cell(cell_type, match.group("body"))


def parse_timestamp(text):
    return datetime.fromisoformat(text.strip())


def read_header_and_times(zf, sheet_path):
    headers = {}
    times = {}

    def wanted(row, col, ref):
        return row == 1 or col == 17

    for row, ref, cell_type, value in stream_cells(zf, sheet_path, wanted):
        if cell_type == "s":
            raise ValueError("Shared-string timestamp/header encountered; safe selective decoder required")
        if row == 1:
            headers[value.strip()] = col_index(ref)
        elif col_index(ref) == 17:
            try:
                times[row] = parse_timestamp(value)
            except (ValueError, TypeError):
                times[row] = None
    if "时间" not in headers or "总辐射1瞬时值" not in headers:
        raise ValueError(f"Expected columns missing: {headers}")
    return headers, times


def ceil_target(t):
    # Map a source timestamp to the right edge of its 15-minute bin.
    minute_floor = t.replace(second=0, microsecond=0)
    q = minute_floor.minute % 15
    if q == 0 and t == minute_floor:
        return minute_floor
    add = 15 - q if q else 15
    return minute_floor + timedelta(minutes=add)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx-root", required=True)
    ap.add_argument("--day-assignment", required=True)
    ap.add_argument("--target-support", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    split_by_date = {}
    with open(args.day_assignment, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            split_by_date[r["date_BJT"]] = r["split"]
    targets = set()
    with open(args.target_support, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            targets.add(datetime.fromisoformat(r["target_time_BJT"]))
    target_split = {t: split_by_date.get(t.date().isoformat()) for t in targets}
    allowed_targets = {t for t, split in target_split.items() if split in {"train", "val"}}
    grouped = {station: defaultdict(list) for station in STATIONS.values()}
    row_counts = Counter()
    invalid_times = Counter()
    files = []
    root = Path(args.xlsx_root)
    paths = []
    for p in sorted(root.glob("天气数据_*.xlsx")):
        stem = p.stem[len("天气数据_"):]
        if (len(stem) == 6 and stem.isdigit()) or stem in (
                "2026-06-01-2026-07-06", "2026-07-07-2026-07-14"):
            paths.append(p)
    if len(paths) != 12:
        raise ValueError(f"Expected 12 authoritative workbooks, got {len(paths)}")
    for path in paths:
        print("TRAIN_VAL_ONLY", path.name, flush=True)
        file_record = {"name": path.name, "sha256": sha256(path), "sheets": {}}
        with zipfile.ZipFile(path) as zf:
            sheets = worksheet_paths(zf)
            if not set(STATIONS).issubset(sheets):
                raise ValueError(f"Expected station sheets missing: {path}")
            for sheet_name, station in STATIONS.items():
                sheet_path = sheets[sheet_name]
                headers, times = read_header_and_times(zf, sheet_path)
                time_col, ghi_col = headers["时间"], headers["总辐射1瞬时值"]
                allowed_rows = set()
                row_target = {}
                for row, t in times.items():
                    if t is None:
                        invalid_times[station] += 1
                        continue
                    tgt = ceil_target(t)
                    if tgt in allowed_targets and t > tgt - timedelta(minutes=15) and t <= tgt:
                        allowed_rows.add(row)
                        row_target[row] = tgt
                sheet_rows = Counter()

                def wanted(row, col, ref):
                    return row in allowed_rows and col == ghi_col

                for row, ref, cell_type, raw in stream_cells(zf, sheet_path, wanted):
                    if cell_type == "s":
                        raise ValueError("Shared-string GHI encountered; selective decoder required")
                    tgt = row_target[row]
                    t = times[row]
                    grouped[station][tgt].append((t, raw if raw != "" else None))
                    row_counts[station] += 1
                    sheet_rows["captured_train_val_radiation_cells"] += 1
                file_record["sheets"][station] = {
                    "time_rows": len(times),
                    "train_val_radiation_cells_captured": sheet_rows["captured_train_val_radiation_cells"],
                    "timestamp_column_index": time_col,
                    "radiation_column_index": ghi_col,
                }
        files.append(file_record)
    labels_path = out / "ghi15_train_val.csv"
    reason_counts = Counter()
    valid_counts = Counter()
    zero_counts = Counter()
    duplicate_extra = Counter()
    with labels_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["station", "target_time_BJT", "split", "ghi15_raw", "mask", "reason", "source_records", "unique_timestamps"])
        for station in STATIONS.values():
            for target in sorted(allowed_targets):
                records = grouped[station].get(target, [])
                result = aggregate_window(records, target)
                reason = result["reason"]
                mask = int(result["valid"])
                value = result["ghi"] if mask else ""
                w.writerow([station, target.isoformat(sep=" "), target_split[target], value, mask,
                            reason, len(records), result.get("unique_records", "")])
                reason_counts[(station, reason)] += 1
                if mask:
                    valid_counts[(station, target_split[target])] += 1
                    if result["ghi"] == 0:
                        zero_counts[station] += 1
                    duplicate_extra[station] += len(records) - result["unique_records"]
    report = {
        "status": "TRAIN_VAL_LABELS_BUILT_TEST_VALUES_CLOSED_UNIT_PROVISIONAL",
        "split_id": "hubei_monthly_block_2d_buffer1d_seed42_202508_202607_frozen_v1",
        "window": "(target-15min,target], 15 right-closed minute bins; minute means equally weighted",
        "source_field": "总辐射1瞬时值",
        "unit_contract": "legacy project treats raw field as W/m2; station-specific instrument/unit metadata not independently verified",
        "test_radiation_values_read": 0,
        "test_radiation_cells_decoded": 0,
        "only_train_val_radiation_cells_captured": True,
        "train_val_target_slots_per_station": len(allowed_targets),
        "captured_raw_train_val_rows_by_station": dict(row_counts),
        "valid_labels_by_station_split": {s: {sp: valid_counts[(s, sp)] for sp in ("train", "val")} for s in STATIONS.values()},
        "mask_zero_labels_by_station": dict(zero_counts),
        "window_reason_counts_by_station": {s: {r: n for (ss, r), n in reason_counts.items() if ss == s} for s in STATIONS.values()},
        "identical_duplicate_extra_records_in_valid_windows_by_station": dict(duplicate_extra),
        "invalid_timestamp_rows_by_station": dict(invalid_times),
        "source_files": files,
        "label_csv_sha256": sha256(labels_path),
        "normalization_fit": "not performed",
        "daylight_mask": "not applied",
        "test_GHI_values": "closed",
    }
    (out / "REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "test_radiation_values_read", "valid_labels_by_station_split", "label_csv_sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
