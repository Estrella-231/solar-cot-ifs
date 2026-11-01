"""Select eight clean train-only 8-to-16 windows with exact CPP target files."""
import argparse
import csv
from datetime import datetime
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cpp-inventory", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    available = set()
    with a.cpp_inventory.open(newline="", encoding="utf-8") as f:
        available = {(r["split"], r["utc"]) for r in csv.DictReader(f) if r["cpp_present"] == "1"}
    candidates = []
    with a.manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != "train":
                continue
            paths = row["data_relpaths"].split("|")
            bjt0 = datetime.strptime(row["BJT_start"], "%Y-%m-%d %H:%M:%S")
            # Keep all target hours in the local day, avoiding twilight-heavy smoke windows.
            if bjt0.strftime("%Y-%m") not in {"2025-06", "2025-07", "2025-08", "2025-09"}:
                continue
            if not (6 <= bjt0.hour <= 8):
                continue
            future = [Path(x).stem for x in paths[8:24]]
            if all(("train", t) in available for t in future):
                candidates.append({"seq_id": row["seq_id"], "bjt_start": row["BJT_start"],
                                   "month": bjt0.strftime("%Y-%m"),
                                   "history_utc": "|".join(Path(x).stem for x in paths[:8]),
                                   "future_utc": "|".join(future)})
    selected = []
    for month in ("2025-06", "2025-07", "2025-08", "2025-09"):
        month_rows = [r for r in candidates if r["month"] == month]
        # Deterministic date-stratified picks, separated by at least two days.
        picked = []
        for row in month_rows:
            date = row["bjt_start"][:10]
            if all(abs((datetime.strptime(date, "%Y-%m-%d") - datetime.strptime(x["bjt_start"][:10], "%Y-%m-%d")).days) >= 2 for x in picked):
                picked.append(row)
            if len(picked) == 2:
                break
        if len(picked) < 2:
            raise RuntimeError(f"fewer than two separated full-support smoke windows in {month}")
        selected.extend(picked)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / "m1_smoke_train_windows.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(selected[0]))
        w.writeheader()
        w.writerows(selected)
    print(f"selected={len(selected)}; targets_each=16; all_exact_CPP_supported=true")
    for x in selected:
        print(x["seq_id"], x["bjt_start"], x["month"])
    print(out)


if __name__ == "__main__":
    main()
