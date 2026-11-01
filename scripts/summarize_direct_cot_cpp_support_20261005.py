"""Summarize exact CPP temporal support by split, lead, month and season."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--occurrences", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    present = {}
    with a.inventory.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            present[(row["split"], row["utc"])] = int(row["cpp_present"])
    groups = defaultdict(lambda: [0, 0, set(), set()])
    strata = defaultdict(lambda: [0, 0, set(), set()])
    with a.occurrences.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            ok = present[(row["split"], row["utc"])]
            day = row["target_bjt"][:10]
            for key in ((row["split"], row["lead_min"], row["season_bjt"]),
                        (row["split"], row["lead_min"], row["month_bjt"])):
                g = groups[key]
                g[0] += 1
                g[1] += ok
                if ok:
                    g[2].add(day)
                    g[3].add(row["utc"])
            for stratum in (row["season_bjt"], row["month_bjt"]):
                g = strata[(row["split"], stratum)]
                g[0] += 1
                g[1] += ok
                if ok:
                    g[2].add(day)
                    g[3].add(row["utc"])
    a.out_dir.mkdir(parents=True, exist_ok=True)
    path = a.out_dir / "cpp_temporal_support_by_lead_season_month.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        w = csv.writer(stream)
        w.writerow(["split", "lead_min", "stratum", "expected_occurrences", "supported_occurrences",
                    "coverage", "supported_init_days", "supported_unique_target_times"])
        for key in sorted(groups, key=lambda k: (k[0], int(k[1]), k[2])):
            n, yes, days, times = groups[key]
            w.writerow([*key, n, yes, f"{yes/n:.6f}" if n else "", len(days), len(times)])
    totals = defaultdict(lambda: [0, 0])
    with a.occurrences.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = row["split"]
            totals[key][0] += 1
            totals[key][1] += present[(key, row["utc"])]
    print(f"summary={path}")
    for split, (n, yes) in sorted(totals.items()):
        print(f"{split}: sequence-target support {yes}/{n}={yes/n:.4%}")
    for lead in (15, 60, 120, 180, 240):
        for split in ("train", "val"):
            matches = [(key, v) for key, v in groups.items() if key[0] == split and key[1] == str(lead)]
            if matches:
                n = sum(v[0] for _, v in matches)
                yes = sum(v[1] for _, v in matches)
                print(f"{split}/lead{lead}: {yes}/{n}={yes/n:.4%}")
    strata_path = a.out_dir / "cpp_temporal_support_by_split_season_month.csv"
    with strata_path.open("w", newline="", encoding="utf-8") as stream:
        w = csv.writer(stream)
        w.writerow(["split", "stratum", "expected_occurrences", "supported_occurrences",
                    "coverage", "supported_BJT_days", "supported_unique_target_times"])
        for key in sorted(strata):
            n, yes, days, times = strata[key]
            w.writerow([*key, n, yes, f"{yes/n:.6f}" if n else "", len(days), len(times)])
    print(f"strata_summary={strata_path}")
    for key, (n, yes, days, times) in sorted(strata.items()):
        print(f"{key[0]}/{key[1]}: {yes}/{n}={yes/n:.2%}; days={len(days)} unique_times={len(times)}")


if __name__ == "__main__":
    main()
