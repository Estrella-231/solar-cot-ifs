"""Count causal train-only forecast AGRI / CPP-reference COT pairs by lead."""
import csv
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path("/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907")
STATIONS = ("sili", "zhujia")


def main():
    with (ROOT / "data/cot_repaired_pack_20260907_v1/rows.csv").open(newline="") as f:
        source = list(csv.DictReader(f))
    train_targets = {(r["station_id"].lower(), datetime.fromisoformat(r["timestamp_utc"].replace("Z", "+00:00")))
                     for r in source if r["split"] == "train"}
    if len(train_targets) != 11978:
        raise RuntimeError("unexpected unique COT train target count")
    index_path = ROOT / "data/frozen_forecast_trainval_20260907_v1/train/index.csv"
    counts = Counter()
    unique = set()
    with index_path.open(newline="") as f:
        index = list(csv.DictReader(f))
    for row in index:
        if row["split"] != "train":
            raise RuntimeError("forecast cache contains nontrain sequence")
        init = datetime.fromisoformat(row["init_time_utc"].replace("Z", "+00:00"))
        for lead in range(1, 17):
            target = init + timedelta(minutes=lead * 15)
            for station in STATIONS:
                key = (station, target)
                if key in train_targets:
                    counts[lead * 15] += 1
                    unique.add(key)
    result = {"state": "COMPLETE_TRAIN_ONLY_FORECAST_COT_PAIR_INVENTORY",
              "test_used": False, "train_reference_targets": len(train_targets),
              "forecast_train_sequences": len(index), "matched_pairs": sum(counts.values()),
              "unique_matched_reference_targets": len(unique),
              "pairs_by_lead_minutes": {str(lead): counts[lead] for lead in range(15, 241, 15)},
              "note": "Metadata-only count; no future real AGRI/CPP supplied to deployable inference."}
    output = ROOT / "audits/forecast_domain_cot_train_pair_inventory_20260922.json"
    if output.exists():
        raise RuntimeError("refuse to overwrite forecast-domain pair inventory")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
