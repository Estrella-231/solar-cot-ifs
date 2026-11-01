import csv
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from build_future_ghi_oof_manifests import build


def rows_for(start_utc: datetime, split: str):
    paths = [(start_utc + timedelta(minutes=15 * i)).strftime("%Y/%Y%m/%Y%m%d/%Y%m%d%H%M.npy") for i in range(24)]
    bjt = start_utc + timedelta(hours=8)
    return {"split": split, "seq_id": "hunan_" + start_utc.strftime("%Y%m%d%H%M"),
            "BJT_start": bjt.strftime("%Y-%m-%d %H:%M:%S"), "data_relpaths": "|".join(paths)}


def write(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def main():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = datetime(2026, 4, 1, 0, 0)
        official_rows = [rows_for(base + timedelta(days=i), "train") for i in range(8)]
        official_rows += [rows_for(base + timedelta(days=9), "val")]
        bank_rows = [{"bank_index": str(i), "split": "train", "sequence_index_within_split": str(i),
                      "seq_id": r["seq_id"]} for i, r in enumerate(official_rows[:6])]
        official, bank = root / "official.csv", root / "bank.csv"
        write(official, official_rows, ["split", "seq_id", "BJT_start", "data_relpaths"])
        write(bank, bank_rows, ["bank_index", "split", "sequence_index_within_split", "seq_id"])
        out = root / "out"
        result = build(official, bank, out, k=3, salt="witness")
        assert result["official_train_sequences"] == 8
        assert result["bank_train_sequences"] == result["oof_bank_coverage"] == 6
        assert sum(f["train_sequences"] + f["heldout_sequences"] for f in result["folds"]) == 24
        assert all(f["train_heldout_frame_intersection"] == 0 for f in result["folds"])
        assert all(f["train_sequences"] > f["heldout_sequences"] for f in result["folds"])
        assert all((out / f"fold_{i}" / "manifest_audit.json").exists() for i in range(3))
    print("OOF_MANIFEST_BUILDER_WITNESS_PASS")


if __name__ == "__main__":
    main()
