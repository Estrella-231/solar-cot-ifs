#!/usr/bin/env python3
"""Prove train GHI rows use their held-out fold SimVP->R COT bank."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


BASE = Path("/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021")
ROLES = [f"fold_{i}" for i in range(5)]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def bjt_sequence(row: dict[str, str]) -> tuple[str, list[str]]:
    data = row["data_paths"].split("|")
    aux = row["aux_paths"].split("|")
    if len(data) != 24 or len(aux) != 24:
        raise ValueError("Each SimVP sequence must contain 8 history + 16 future frames")
    stamps = []
    for d, a in zip(data, aux):
        utc = datetime.strptime(Path(d).stem[:12], "%Y%m%d%H%M")
        bjt = datetime.strptime(Path(a).stem[:12], "%Y%m%d%H%M")
        if bjt - utc != timedelta(hours=8):
            raise ValueError("SimVP source UTC and aux BJT keys disagree")
        stamps.append(bjt.strftime("%Y%m%d%H%M"))
    if any(datetime.strptime(stamps[i + 1], "%Y%m%d%H%M") - datetime.strptime(stamps[i], "%Y%m%d%H%M") != timedelta(minutes=15) for i in range(23)):
        raise ValueError("SimVP sequence is not consecutive at 15-minute cadence")
    return stamps[7], stamps[8:24]


def audit(base: Path, old_pack: Path, output: Path) -> dict[str, object]:
    # The date assignment is an input preparation artifact, kept outside the
    # packed arrays. Bind to that authoritative source and record its path.
    fold_path = base / "preparation_attempt1/metadata/train_date_folds.csv"
    day_rows = rows(fold_path)
    heldout = {r["date_BJT"].replace("-", ""): int(r["heldout_fold"]) for r in day_rows}
    if not heldout:
        raise ValueError("Missing frozen train-date fold assignment")
    banks: dict[str, list[dict[str, str]]] = {}
    index_meta = {}
    for role in ROLES + ["validation"]:
        if role == "validation":
            index_path = base / "prepared_attempt12/indexes/full.csv"
            index_rows = [r for r in rows(index_path) if r["split"] == "val"]
            bank_path = base / "S_banks_attempt12/full/val/keys.csv"
        else:
            index_path = base / f"prepared_attempt12/indexes/{role}_oof.csv"
            index_rows = rows(index_path)
            if any(r["split"] != "val" for r in index_rows):
                raise ValueError(f"OOF forecast index {role} contains non-held-out rows")
            bank_path = base / f"S_banks_attempt12/{role}/val/keys.csv"
        bank_rows = rows(bank_path)
        if len(index_rows) != len(bank_rows):
            raise ValueError(f"{role}: index/bank row count mismatch {len(index_rows)} != {len(bank_rows)}")
        for i, (src, bank) in enumerate(zip(index_rows, bank_rows)):
            init, targets = bjt_sequence(src)
            if bank["split"] != "val" or bank["init_time_bjt"] != init or bank["target_times_bjt"].split("|") != targets:
                raise ValueError(f"{role} bank key does not match source index at sequence {i}")
            if role != "validation":
                fold = heldout.get(init[:8])
                if fold != int(role[-1]):
                    raise ValueError(f"OOF fold leakage/misrouting: {role} sequence {init} belongs to heldout fold {fold}")
        banks[role] = bank_rows
        index_meta[role] = {"index_path": str(index_path), "index_sha256": sha(index_path),
                            "key_path": str(bank_path), "key_sha256": sha(bank_path),
                            "sequences": len(index_rows)}

    old_rows = rows(old_pack / "rows.csv")
    role_counts = Counter((r["split"], r["source_role"]) for r in old_rows)
    checked_rows = 0
    unique_seq = set()
    for r in old_rows:
        split, role = r["split"], r["source_role"]
        if split == "train" and role not in ROLES:
            raise ValueError(f"Train row uses a non-OOF forecast source: {role}")
        if split == "val" and role != "validation":
            raise ValueError(f"Validation row does not use the full-train S->R source: {role}")
        if split not in ("train", "val"):
            raise ValueError(f"Unexpected downstream split: {split}")
        seq_i = int(r["source_sequence_index"])
        if not 0 <= seq_i < len(banks[role]):
            raise ValueError(f"Out-of-range sequence index {seq_i} for {role}")
        k = banks[role][seq_i]
        lead = int(r["lead_minutes"])
        if lead < 15 or lead > 240 or lead % 15:
            raise ValueError(f"Invalid canonical lead {lead}")
        expected_target = k["target_times_bjt"].split("|")[lead // 15 - 1]
        if k["init_time_bjt"] != r["init_time_bjt"] or expected_target != r["target_time_bjt"]:
            raise ValueError(f"GHI pack row is not bound to its selected forecast bank: row={r}")
        unique_seq.add((role, seq_i))
        checked_rows += 1
    train_rows = sum(n for (split, _), n in role_counts.items() if split == "train")
    val_rows = sum(n for (split, _), n in role_counts.items() if split == "val")
    fold_total = sum(index_meta[r]["sequences"] for r in ROLES)
    if fold_total != 4086 or index_meta["validation"]["sequences"] != 407 or train_rows != 153515 or val_rows != 15275:
        raise ValueError(f"Unexpected frozen coverage: oof_seq={fold_total}, val_seq={index_meta['validation']['sequences']}, train_rows={train_rows}, val_rows={val_rows}")
    report = {"state": "PASS_OOF_TRAIN_BANK_LINEAGE_AND_GHI_ROW_BINDING",
              "scope": "training rows use date-held-out fold SimVP->R banks; validation rows use full-train SimVP->R bank; reused validation, not independent test",
              "train_rows_checked": train_rows, "validation_rows_checked": val_rows,
              "unique_source_sequences_used": len(unique_seq), "all_pack_rows_time_bound": checked_rows,
              "source_role_counts": {f"{s}|{r}": n for (s, r), n in sorted(role_counts.items())},
              "fold_sequence_total": fold_total, "validation_sequence_count": index_meta["validation"]["sequences"],
              "date_fold_assignment_path": str(fold_path),
              "date_fold_assignment_sha256": sha(fold_path),
              "source_indices_and_keys": index_meta, "test_payloads_read": 0, "buffer_payloads_read": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--old-pack", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(audit(a.base, a.old_pack, a.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
