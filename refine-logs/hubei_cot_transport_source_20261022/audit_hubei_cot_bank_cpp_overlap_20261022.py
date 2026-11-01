#!/usr/bin/env python3
"""Check whether the frozen Hubei R->COT bank can support CPP-scored forecasts.

Reads train/validation timestamp manifests only. It does not open COT/CPP/AGRI
arrays, labels, test payloads, or buffer payloads and does not authorize training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

CADENCE = timedelta(minutes=15)
HISTORY = 8
LEADS = range(15, 241, 15)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path: Path, time_col: str = "time_bjt") -> dict[datetime, str]:
    result: dict[datetime, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if not {time_col, "split"}.issubset(r.fieldnames or []):
            raise ValueError(f"{path}: expected {time_col},split; got {r.fieldnames}")
        for line, row in enumerate(r, 2):
            t = datetime.strptime(row[time_col].strip(), "%Y%m%d%H%M")
            split = row["split"].strip().lower()
            if split not in {"train", "val"}:
                raise ValueError(f"{path}:{line}: unexpected split={split!r}")
            if t in result:
                raise ValueError(f"{path}:{line}: duplicate timestamp {t}")
            result[t] = split
    return result


def read_cpp(path: Path) -> tuple[dict[datetime, str], dict[str, int]]:
    valid: dict[datetime, str] = {}
    counts = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        required = {"time_bjt", "time_cpp_utc", "split", "valid_cot_pixels", "valid_clp_pixels"}
        if not required.issubset(r.fieldnames or []):
            raise ValueError(f"{path}: missing {required - set(r.fieldnames or [])}")
        for line, row in enumerate(r, 2):
            t = datetime.strptime(row["time_bjt"].strip(), "%Y%m%d%H%M")
            split = row["split"].strip().lower()
            utc = datetime.strptime(row["time_cpp_utc"].strip(), "%Y%m%d%H%M%S")
            if utc != t - timedelta(hours=8):
                raise ValueError(f"{path}:{line}: BJT/UTC mismatch")
            cot_n, clp_n = int(row["valid_cot_pixels"]), int(row["valid_clp_pixels"])
            if cot_n != clp_n or cot_n not in {0, 256}:
                raise ValueError(f"{path}:{line}: unsupported COT/CLP mask {cot_n}/{clp_n}")
            if t in valid:
                raise ValueError(f"{path}:{line}: duplicate CPP timestamp {t}")
            counts[f"{split}_rows"] += 1
            if cot_n == 256:
                valid[t] = split
                counts[f"{split}_full_support"] += 1
            else:
                counts[f"{split}_zero_support"] += 1
    return valid, dict(counts)


def run(bank_csv: Path, agri_csv: Path, cpp_csv: Path) -> dict:
    bank = read_manifest(bank_csv)
    agri = read_manifest(agri_csv)
    cpp, cpp_counts = read_cpp(cpp_csv)
    differences = [
        {"time_bjt": t.strftime("%Y%m%d%H%M"), "bank_split": bank.get(t), "agri_split": agri.get(t)}
        for t in sorted(set(bank) | set(agri))
        if bank.get(t) != agri.get(t)
    ]
    if differences:
        raise ValueError(f"R-bank/AGRI manifest key or split mismatch, examples={differences[:5]}")
    cpp_keys_missing_from_bank = sorted(set(cpp) - set(bank))
    cpp_split_mismatch = sorted(t for t, split in cpp.items() if bank.get(t) != split)
    if cpp_keys_missing_from_bank or cpp_split_mismatch:
        raise ValueError(
            "CPP support is not a subset of same-split R-bank keys: "
            f"missing={len(cpp_keys_missing_from_bank)}, split_mismatch={len(cpp_split_mismatch)}"
        )

    split_results = {}
    for split in ("train", "val"):
        inputs = {t for t, s in bank.items() if s == split}
        targets = {t for t, s in cpp.items() if s == split and s == bank.get(t)}
        init_times = []
        for t in sorted(inputs):
            if all(t - CADENCE * k in inputs for k in range(HISTORY)):
                init_times.append(t)
        by_lead = {}
        all_pairs = 0
        init_dates = set()
        target_dates = set()
        for lead in LEADS:
            pairs = [(init, init + timedelta(minutes=lead)) for init in init_times if init + timedelta(minutes=lead) in targets]
            all_pairs += len(pairs)
            init_dates.update(x.date().isoformat() for x, _ in pairs)
            target_dates.update(y.date().isoformat() for _, y in pairs)
            by_lead[str(lead)] = {
                "init_target_pairs": len(pairs),
                "unique_init_dates": len({x.date().isoformat() for x, _ in pairs}),
                "unique_target_dates": len({y.date().isoformat() for _, y in pairs}),
            }
        split_results[split] = {
            "cot_bank_input_frames": len(inputs),
            "cpp_full_support_targets": len(targets),
            "initializations_with_8_contiguous_history_frames": len(init_times),
            "scorable_init_lead_pairs": all_pairs,
            "unique_init_dates_with_scored_pairs": len(init_dates),
            "unique_target_dates_with_scored_pairs": len(target_dates),
            "pairs_by_lead_min": by_lead,
        }

    return {
        "state": "PASS_MANIFEST_ALIGNMENT_TEST_BUFFER_CLOSED",
        "bank_manifest": str(bank_csv),
        "bank_manifest_sha256": digest(bank_csv),
        "agri_manifest": str(agri_csv),
        "agri_manifest_sha256": digest(agri_csv),
        "cpp_manifest": str(cpp_csv),
        "cpp_manifest_sha256": digest(cpp_csv),
        "bank_rows": len(bank),
        "agri_rows": len(agri),
        "bank_agri_key_split_mismatch_count": len(differences),
        "cpp_timestamp_keys_missing_from_bank": len(cpp_keys_missing_from_bank),
        "cpp_bank_split_mismatch_count": len(cpp_split_mismatch),
        "cpp_counts": cpp_counts,
        "split_overlap": len(set(t for t, s in bank.items() if s == "train") & set(t for t, s in bank.items() if s == "val")),
        "by_split": split_results,
        "scope": "Exact timestamp/split support only; CPP is the existing station-local 16x16 retrieval reference. The R bank is frozen-model output on actual AGRI. Train R outputs are in-domain, not OOF. No payload arrays, labels, test, or buffer were read.",
        "training_decision": "NOT_AUTHORIZED_BY_THIS_AUDIT; validation support and scientific gate still require review.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bank-csv", type=Path, required=True)
    p.add_argument("--agri-csv", type=Path, required=True)
    p.add_argument("--cpp-csv", type=Path, required=True)
    p.add_argument("--json-output", type=Path, required=True)
    p.add_argument("--markdown-output", type=Path, required=True)
    a = p.parse_args()
    result = run(a.bank_csv, a.agri_csv, a.cpp_csv)
    a.json_output.parent.mkdir(parents=True, exist_ok=True)
    a.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lead_rows = []
    for split in ("train", "val"):
        for lead, values in result["by_split"][split]["pairs_by_lead_min"].items():
            lead_rows.append(f"| {split} | +{int(lead)} | {values['init_target_pairs']} | {values['unique_init_dates']} | {values['unique_target_dates']} |")
    text = [
        "# Hubei R-bank to CPP forecast-key overlap audit",
        "",
        f"State: `{result['state']}`",
        "",
        "This is a metadata-only join of the frozen R-derived 16x16 COT bank, the frozen train/validation AGRI index, and the existing corrected-UTC CPP reference manifest. No arrays, labels, test, or buffer were opened.",
        "",
        f"- R-bank rows: {result['bank_rows']}; AGRI rows: {result['agri_rows']}; exact key/split mismatches: {result['bank_agri_key_split_mismatch_count']}",
        f"- CPP manifest: {result['cpp_counts']}",
        "- Train-history COT maps are frozen-R predictions on its own training AGRI frames (not OOF); validation maps are held-out R predictions.",
        "- CPP targets are the existing station-local 16x16 retrieval reference, not independent cloud truth.",
        "",
        "| Split | Lead | Scorable init/target pairs | Init dates | Target dates |",
        "|---|---:|---:|---:|---:|",
        *lead_rows,
        "",
        "These counts establish only metadata coverage. They do not prove CPP file/QC support beyond the existing sidecar, forecast skill, or adequacy of date-independent validation. This audit does not authorize model training.",
        "",
    ]
    a.markdown_output.write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({"state": result["state"], "bank_agri_mismatches": result["bank_agri_key_split_mismatch_count"], "by_split": result["by_split"]}, indent=2))


if __name__ == "__main__":
    main()
