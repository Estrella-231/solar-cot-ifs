"""CPU-only, zero-training C13 motion and 16x16 COT support diagnostic.

P16 is deliberately a bounded diagnostic: it cannot recover clouds entering
from beyond the 16x16 historical COT map.  No future image or GHI is loaded.
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.ndimage import shift


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def phase_shift(first, second, first_valid, second_valid):
    """Return second image's translation (dy, dx) relative to the first."""
    valid = first_valid & second_valid & np.isfinite(first) & np.isfinite(second)
    if valid.mean() < 0.95:
        return 0.0, 0.0, "low_valid"
    a = np.where(valid, first, 0).astype(np.float64)
    b = np.where(valid, second, 0).astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    if min(a.std(), b.std()) < 1e-5:
        return 0.0, 0.0, "low_texture"
    window = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None]
    fa = np.fft.fft2(a * window)
    fb = np.fft.fft2(b * window)
    spectrum = fb * fa.conj()
    magnitude = np.abs(spectrum)
    if magnitude.max() < 1e-10:
        return 0.0, 0.0, "low_texture"
    correlation = np.fft.ifft2(spectrum / np.maximum(magnitude, 1e-10)).real
    y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
    h, w = a.shape
    if y > h // 2:
        y -= h
    if x > w // 2:
        x -= w
    if abs(y) > 8 or abs(x) > 8:
        return 0.0, 0.0, "outlier_flow"
    return float(y), float(x), "ok"


def advect_physical(logcot, dy, dx, lead):
    physical = np.expm1(logcot.astype(np.float64))
    displaced = shift(physical, (dy * lead, dx * lead), order=1,
                      mode="constant", cval=0.0, prefilter=False)
    support = shift(np.ones_like(physical), (dy * lead, dx * lead), order=1,
                    mode="constant", cval=0.0, prefilter=False)
    valid = support >= 1 - 1e-9
    output = np.log1p(np.maximum(displaced, 0.0)).astype(np.float32)
    output[~valid] = np.nan
    return output, valid


def read_c13(data_root, rel, source_root):
    sys.path.insert(0, str(source_root))
    from hunan_data import resolve_aux
    utc = datetime.strptime(Path(rel).stem, "%Y%m%d%H%M")
    bjt = utc + timedelta(hours=8)
    raw = np.load(data_root / "data" / rel, mmap_mode="r", allow_pickle=False)
    if raw.shape != (20, 256, 256):
        raise RuntimeError("AGRI channel layout mismatch")
    c13 = np.asarray(raw[12], dtype=np.float32)
    with np.load(resolve_aux(data_root, bjt), allow_pickle=False) as aux:
        valid = (aux["agri_valid_bits"] & (1 << 12)) != 0
    return c13, valid & np.isfinite(c13)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--sequences", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refuse to overwrite P16 pilot")
    if args.sequences < 1:
        raise ValueError("sequences must be positive")
    config = json.loads((args.project / "configs/s_frozen_hunan_seed42.json").read_text())
    manifest = Path(config["manifest"])
    if hash_file(manifest) != config["manifest_sha256"]:
        raise RuntimeError("manifest drift")
    with manifest.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == args.split]
    history_path = args.project / f"data/history_cot_trajectory_real_agri_rpair_20260923/original_R/{args.split}_history_cot_log1p.npy"
    history = np.load(history_path, mmap_mode="r")
    if history.shape != (len(rows), 2, 8, 1, 16, 16):
        raise RuntimeError("history bank sequence mismatch")
    indices = np.linspace(0, len(rows) - 1, min(args.sequences, len(rows)), dtype=int)
    indices = np.unique(indices)
    maps = np.full((len(indices), 2, 16, 16, 16), np.nan, np.float32)
    masks = np.zeros_like(maps, dtype=bool)
    records = []
    for output_index, sequence_index in enumerate(indices):
        row = rows[int(sequence_index)]
        paths = row["data_relpaths"].split("|")
        for station_index, (station, box) in enumerate(config["station_patches"].items()):
            r0, r1, c0, c1 = box
            center_r, center_c = r0 + 8, c0 + 8
            c13, valid = [], []
            for rel in paths[:8]:
                image, mask = read_c13(args.data_root, rel, args.source_root)
                c13.append(image[center_r - 32:center_r + 32, center_c - 32:center_c + 32])
                valid.append(mask[center_r - 32:center_r + 32, center_c - 32:center_c + 32])
            motions = [phase_shift(c13[t - 1], c13[t], valid[t - 1], valid[t])
                       for t in range(4, 8)]
            accepted = [(dy, dx) for dy, dx, status in motions if status == "ok"]
            dy = float(np.median([p[0] for p in accepted])) if accepted else 0.0
            dx = float(np.median([p[1] for p in accepted])) if accepted else 0.0
            last = np.asarray(history[sequence_index, station_index, 7, 0])
            if not np.isfinite(last).all() or (last < 0).any():
                raise RuntimeError("historical COT invalid")
            coverage = []
            center_coverage = []
            for lead in range(1, 17):
                forecast, support = advect_physical(last, dy, dx, lead)
                maps[output_index, station_index, lead - 1] = forecast
                masks[output_index, station_index, lead - 1] = support
                coverage.append(float(support.mean()))
                center_coverage.append(float(support[6:11, 6:11].mean()))
            records.append({"sequence_index": int(sequence_index), "seq_id": row["seq_id"],
                            "station": station, "dy_pixels_per_15min": dy, "dx_pixels_per_15min": dx,
                            "pair_motion": motions, "valid_full_patch_by_lead": coverage,
                            "valid_center_5x5_by_lead": center_coverage})
    args.output.mkdir(parents=True)
    np.save(args.output / "sequence_indices.npy", indices)
    np.save(args.output / "p16_cot_log1p_nan_outside_support.npy", maps)
    np.save(args.output / "p16_valid_support.npy", masks)
    report = {"state": "COMPLETE_P16_SUPPORT_PILOT", "test_used": False,
              "split": args.split,
              "source": "historical C13 only; fixed FFT phase correlation of last four pairs",
              "history_cot_sha256": hash_file(history_path), "manifest_sha256": config["manifest_sha256"],
              "script_sha256": hash_file(__file__), "sequences": len(indices),
              "coverage_full_by_lead": np.mean([r["valid_full_patch_by_lead"] for r in records], axis=0).tolist(),
              "coverage_center_by_lead": np.mean([r["valid_center_5x5_by_lead"] for r in records], axis=0).tolist(),
              "motion_status_counts": {s: sum(s == pair[2] for r in records for pair in r["pair_motion"])
                                       for s in ("ok", "low_valid", "low_texture", "outlier_flow")},
              "records": records}
    (args.output / "pilot.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("state", "sequences", "coverage_center_by_lead", "motion_status_counts")}), flush=True)


if __name__ == "__main__":
    main()
