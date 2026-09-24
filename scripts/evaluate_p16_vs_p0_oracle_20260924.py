"""Fixed-sample, supported-pixel P16 advection versus persistence diagnostic.

Uses real future AGRI through frozen R only as a reference; neither P16 nor
P0 may use those future images in forecasting. Invalid advection source pixels
are excluded from BOTH arms for a paired comparison, never filled as clear sky.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refuse to overwrite P16/P0 comparison")
    pilot = json.loads((args.pilot / "pilot.json").read_text())
    oracle_contract = json.loads((args.oracle / "complete.json").read_text())
    history_path = args.history / f"{args.split}_history_cot_log1p.npy"
    history_hash = sha(history_path)
    if (pilot["state"] != "COMPLETE_P16_SUPPORT_PILOT" or pilot["test_used"] or
            pilot["history_cot_sha256"] != history_hash):
        raise RuntimeError("invalid advection pilot")
    if (oracle_contract["state"] != "COMPLETE_REAL_FUTURE_R_ORACLE" or oracle_contract["test_used"] or
            oracle_contract["deployable"] or oracle_contract["split"] != args.split):
        raise RuntimeError("invalid real-future oracle reference")
    indices = np.load(args.pilot / "sequence_indices.npy", allow_pickle=False)
    p16 = np.load(args.pilot / "p16_cot_log1p_nan_outside_support.npy", allow_pickle=False)
    support = np.load(args.pilot / "p16_valid_support.npy", allow_pickle=False)
    ref_path = args.oracle / "real_future_cot_log1p.npy"
    if sha(ref_path) != oracle_contract["cot_sha256"]:
        raise RuntimeError("oracle COT hash drift")
    ref = np.load(ref_path, mmap_mode="r", allow_pickle=False)[indices, :, :, 0]
    history = np.load(history_path, mmap_mode="r", allow_pickle=False)
    p0 = np.broadcast_to(history[indices, :, 7, 0, None, :, :], p16.shape)
    if p16.shape != ref.shape or support.shape != ref.shape or p16.shape[1:] != (2, 16, 16, 16):
        raise RuntimeError("sequence/station/lead spatial contract drift")
    if not np.isfinite(ref).all() or not np.isfinite(p0).all() or not np.isfinite(p16[support]).all():
        raise RuntimeError("nonfinite reference or supported prediction")
    if np.isfinite(p16[~support]).any():
        raise RuntimeError("invalid advection region must remain NaN")
    per_lead = []
    for k in range(16):
        valid = support[:, :, k]
        center = np.zeros_like(valid)
        center[..., 6:11, 6:11] = True
        item = {"lead_minutes": 15*(k+1), "source_support_full_fraction": float(valid.mean()),
                "source_support_center5_fraction": float(valid[..., 6:11, 6:11].mean())}
        for region, region_valid in (("full16", valid), ("center5", valid & center)):
            if not region_valid.any():
                item[region] = None
                continue
            d0 = np.abs(p0[:, :, k][region_valid] - ref[:, :, k][region_valid])
            d16 = np.abs(p16[:, :, k][region_valid] - ref[:, :, k][region_valid])
            item[region] = {"paired_supported_pixels": int(region_valid.sum()),
                            "P0_mae_log1p_cot": float(d0.mean()),
                            "P16_mae_log1p_cot": float(d16.mean()),
                            "P16_minus_P0_mae_log1p_cot": float(d16.mean()-d0.mean())}
        per_lead.append(item)
    report = {"state":"COMPLETE_P16_P0_SUPPORTED_PIXEL_DIAGNOSTIC", "test_used":False,
              "split":args.split, "sequences":len(indices), "stations":2,
              "reference":"frozen R(real future AGRI), not observed physical COT or deployed input",
              "comparison":"P16 and persistence on identical valid P16 source pixels at each lead",
              "limitation":"small fixed sequence pilot; long-lead invalid source pixels excluded, not a full-horizon GHI test",
              "pilot_sha256":sha(args.pilot / "pilot.json"),
              "oracle_complete_sha256":sha(args.oracle / "complete.json"),
              "history_sha256":history_hash,
              "per_lead":per_lead}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(report["state"], args.split, len(indices),
          json.dumps({"lead15":per_lead[0]["center5"], "lead240":per_lead[-1]["center5"]}), flush=True)


if __name__ == "__main__":
    main()
