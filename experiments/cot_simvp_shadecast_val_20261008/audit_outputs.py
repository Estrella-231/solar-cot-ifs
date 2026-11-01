"""Independent CPU checks for saved monthly-val per-sample metric totals."""
import hashlib
import json
from pathlib import Path

import numpy as np


METHODS = ("simvp", "shadecast_nowcaster", "shadecast_diffusion_mean_agri")
GROUPS = {"short_15_60": (0, 4), "long_120_240": (7, 16)}
NBOOT = 10000
SEED = 20261008


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_close(actual, expected, label):
    if not np.allclose(actual, expected, rtol=1e-11, atol=1e-12):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def bootstrap(blocks, method, kind, lo, hi):
    codes = np.unique(blocks["month_block"])
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(codes), size=(NBOOT, len(codes)))
    sums = []
    for code in codes:
        mask = blocks["month_block"] == code
        if kind in ("agri_all", "agri_safe"):
            prefix = "agri_all" if kind == "agri_all" else "agri_safe"
            abs_by_method = blocks[prefix+"_abs_z"][mask, :, :, lo:hi].sum((0, 2, 3))
            sq_by_method = blocks[prefix+"_sq_z"][mask, :, :, lo:hi].sum((0, 2, 3))
            counts = blocks[prefix+"_count"][mask]
            count = counts[:, :, lo:hi].sum() if kind == "agri_all" else counts[:, 0, :, lo:hi].sum()
            sums.append(np.stack((abs_by_method, sq_by_method,
                                  np.full(len(METHODS), count, dtype=np.float64)), axis=-1))
        elif kind == "cot_log":
            absolute = blocks["cot_log_abs"][mask, :, lo:hi].sum((0, 2))
            squared = blocks["cot_log_sq"][mask, :, lo:hi].sum((0, 2))
            count = blocks["cot_count"][mask, :, lo:hi].sum((0, 2))
            sums.append(np.stack((absolute, squared, count), axis=-1))
        elif kind == "thick_miss":
            miss = blocks["thick_miss"][mask, :, lo:hi].sum((0, 2))
            ref = blocks["thick_ref"][mask, :, lo:hi].sum((0, 2))
            sums.append(np.stack((miss, ref), axis=-1))
        else:
            raise ValueError(kind)
    sums = np.stack(sums).astype(np.float64)
    boot = sums[picks].sum(1)
    if kind == "thick_miss":
        score = boot[:, :, 0] / np.maximum(boot[:, :, 1], 1)
        point = sums[:, :, 0].sum(0) / np.maximum(sums[:, :, 1].sum(0), 1)
    else:
        score = np.sqrt(boot[:, :, 1] / np.maximum(boot[:, :, 2], 1))
        point = np.sqrt(sums[:, :, 1].sum(0) / np.maximum(sums[:, :, 2].sum(0), 1))
    return float(point[method] - point[0]), np.percentile(score[:, method]-score[:, 0], [2.5, 97.5])


def main(folder):
    folder = Path(folder)
    summary_path = folder / "summary.json"
    complete_path = folder / "complete.json"
    totals_path = folder / "per_sample_totals.npz"
    summary = json.loads(summary_path.read_text())
    complete = json.loads(complete_path.read_text())
    assert summary["state"] == complete["state"] == "COMPLETE_EXPLORATORY_R_PROXY_VALIDATION_COMPARISON"
    assert complete["summary_sha256"] == sha(summary_path)
    assert complete["per_sample_totals_sha256"] == sha(totals_path)
    assert summary["validation_sequences"] == complete["validation_sequences"] == 3140
    assert summary["monthly_validation_blocks"] == 26
    assert summary["cpp_used"] is False and summary["test_payloads_read"] == 0
    assert summary["r_provenance"]["eligible_new_val_unique_target_frames"] == 3403
    assert summary["r_provenance"]["excluded_new_val_unique_target_frames_seen_in_r_train"] == 431
    assert summary["r_provenance"]["excluded_new_val_unique_target_frames_seen_in_r_checkpoint_validation"] == 191

    with np.load(totals_path, allow_pickle=False) as z:
        data = {key: z[key].copy() for key in z.files}
    ids = data["index"]
    assert ids.shape == (3140,) and np.array_equal(np.sort(ids), np.arange(3140))
    assert len(np.unique(ids)) == 3140
    assert len(np.unique(data["month_block"])) == 26
    for name in ("agri_all_abs_z", "agri_all_sq_z", "agri_all_abs_phys", "agri_all_sq_phys",
                 "agri_safe_abs_z", "agri_safe_sq_z", "agri_safe_abs_phys", "agri_safe_sq_phys"):
        assert data[name].shape == (3140, 3, 13, 16), (name, data[name].shape)
        assert np.isfinite(data[name]).all(), name
    assert data["agri_all_count"].shape == (3140, 13, 16)
    assert data["agri_safe_count"].shape == (3140, 3, 13, 16)
    assert np.array_equal(data["agri_safe_count"], np.broadcast_to(data["agri_safe_count"][:, :1], data["agri_safe_count"].shape))
    for name in ("agri_all_count", "agri_safe_count", "cot_count", "thick_ref", "thick_nonref"):
        assert np.isfinite(data[name]).all() and (data[name] >= 0).all(), name
    for name in ("cot_log_abs", "cot_log_sq", "cot_phys_abs", "cot_phys_sq", "cot_count",
                 "thick_miss", "thick_ref", "thick_false_alarm", "thick_nonref"):
        expected_shape = (3140, 3, 16)
        assert data[name].shape == expected_shape, (name, data[name].shape)
        assert np.isfinite(data[name]).all(), name
    assert np.array_equal(data["cot_count"], np.broadcast_to(data["cot_count"][:, :1], data["cot_count"].shape))
    assert np.array_equal(data["thick_ref"], np.broadcast_to(data["thick_ref"][:, :1], data["thick_ref"].shape))
    assert np.array_equal(data["thick_nonref"], np.broadcast_to(data["thick_nonref"][:, :1], data["thick_nonref"].shape))

    # Recompute point estimates directly from the saved per-sample sufficient
    # statistics, independently of the summary-generation routine.
    for group, (lo, hi) in GROUPS.items():
        for method, name in enumerate(METHODS):
            entry = summary["metrics"][name]
            for support, prefix in (("agri_full_spatial_nonleak_validation", "agri_all"),
                                    ("agri_r_safe_daylight", "agri_safe")):
                counts = data[prefix+"_count"]
                if prefix == "agri_safe":
                    counts = counts[:, method]
                denom = counts[:, :, lo:hi].sum()
                abs_z = data[prefix+"_abs_z"][:, method, :, lo:hi]
                sq_z = data[prefix+"_sq_z"][:, method, :, lo:hi]
                abs_phys = data[prefix+"_abs_phys"][:, method, :, lo:hi]
                sq_phys = data[prefix+"_sq_phys"][:, method, :, lo:hi]
                target = entry[support][group]
                check_close(abs_z.sum()/denom, target["mae_z"], f"{name}/{support}/{group}/mae_z")
                check_close(np.sqrt(sq_z.sum()/denom), target["rmse_z"], f"{name}/{support}/{group}/rmse_z")
                per_channel_count = counts[:, :, lo:hi].sum((0, 2))
                check_close(abs_phys.sum((0, 2))/per_channel_count,
                            target["mae_physical_by_channel"], f"{name}/{support}/{group}/mae_phys")
                check_close(np.sqrt(sq_phys.sum((0, 2))/per_channel_count),
                            target["rmse_physical_by_channel"], f"{name}/{support}/{group}/rmse_phys")
                check_close(denom, target["pixel_channel_count"], f"{name}/{support}/{group}/count")

            cot = entry["cot_proxy_vs_R_real_agri"][group]
            cot_count = data["cot_count"][:, method, lo:hi].sum()
            check_close(data["cot_log_abs"][:, method, lo:hi].sum()/cot_count,
                        cot["log1p_cot_mae"], f"{name}/{group}/cot_log_mae")
            check_close(np.sqrt(data["cot_log_sq"][:, method, lo:hi].sum()/cot_count),
                        cot["log1p_cot_rmse"], f"{name}/{group}/cot_log_rmse")
            check_close(data["cot_phys_abs"][:, method, lo:hi].sum()/cot_count,
                        cot["physical_cot_mae"], f"{name}/{group}/cot_phys_mae")
            check_close(np.sqrt(data["cot_phys_sq"][:, method, lo:hi].sum()/cot_count),
                        cot["physical_cot_rmse"], f"{name}/{group}/cot_phys_rmse")
            check_close(cot_count, cot["common_pixels"], f"{name}/{group}/cot_count")

            thick = entry["retrieved_thick_proxy"][group]
            ref = data["thick_ref"][:, method, lo:hi].sum()
            nonref = data["thick_nonref"][:, method, lo:hi].sum()
            check_close(data["thick_miss"][:, method, lo:hi].sum()/ref,
                        thick["miss_rate"], f"{name}/{group}/thick_miss")
            check_close(data["thick_false_alarm"][:, method, lo:hi].sum()/nonref,
                        thick["false_alarm_rate"], f"{name}/{group}/thick_false_alarm")
            check_close(ref, thick["thick_reference_pixels"], f"{name}/{group}/thick_ref")
            check_close(nonref, thick["non_thick_reference_pixels"], f"{name}/{group}/thick_nonref")

    for group, (lo, hi) in GROUPS.items():
        for method in (1, 2):
            name = METHODS[method]
            stored = summary["monthly_block_bootstrap"][group][name]
            for field, kind in (("agri_full_rmse_z", "agri_all"),
                                ("agri_r_safe_rmse_z", "agri_safe"),
                                ("cot_proxy_log1p_rmse", "cot_log"),
                                ("retrieved_thick_proxy_miss_rate", "thick_miss")):
                delta, ci = bootstrap(data, method, kind, lo, hi)
                check_close(delta, stored[field]["candidate_minus_simvp"], f"{group}/{name}/{field}/delta")
                check_close(ci, stored[field]["delta_ci95"], f"{group}/{name}/{field}/ci")
                assert stored[field]["bootstrap_draws"] == NBOOT
                assert stored[field]["clusters"] == 26
    print(json.dumps({"state": "PASS_INDEPENDENT_OUTPUT_AUDIT", "sequences": 3140,
                      "unique_indices": 3140, "monthly_blocks": 26,
                      "cpp_used": False, "test_payloads_read": 0,
                      "summary_sha256": sha(summary_path),
                      "per_sample_totals_sha256": sha(totals_path)}, indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    main(p.parse_args().folder)
