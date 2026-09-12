"""Bounded frozen-R numerical backend witness; never edits models or caches.

Fixed before GPU execution: every numerical comparison uses max absolute
error <= 0.001. A numerical failure is reported, not raised or relaxed; all
six previously selected sequences are completed. Provenance/shape failures
stop execution. R stays FP32, eval, no_grad and requires_grad=False.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from train_cot_repaired import COTUNet


ABS_TOL = 0.001
QUANTILES = (0.0, 0.5, 0.9, 0.99, 0.999, 1.0)
FIXED_SAMPLES = (
    ("train", "shard_00000.npz", 3),
    ("train", "shard_00577.npz", 13848),
    ("train", "shard_01152.npz", 27648),
    ("val", "shard_00000.npz", 0),
    ("val", "shard_00088.npz", 2112),
    ("val", "shard_00177.npz", 4248),
)
C_FEATURES = ("mean_log1p_cot", "std_log1p_cot", "p90_log1p_cot", "center_log1p_cot")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def stats(left, right):
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    require(a.shape == b.shape, f"comparison shape mismatch: {a.shape}/{b.shape}")
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        return dict(status="FAIL_NONFINITE", shape=list(a.shape), n_points=int(a.size),
                    threshold_absolute=ABS_TOL, nonfinite_left=int((~np.isfinite(a)).sum()),
                    nonfinite_right=int((~np.isfinite(b)).sum()))
    error = a - b
    absolute = np.abs(error)
    maximum = float(absolute.max())
    return dict(status="PASS" if maximum <= ABS_TOL else "FAIL_ABSOLUTE_TOLERANCE",
                shape=list(a.shape), n_points=int(a.size), threshold_absolute=ABS_TOL,
                max_absolute_error=maximum, rmse=float(np.sqrt(np.mean(error ** 2))),
                mean_signed_error=float(error.mean()), mean_absolute_error=float(absolute.mean()),
                exact_equal=bool(np.array_equal(a, b)), quantile_levels=list(QUANTILES),
                absolute_error_quantiles=np.quantile(absolute, QUANTILES).tolist())


def backend_state():
    return dict(cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_enabled=torch.backends.cudnn.enabled,
                mkldnn_enabled=torch.backends.mkldnn.enabled,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                default_dtype=str(torch.get_default_dtype()))


def configure_gpu(allow_tf32):
    # These two configurations change both autotuning and cuDNN TF32 policy.
    # Their difference alone must not be attributed solely to TF32.
    torch.backends.cudnn.benchmark = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False


def explicit_torch(cot):
    flat = cot.flatten(-2)
    return torch.stack((flat.mean(-1), flat.std(-1, unbiased=False),
                        torch.quantile(flat, 0.9, dim=-1), cot[..., 8, 8]), -1)


def explicit_numpy(cot):
    flat = cot.reshape(*cot.shape[:-2], -1)
    return np.stack((flat.mean(-1), flat.std(-1),
                     np.quantile(flat, 0.9, axis=-1), cot[..., 8, 8]), -1)


def load_contracts(project):
    cache = project / "data/frozen_forecast_trainval_20260907_v1"
    diagnostic_path = project / "audits/cot_feature_diagnostics_20260908.json"
    diagnostic = read_json(diagnostic_path)
    samples = diagnostic["cached_shard_samples"]
    require(tuple((s["split"], s["shard"], s["sequence"]) for s in samples) == FIXED_SAMPLES,
            "diagnostic six-sequence selection changed")
    require(diagnostic["test_used"] is False, "diagnostic used test")
    contract = read_json(cache / "contract.json")
    require(sha(cache / "contract.json") == diagnostic["cache_contract_sha256"], "cache contract hash mismatch")
    require(sha(project / "scripts/cache_frozen_forecast.py") == contract["script_sha256"],
            "cache inference source changed")
    require(sha(inspect.getfile(COTUNet)) == contract["R_source_sha256"], "R model source changed")
    rc = read_json(project / "configs/r_frozen_repaired_seed42.json")
    require(rc == contract["R"], "frozen R config differs from cache contract")
    require(sha(rc["checkpoint"]) == rc["checkpoint_sha256"] == diagnostic["R_checkpoint_sha256"],
            "R checkpoint hash mismatch")
    checkpoint = torch.load(rc["checkpoint"], map_location="cpu")
    metadata = checkpoint["metadata"]
    require(metadata["source_code_sha256"] == contract["R_source_sha256"], "R checkpoint/source mismatch")
    require(metadata["test_used"] is False, "R checkpoint test contract")
    norm_path = Path(metadata["pack"]) / "norm.json"
    rnorm = metadata["norm"]
    require(sha(norm_path) == metadata["pack_sha256"]["norm.json"], "R norm hash mismatch")
    require(read_json(norm_path) == rnorm == diagnostic["r_norm"], "R norm content mismatch")
    channels = [f"C{i:02d}" for i in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15)]
    require(rnorm["feature_names"] == channels + ["cosSOZ", "cosRAA", "day_mask"], "R channel order")
    require(rnorm["test_used"] is False, "R normalization scope")
    require(np.isfinite(rnorm["mean"]).all() and np.isfinite(rnorm["std"]).all()
            and (np.asarray(rnorm["std"]) > 0).all(), "nonfinite R normalization")
    for sample in samples:
        complete = read_json(cache / sample["split"] / "complete.json")
        require(complete["state"] == "COMPLETE" and complete["test_used"] is False, "cache split state")
        matches = [item for item in complete["shards"] if item["name"] == sample["shard"]]
        require(len(matches) == 1 and matches[0]["sha256"] == sample["shard_sha256"], "shard receipt binding")
        path = cache / sample["split"] / sample["shard"]
        require(sha(path) == sample["shard_sha256"], "source shard hash mismatch")
        receipt = read_json(path.with_suffix(".json"))
        require(receipt == matches[0], "sidecar differs from complete manifest")
        with np.load(path, allow_pickle=False) as shard:
            indices = shard["indices"]
            require(len(indices) == receipt["samples"] and (indices == sample["sequence"]).sum() == 1,
                    "fixed sequence not unique in original batch")
        print("SHARD_METADATA_VERIFIED", sample["split"], sample["shard"], flush=True)
    pack = project / "data/head_pack_trainval_20260908_v1"
    head_audit = read_json(pack / "audit.json")
    require(sha(pack / "audit.json") == diagnostic["pack_audit_sha256"], "head pack audit mismatch")
    head_norm = read_json(pack / "norm.json")
    require(sha(pack / "norm.json") == head_audit["norm_sha256"] == diagnostic["norm_sha256"],
            "head C statistics hash mismatch")
    require(head_norm["c_features"] == list(C_FEATURES) and head_norm["fit_split"] == "train", "C definition")
    # Only feature values and index keys are read; no GHI, kt, loss or test payload.
    keys = ("c", "split", "sequence", "station", "lead")
    head_arrays = {}
    for key in keys:
        path = pack / f"{key}.npy"
        require(sha(path) == head_audit["arrays"][key]["sha256"] == diagnostic["array_sha256"][key],
                f"head feature/index hash mismatch: {key}")
        head_arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
    require(set(np.unique(head_arrays["split"])) == {0, 1}, "unexpected head split")
    provenance = dict(diagnostic_sha256=sha(diagnostic_path), cache_contract_sha256=sha(cache / "contract.json"),
                      R_checkpoint_sha256=rc["checkpoint_sha256"], R_source_sha256=contract["R_source_sha256"],
                      R_norm_file_sha256=sha(norm_path), R_norm=rnorm,
                      cache_script_sha256=contract["script_sha256"],
                      head_norm_sha256=sha(pack / "norm.json"), head_C_stats=head_norm["c"],
                      head_pack_audit_sha256=sha(pack / "audit.json"),
                      exact_six_samples=[dict(split=s["split"], shard=s["shard"], sequence=s["sequence"],
                                              sha256=s["shard_sha256"]) for s in samples])
    return cache, samples, checkpoint, rnorm, head_norm, head_arrays, provenance


@torch.no_grad()
def witness(project, report):
    loaded = load_contracts(project)
    cache, samples, checkpoint, rnorm, head_norm, head_arrays, provenance = loaded
    report["provenance"] = provenance
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "requires exactly one allocated visible GPU")
    require(torch.get_default_dtype() == torch.float32, "default FP32 contract changed")
    report["initial_backend_defaults"] = backend_state()
    report["runtime"] = dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                            cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
                            pbs_job_id=os.environ.get("PBS_JOBID"),
                            CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"))
    cpu_model = COTUNet().float().eval().requires_grad_(False)
    cpu_model.load_state_dict(checkpoint["model"], strict=True)
    gpu_model = COTUNet().float().eval().requires_grad_(False).cuda()
    gpu_model.load_state_dict(checkpoint["model"], strict=True)
    require(all(not p.requires_grad for model in (cpu_model, gpu_model) for p in model.parameters()), "R not frozen")
    rm_cpu = torch.tensor(rnorm["mean"], dtype=torch.float32)[None, :, None, None]
    rs_cpu = torch.tensor(rnorm["std"], dtype=torch.float32)[None, :, None, None]
    rm_gpu, rs_gpu = rm_cpu.cuda(), rs_cpu.cuda()
    cm = np.asarray(head_norm["c"]["mean"], np.float32)
    cs = np.asarray(head_norm["c"]["std"], np.float32)
    report["samples"] = []
    for sample in samples:
        started = time.monotonic()
        with np.load(cache / sample["split"] / sample["shard"], allow_pickle=False) as shard:
            indices = shard["indices"]
            agri = shard["predicted_agri_physical"]
            geom = shard["geometry"]
            saved = shard["cot_log1p"]
            saved_explicit = shard["cot_features"]
        batch = len(indices)
        require(agri.shape == (batch, 2, 16, 13, 16, 16) and geom.shape == (batch, 2, 16, 3, 16, 16), "input shape")
        require(saved.shape == (batch, 2, 16, 16, 16) and saved_explicit.shape == (batch, 2, 16, 4), "output shape")
        require(all(a.dtype == np.float32 and np.isfinite(a).all()
                    for a in (agri, geom, saved, saved_explicit)), "cached FP32 finite contract")
        local = int(np.flatnonzero(indices == sample["sequence"])[0])
        raw_cpu = torch.cat((torch.from_numpy(agri), torch.from_numpy(geom)), 3).flatten(0, 2)
        require(raw_cpu.shape == (batch * 32, 16, 16, 16), "flatten order changed")
        numpy_norm = (raw_cpu.numpy() - rm_cpu.numpy()) / rs_cpu.numpy()
        cpu_norm = (raw_cpu - rm_cpu) / rs_cpu
        raw_gpu = raw_cpu.cuda()
        gpu_norm = (raw_gpu - rm_gpu) / rs_gpu
        gpu_norm_cpu = gpu_norm.cpu()
        slot = slice(local * 32, (local + 1) * 32)
        comparisons = {
            "normalization_torch_cpu_vs_numpy_fp32": stats(cpu_norm.numpy(), numpy_norm),
            "normalization_torch_gpu_vs_torch_cpu_fp32": stats(gpu_norm_cpu.numpy(), cpu_norm.numpy()),
            "normalization_torch_gpu_vs_numpy_fp32": stats(gpu_norm_cpu.numpy(), numpy_norm),
        }
        cpu_from_torch = cpu_model(cpu_norm[slot]).numpy().reshape(2, 16, 16, 16)
        cpu_from_numpy = cpu_model(torch.from_numpy(numpy_norm[slot])).numpy().reshape(2, 16, 16, 16)
        cpu_from_gpu_norm = cpu_model(gpu_norm_cpu[slot]).numpy().reshape(2, 16, 16, 16)
        modes = {}
        mode_settings = {}
        for allow, name in ((True, "benchmark_true_tf32_true"), (False, "benchmark_false_tf32_false")):
            configure_gpu(allow)
            mode_settings[name] = backend_state()
            # Same full original shard batch and FP32 input; no autocast context.
            features = gpu_model.forward_features(gpu_norm)
            cot = torch.nn.functional.softplus(gpu_model.head(features)).reshape(batch, 2, 16, 16, 16)
            require(cot.dtype == torch.float32, "R unexpectedly left FP32")
            modes[name] = cot.cpu().numpy()
            comparisons[name + "_vs_saved_full_shard"] = stats(modes[name], saved)
            comparisons[name + "_explicit_vs_saved_full_shard"] = stats(explicit_torch(cot).cpu().numpy(), saved_explicit)
            del features, cot
        on, off = modes["benchmark_true_tf32_true"], modes["benchmark_false_tf32_false"]
        comparisons.update({
            "gpu_on_vs_gpu_off_full_shard": stats(on, off),
            "cpu_numpy_norm_vs_saved_fixed_sequence": stats(cpu_from_numpy, saved[local]),
            "cpu_torch_norm_vs_saved_fixed_sequence": stats(cpu_from_torch, saved[local]),
            "gpu_on_vs_cpu_torch_norm_fixed_sequence": stats(on[local], cpu_from_torch),
            "gpu_off_vs_cpu_torch_norm_fixed_sequence": stats(off[local], cpu_from_torch),
            "gpu_off_vs_cpu_same_gpu_normalized_input_fixed_sequence": stats(off[local], cpu_from_gpu_norm),
            "cpu_torch_norm_vs_cpu_numpy_norm_fixed_sequence": stats(cpu_from_torch, cpu_from_numpy),
            "cpu_gpu_norm_vs_cpu_cpu_norm_fixed_sequence": stats(cpu_from_gpu_norm, cpu_from_torch),
            "saved_log1p_to_explicit_numpy_full_shard": stats(explicit_numpy(saved), saved_explicit),
            "saved_log1p_to_explicit_torch_cpu_full_shard": stats(explicit_torch(torch.from_numpy(saved)).numpy(), saved_explicit),
            "saved_log1p_to_explicit_torch_gpu_full_shard": stats(explicit_torch(torch.from_numpy(saved).cuda()).cpu().numpy(), saved_explicit),
        })
        sid = 0 if sample["split"] == "train" else 1
        rows = np.flatnonzero((head_arrays["split"] == sid) & (head_arrays["sequence"] == sample["sequence"]))
        require(len(rows) == sample["labeled_windows"], "diagnostic labeled feature keys changed")
        stations, leads = head_arrays["station"][rows], head_arrays["lead"][rows]
        feature_raw = saved_explicit[local, stations, leads]
        comparisons["saved_explicit_to_pack_C_float32_formula"] = stats((feature_raw - cm) / cs, head_arrays["c"][rows])
        comparisons["pack_C_inverse_to_saved_explicit"] = stats(
            head_arrays["c"][rows].astype(np.float64) * cs.astype(np.float64) + cm.astype(np.float64), feature_raw)
        result = dict(split=sample["split"], shard=sample["shard"], shard_sha256=sample["shard_sha256"],
                      sequence=sample["sequence"], sequence_local_index=local,
                      original_shard_sequences=batch, original_R_batch=batch * 32,
                      cpu_fixed_sequence_patches=32, cpu_fixed_sequence_pixels=8192,
                      head_feature_rows=rows.tolist(), gpu_mode_settings=mode_settings,
                      comparisons=comparisons, seconds=time.monotonic() - started)
        report["samples"].append(result)
        print("NUMERIC_WITNESS", sample["split"], sample["sequence"],
              json.dumps({key: value["status"] for key, value in comparisons.items()}), flush=True)
        del agri, geom, saved, saved_explicit, raw_cpu, raw_gpu, numpy_norm, cpu_norm, gpu_norm, gpu_norm_cpu, modes, on, off
    names = report["samples"][0]["comparisons"]
    report["comparison_summary"] = {}
    for name in names:
        items = [sample["comparisons"][name] for sample in report["samples"]]
        finite = all("max_absolute_error" in item for item in items)
        report["comparison_summary"][name] = dict(
            status="PASS" if all(item["status"] == "PASS" for item in items) else "FAIL",
            sample_count=len(items), point_count=sum(item["n_points"] for item in items),
            max_absolute_error=max(item["max_absolute_error"] for item in items) if finite else None,
            pooled_rmse=float(np.sqrt(sum(item["rmse"] ** 2 * item["n_points"] for item in items) /
                                      sum(item["n_points"] for item in items))) if finite else None)
    report["state"] = ("COMPLETE_ALL_COMPARISONS_PASS" if
        all(item["status"] == "PASS" for item in report["comparison_summary"].values())
        else "COMPLETE_WITH_NUMERICAL_DIFFERENCES")


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if args.metadata_only:
        *_, provenance = load_contracts(args.project)
        print("PASS_CPU_METADATA_ONLY_NO_FORWARD", json.dumps(provenance), flush=True)
        return
    require(args.output is not None, "output is required for a numerical witness")
    require(not args.output.exists(), "refuse to overwrite a previous witness report")
    report = dict(state="IN_PROGRESS", started_utc=datetime.now(timezone.utc).isoformat(),
                  script_sha256=sha(__file__), test_used=False, GHI_or_kt_read=False,
                  models_or_cache_modified=False, R_frozen=True, R_forward_precision="FP32 without autocast",
                  threshold_absolute=ABS_TOL, threshold_timing="fixed before this GPU experiment; never relaxed",
                  numerical_failures="record every comparison and finish all six samples",
                  C_features=list(C_FEATURES), C_feature_domain="statistics of log1p(COT), center [8,8]",
                  backend_attribution="on/off changes both cuDNN benchmark and TF32; do not infer TF32 alone",
                  comparison_scope="GPU uses whole original shard batch; CPU uses all 32 patches of each fixed sequence")
    try:
        witness(args.project, report)
    except Exception as error:
        report.update(state="ERROR_INCOMPLETE_WITNESS", error_type=type(error).__name__, error=str(error),
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_report(args.output, report)
        raise
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_report(args.output, report)
    print(report["state"], flush=True)


if __name__ == "__main__":
    main()
