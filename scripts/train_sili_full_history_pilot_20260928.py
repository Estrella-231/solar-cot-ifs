"""Matched short-vs-full COT-history residual pilot; test split is never loaded."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_sili_transport_thickness_pilot_20260927 import (
    ResidualBlock, TransportThicknessNet, prepare, make_input,
)


def full_history_input(history, forecast, adv, motion):
    # All eight past COT maps are separate channels, repeated over target lead.
    b, _, _, h, w = history.shape
    hist = history[:, :, 0].unsqueeze(2).expand(-1, -1, 16, -1, -1)
    fc = forecast.permute(0, 2, 1, 3, 4)
    lead = torch.linspace(1 / 16, 1, 16, device=forecast.device)[None, None, :, None, None]
    lead = lead.expand(b, 1, -1, h, w)
    mot = (motion / 4)[:, :, None, None, None].expand(-1, -1, 16, h, w)
    return torch.cat((adv, fc, hist, fc - adv, lead, mot), dim=1)


class FullHistoryNet(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.inp = nn.Conv3d(14, width, 3, padding=1)
        self.b1 = ResidualBlock(width)
        self.b2 = ResidualBlock(width)
        self.out = nn.Conv3d(width, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, adv):
        delta = self.out(self.b2(self.b1(F.silu(self.inp(x)))))
        return torch.clamp(adv + delta, min=0, max=float(np.log1p(100)))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def matched_full_state(short_state):
    full = FullHistoryNet()
    state = full.state_dict()
    for name in ("b1", "b2", "out"):
        for key, value in short_state.items():
            if key.startswith(name + "."):
                state[key] = value.detach().cpu().clone()
    old = short_state["inp.weight"].detach().cpu()
    weight = torch.zeros_like(state["inp.weight"])
    # Preserve the current representation exactly: channel 2 is last COT;
    # channel 3 is last-minus-previous. Six older maps start with zero weight.
    weight[:, 0] = old[:, 0]      # transported COT
    weight[:, 1] = old[:, 1]      # forecast COT
    weight[:, 9] = old[:, 2] + old[:, 3]  # last COT plus trend contribution
    weight[:, 8] = -old[:, 3]     # previous COT contribution to the trend
    weight[:, 10] = old[:, 4]     # forecast minus transport
    weight[:, 11] = old[:, 5]     # lead
    weight[:, 12:14] = old[:, 6:8]
    state["inp.weight"] = weight
    state["inp.bias"] = short_state["inp.bias"].detach().cpu().clone()
    full.load_state_dict(state)
    return full


def loss_without_clear_penalty(prediction, target, mask):
    pred = prediction[:, 0]
    cot = torch.expm1(target)
    weight = 1 + 2 * (cot >= 30).float() + (cot < 5).float()
    huber = F.smooth_l1_loss(pred, target, reduction="none")
    return (huber * weight * mask).sum() / torch.clamp((weight * mask).sum(), min=1)


def per_sample_metrics(pred, truth, mask):
    dims = tuple(range(1, pred.ndim))
    err = pred - truth
    def total(value):
        return value.sum(dim=dims).detach().cpu().numpy()
    out = {
        "n": total(mask), "sq": total(err.square() * mask),
        "abs": total(err.abs() * mask), "bias": total(err * mask),
    }
    thick = mask & (truth >= 30)
    out.update({"thick_n": total(thick), "thick_sq": total(err.square() * thick),
                "thick_miss": total((thick & (pred < 10)).float())})
    clear = mask & (truth < 5)
    out.update({"clear_n": total(clear), "clear_false": total((clear & (pred >= 10)).float())})
    for label, lo, hi in (("0_5", 0, 5), ("5_10", 5, 10), ("10_30", 10, 30),
                          ("30_60", 30, 60), ("60_100", 60, 100.001)):
        q = mask & (truth >= lo) & (truth < hi)
        out[label] = {"n": total(q), "sq": total(err.square() * q), "bias": total(err * q)}
    return out


def add_sample_stats(bucket, arrays, i):
    for key in ("n", "sq", "abs", "bias", "thick_n", "thick_sq", "thick_miss", "clear_n", "clear_false"):
        bucket[key] += float(arrays[key][i])
    for label in ("0_5", "5_10", "10_30", "30_60", "60_100"):
        sub = bucket.setdefault(label, {"n": 0.0, "sq": 0.0, "bias": 0.0})
        for key in ("n", "sq", "bias"):
            sub[key] += float(arrays[label][key][i])


def blank_bucket():
    return {"n": 0, "sq": 0.0, "abs": 0.0, "bias": 0.0,
            "thick_n": 0, "thick_sq": 0.0, "thick_miss": 0,
            "clear_n": 0, "clear_false": 0}


def finalize_bucket(x):
    return {
        "n": x["n"], "mae": x["abs"] / x["n"] if x["n"] else None,
        "rmse": (x["sq"] / x["n"]) ** .5 if x["n"] else None,
        "bias": x["bias"] / x["n"] if x["n"] else None,
        "thick_n": x["thick_n"],
        "thick_rmse": (x["thick_sq"] / x["thick_n"]) ** .5 if x["thick_n"] else None,
        "thick_miss_rate": x["thick_miss"] / x["thick_n"] if x["thick_n"] else None,
        "clear_n": x["clear_n"],
        "clear_false_rate": x["clear_false"] / x["clear_n"] if x["clear_n"] else None,
        "bins": {k: {"n": v["n"], "rmse": (v["sq"] / v["n"]) ** .5 if v["n"] else None,
                     "bias": v["bias"] / v["n"] if v["n"] else None}
                 for k, v in x.items() if k in {"0_5", "5_10", "10_30", "30_60", "60_100"}},
    }


def evaluate_with_days(model, data, days, arm, update, eval_batch=16):
    model.eval()
    names = ("simvp", "persistence", "transport", "learned")
    totals = {scope: {name: blank_bucket() for name in names} for scope in ("patch", "station")}
    blocks = {}
    with torch.no_grad():
        for start in range(0, len(days), eval_batch):
            ids = np.arange(start, min(start + eval_batch, len(days)))
            h = data["h"][ids].cuda(); f = data["f"][ids].cuda(); y = data["y"][ids].cuda()
            mask = data["mask"][ids].cuda(); motion = data["motion"][ids].cuda()
            adv = data["adv"][ids].cuda(); support = data["support"][ids].cuda()[:, 0]
            x = make_input(h, f, adv, motion) if arm == "short_history" else full_history_input(h, f, adv, motion)
            pred = model(x, adv)[:, 0]
            target = torch.expm1(y)
            fields = {
                "simvp": torch.expm1(f[:, 0]),
                "persistence": torch.expm1(h[:, -1, 0]).unsqueeze(1).expand(-1, 16, -1, -1),
                "transport": torch.expm1(adv[:, 0]),
                "learned": torch.expm1(pred),
            }
            valid = mask & support
            for name, field in fields.items():
                patch_arrays = per_sample_metrics(field, target, valid)
                station_arrays = per_sample_metrics(field[:, :, 32, 32], target[:, :, 32, 32], valid[:, :, 32, 32])
                for j, seq_idx in enumerate(ids):
                    day = str(days[seq_idx])
                    block = blocks.setdefault(day, {scope: {method: blank_bucket() for method in names}
                                                    for scope in ("patch", "station")})
                    add_sample_stats(totals["patch"][name], patch_arrays, j)
                    add_sample_stats(block["patch"][name], patch_arrays, j)
                    add_sample_stats(totals["station"][name], station_arrays, j)
                    add_sample_stats(block["station"][name], station_arrays, j)
    metrics = {scope: {name: finalize_bucket(value) for name, value in methods.items()}
               for scope, methods in totals.items()}
    return {"arm": arm, "update": update, "metrics": metrics, "day_sums": blocks}


def bootstrap_rmse_diff(a, method_a, b, method_b, scope, regime, reps=5000, seed=20260928):
    # Return a paired init-day bootstrap CI for RMSE(method_a)-RMSE(method_b).
    day_ids = sorted(set(a["day_sums"]) & set(b["day_sums"]))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        chosen = rng.choice(day_ids, size=len(day_ids), replace=True)
        sums = [{"sq": 0.0, "n": 0} for _ in range(2)]
        for day in chosen:
            for pos, (snapshot, method) in enumerate(((a, method_a), (b, method_b))):
                v = snapshot["day_sums"][str(day)][scope][method]
                q = v if regime == "all" else v["_by_regime"][regime]
                sums[pos]["sq"] += q["sq"]
                sums[pos]["n"] += q["n"]
        if sums[0]["n"] and sums[1]["n"]:
            out.append((sums[0]["sq"] / sums[0]["n"]) ** .5 - (sums[1]["sq"] / sums[1]["n"]) ** .5)
    return {"days": len(day_ids), "replicates": reps, "seed": seed,
            "ci95": [float(v) for v in np.quantile(out, [.025, .975])] if out else None}


def add_bootstrap_bins(snapshot):
    for block in snapshot["day_sums"].values():
        for method in block.values():
            for scope in method.values():
                by = {}
                for key in ("all", "0_5", "5_10", "10_30", "30_60", "60_100", "thick"):
                    if key == "all":
                        by[key] = {"n": scope["n"], "sq": scope["sq"]}
                    elif key == "thick":
                        by[key] = {"n": scope["thick_n"], "sq": scope["thick_sq"]}
                    else:
                        by[key] = scope.get(key, {"n": 0, "sq": 0.0})
                scope["_by_regime"] = by


def profile_batch_sizes(train, short_state, full_state, max_mem_gib=70.0):
    rows = []
    for batch in (8, 16, 32, 64, 128, 256, 512):
        model = FullHistoryNet().cuda(); model.load_state_dict(full_state)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        ids = np.arange(batch)
        h = train["h"][ids].cuda(); f = train["f"][ids].cuda(); y = train["y"][ids].cuda()
        mask = train["mask"][ids].cuda(); motion = train["motion"][ids].cuda(); adv = train["adv"][ids].cuda()
        pred = loss = None
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        try:
            pred = model(full_history_input(h, f, adv, motion), adv)
            loss = loss_without_clear_penalty(pred, y, mask)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            rows.append({"batch": batch, "peak_allocated_gib": peak,
                         "step_seconds": time.time() - t0, "status": "ok",
                         "eligible": peak <= max_mem_gib})
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch": batch, "status": "oom", "eligible": False})
            torch.cuda.empty_cache()
        del model, opt, h, f, y, mask, motion, adv, pred, loss
        torch.cuda.empty_cache()
        if rows[-1]["status"] == "oom":
            break
    eligible = [r for r in rows if r.get("eligible")]
    if not eligible:
        raise RuntimeError("No profiled batch fits the 70-GiB reserve gate")
    return max(eligible, key=lambda r: r["batch"])["batch"], rows


def train_arm(name, model_state, arm, train, val, days, output, batch, max_updates=512, eval_interval=16):
    seed_all(42)
    model = (TransportThicknessNet() if arm == "short_history" else FullHistoryNet()).cuda()
    model.load_state_dict(model_state)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    arm_dir = output / name
    arm_dir.mkdir(parents=True, exist_ok=False)
    step0 = evaluate_with_days(model, val, days, arm, 0)
    add_bootstrap_bins(step0)
    best_score = step0["metrics"]["patch"]["transport"]["thick_rmse"]
    best = {"update": 0, "candidate": "step0_zero_residual_transport", "snapshot": step0}
    torch.save({"model": model.state_dict(), "update": 0, "candidate": best["candidate"]}, arm_dir / "best.pt")
    (arm_dir / "step0.json").write_text(json.dumps(step0["metrics"], indent=2) + "\n")
    snapshots = {0: step0}
    (arm_dir / "day_metrics.jsonl").write_text(json.dumps(step0, separators=(",", ":")) + "\n")
    update = 0; stale = 0; eval_rows = []; started = time.time(); epoch = 0
    while update < max_updates and stale < 16:
        epoch += 1
        order = np.random.permutation(512)
        losses = []
        for start in range(0, 512, batch):
            if update >= max_updates:
                break
            ids = order[start:start + batch]
            h = train["h"][ids].cuda(); f = train["f"][ids].cuda(); y = train["y"][ids].cuda()
            mask = train["mask"][ids].cuda(); motion = train["motion"][ids].cuda(); adv = train["adv"][ids].cuda()
            x = make_input(h, f, adv, motion) if arm == "short_history" else full_history_input(h, f, adv, motion)
            pred = model(x, adv); loss = loss_without_clear_penalty(pred, y, mask)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach())); update += 1
            if update % eval_interval == 0 or update == max_updates:
                snap = evaluate_with_days(model, val, days, arm, update)
                add_bootstrap_bins(snap); snapshots[update] = snap
                score = snap["metrics"]["patch"]["learned"]["thick_rmse"]
                rec = {"update": update, "epoch": epoch, "mean_train_loss_since_last_eval": float(np.mean(losses)),
                       "metrics": snap["metrics"], "elapsed_seconds": time.time() - started}
                eval_rows.append(rec); losses = []
                with (arm_dir / "epochs.jsonl").open("a", encoding="utf-8") as fobj:
                    fobj.write(json.dumps(rec) + "\n")
                with (arm_dir / "day_metrics.jsonl").open("a", encoding="utf-8") as fobj:
                    fobj.write(json.dumps(snap, separators=(",", ":")) + "\n")
                if score < best_score:
                    best_score = score; stale = 0
                    best = {"update": update, "candidate": f"learned_update_{update}", "snapshot": snap}
                    torch.save({"model": model.state_dict(), "update": update,
                                "candidate": best["candidate"]}, arm_dir / "best.pt")
                else:
                    stale += 1
        # With permitted batch sizes dividing 512, the inner loop ends at an epoch boundary.
    selected = best["snapshot"]
    report = {"state": "COMPLETE_SILI_HISTORY_ARM", "test_used": False, "arm": arm,
              "updates": update, "evals": eval_rows, "selected_update": best["update"],
              "selected_candidate": best["candidate"], "selected_metrics": selected["metrics"],
              "step0_thick_rmse": step0["metrics"]["patch"]["transport"]["thick_rmse"],
              "profiled_batch": batch}
    (arm_dir / "training_complete.json").write_text(json.dumps(report, indent=2) + "\n")
    del model, opt
    torch.cuda.empty_cache()
    return report, best["snapshot"], snapshots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--sequence-days", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-updates", type=int, default=512)
    a = ap.parse_args()
    meta = json.loads((a.bank / "complete.json").read_text())
    if meta.get("state") != "COMPLETE_SILI_COT_DYNAMICS_BANK" or meta.get("test_used") is not False:
        raise RuntimeError("Unexpected or test-contaminated bank")
    day_npz = np.load(a.sequence_days)
    days = day_npz["init_day_BJT"].astype(str)
    ids = day_npz["sequence_index"].astype(np.int64)
    bank_ids = np.load(a.bank / "val" / "sequence_indices.npy").astype(np.int64)
    if len(days) != 512 or not np.array_equal(ids, bank_ids):
        raise RuntimeError("Validation day sidecar does not exactly match the fixed 512 bank rows")
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    print("PREPARE_FIXED_TRAIN", flush=True); train = prepare(a.bank, "train", np.arange(512), "cuda")
    print("PREPARE_FIXED_VALIDATION", flush=True); val = prepare(a.bank, "val", np.arange(512), "cuda")

    seed_all(42)
    short = TransportThicknessNet()
    short_state = {k: v.detach().cpu().clone() for k, v in short.state_dict().items()}
    full = matched_full_state(short_state)
    # Input-function equivalence witness before the older history channels are trained.
    h = val["h"][:2].cpu(); f = val["f"][:2].cpu(); adv = val["adv"][:2].cpu(); motion = val["motion"][:2].cpu()
    short_eval = TransportThicknessNet().eval(); short_eval.load_state_dict(short_state)
    full.eval()
    with torch.no_grad():
        p_short = short_eval(make_input(h, f, adv, motion), adv)
        p_full = full(full_history_input(h, f, adv, motion), adv)
        init_error = float((p_short - p_full).abs().max())
    if init_error > 1e-6:
        raise RuntimeError(f"Matched input-function initialization failed: max diff={init_error}")
    del short, full, short_eval, p_short, p_full
    torch.cuda.empty_cache()

    full_cpu = matched_full_state(short_state).state_dict()
    batch, profiles = profile_batch_sizes(train, short_state, full_cpu)
    (a.output / "batch_profile.json").write_text(json.dumps({"selected_batch": batch, "reserve_gate_gib": 70,
                                                               "candidates": profiles}, indent=2) + "\n")
    print("BATCH_PROFILE", json.dumps({"selected_batch": batch, "candidates": profiles}), flush=True)

    control_full_state = {k: v.detach().cpu().clone() for k, v in short_state.items()}
    full_state = {k: v.detach().cpu().clone() for k, v in full_cpu.items()}
    control, control_best, control_snaps = train_arm("short_history", control_full_state, "short_history",
                                                       train, val, days, a.output, batch, a.max_updates)
    full_report, full_best, full_snaps = train_arm("full_history", full_state, "full_history",
                                                     train, val, days, a.output, batch, a.max_updates)
    # The selected candidates' paired day-block intervals compare full history with
    # step 0 and the matched short-history arm on the same bank and support.
    intervals = {}
    for scope in ("patch", "station"):
        intervals[scope] = {}
        for regime in ("all", "thick"):
            intervals[scope][f"full_minus_transport_{regime}_rmse"] = bootstrap_rmse_diff(
                full_best, "learned", full_best, "transport", scope, regime)
            intervals[scope][f"short_minus_transport_{regime}_rmse"] = bootstrap_rmse_diff(
                control_best, "learned", control_best, "transport", scope, regime)
            intervals[scope][f"full_minus_short_{regime}_rmse"] = bootstrap_rmse_diff(
                full_best, "learned", control_best, "learned", scope, regime)
    # Store per-day sufficient statistics for an independent, explicit post-run paired bootstrap.
    result = {"state": "COMPLETE_SILI_FULL_HISTORY_PILOT", "test_used": False,
              "only_intended_difference": "two-frame last/trend input vs all eight historical COT maps",
              "bank_complete_sha256": hashlib.sha256((a.bank / "complete.json").read_bytes()).hexdigest(),
              "validation_days": int(len(set(days))),
              "input_function_initialization_max_abs_diff": init_error, "selected_batch": batch,
              "batch_profile": profiles, "short_history": control, "full_history": full_report,
              "paired_day_bootstrap_95pct": intervals,
              "decision_gate": "compare selected learned candidates against step0 and paired short-history; CPP retrieval reference; exploratory validation only"}
    (a.output / "training_complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print("SILI_FULL_HISTORY_PILOT_COMPLETE", json.dumps({
        "selected_batch": batch, "short_candidate": control["selected_candidate"],
        "short_thick_rmse": control["selected_metrics"]["patch"]["learned"]["thick_rmse"],
        "full_candidate": full_report["selected_candidate"],
        "full_thick_rmse": full_report["selected_metrics"]["patch"]["learned"]["thick_rmse"],
        "input_init_maxdiff": init_error}), flush=True)


if __name__ == "__main__":
    main()
