"""Capacity-matched full-history versus repeated-last COT control.

Both arms use the same 15-channel Conv3d head. One receives all eight historical
COT fields; the control repeats the final field in all eight history slots while
retaining the same last-minus-previous trend channel. The test split is never read.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_sili_transport_thickness_pilot_20260927 import (
    ResidualBlock, TransportThicknessNet, prepare,
)
from train_sili_full_history_pilot_20260928 import (
    add_bootstrap_bins, add_sample_stats, blank_bucket, bootstrap_rmse_diff,
    finalize_bucket, per_sample_metrics,
)


class HistoryControlNet(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.inp = nn.Conv3d(15, width, 3, padding=1)
        self.b1 = ResidualBlock(width)
        self.b2 = ResidualBlock(width)
        self.out = nn.Conv3d(width, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, adv):
        delta = self.out(self.b2(self.b1(F.silu(self.inp(x)))))
        return torch.clamp(adv + delta, min=0, max=float(np.log1p(100)))


def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def history_input(history, forecast, adv, motion, arm):
    b, _, _, h, w = history.shape
    if arm == "full_history":
        hist = history[:, :, 0]
    elif arm in ("repeat_last", "repeat_last_control"):
        hist = history[:, -1:, 0].expand(-1, 8, -1, -1)
    else:
        raise ValueError(arm)
    hist = hist.unsqueeze(2).expand(-1, -1, 16, -1, -1)
    trend = (history[:, -1, 0] - history[:, -2, 0]).unsqueeze(1).unsqueeze(2)
    trend = trend.expand(-1, 1, 16, -1, -1)
    fc = forecast.permute(0, 2, 1, 3, 4)
    lead = torch.linspace(1 / 16, 1, 16, device=forecast.device)[None, None, :, None, None]
    lead = lead.expand(b, 1, -1, h, w)
    mot = (motion / 4)[:, :, None, None, None].expand(-1, -1, 16, h, w)
    return torch.cat((adv, fc, hist, trend, fc - adv, lead, mot), dim=1)


def matched_state(short_state):
    model = HistoryControlNet()
    state = model.state_dict()
    for name in ("b1", "b2", "out"):
        for key, value in short_state.items():
            if key.startswith(name + "."):
                state[key] = value.detach().cpu().clone()
    old = short_state["inp.weight"].detach().cpu()
    weight = torch.zeros_like(state["inp.weight"])
    weight[:, 0] = old[:, 0]       # transported COT
    weight[:, 1] = old[:, 1]       # forecast COT
    weight[:, 9] = old[:, 2]       # latest historical COT slot
    weight[:, 10] = old[:, 3]      # last-minus-previous trend
    weight[:, 11] = old[:, 4]      # forecast minus transported COT
    weight[:, 12] = old[:, 5]      # lead
    weight[:, 13:15] = old[:, 6:8] # local motion
    state["inp.weight"] = weight
    state["inp.bias"] = short_state["inp.bias"].detach().cpu().clone()
    model.load_state_dict(state)
    return model


def loss_fn(prediction, target, mask):
    pred = prediction[:, 0]
    cot = torch.expm1(target)
    weight = 1 + 2 * (cot >= 30).float() + (cot < 5).float()
    huber = F.smooth_l1_loss(pred, target, reduction="none")
    return (huber * weight * mask).sum() / torch.clamp((weight * mask).sum(), min=1)


def aggregate_snapshot(model, data, days, arm, update, eval_batch=16):
    model.eval()
    names = ("simvp", "persistence", "transport", "learned")
    scopes = ("patch", "station")
    total = {scope: {name: blank_bucket() for name in names} for scope in scopes}
    lead_total = {scope: {str(k): {name: blank_bucket() for name in names}
                          for k in range(1, 17)} for scope in scopes}
    blocks = {}
    with torch.no_grad():
        for start in range(0, len(days), eval_batch):
            ids = np.arange(start, min(start + eval_batch, len(days)))
            h = data["h"][ids].cuda(); f = data["f"][ids].cuda(); y = data["y"][ids].cuda()
            mask = data["mask"][ids].cuda(); motion = data["motion"][ids].cuda()
            adv = data["adv"][ids].cuda(); support = data["support"][ids].cuda()[:, 0]
            pred = model(history_input(h, f, adv, motion, arm), adv)[:, 0]
            target = torch.expm1(y)
            fields = {
                "simvp": torch.expm1(f[:, 0]).expand(-1, 16, -1, -1),
                "persistence": torch.expm1(h[:, -1, 0]).unsqueeze(1).expand(-1, 16, -1, -1),
                "transport": torch.expm1(adv[:, 0]).expand(-1, 16, -1, -1),
                "learned": torch.expm1(pred),
            }
            if any(v.shape != target.shape for v in fields.values()):
                raise RuntimeError(f"Forecast/target shape mismatch: target={tuple(target.shape)}, fields={ {k:tuple(v.shape) for k,v in fields.items()} }")
            valid = mask & support
            for name, field in fields.items():
                all_stats = per_sample_metrics(field, target, valid)
                station_stats = per_sample_metrics(field[:, :, 32, 32], target[:, :, 32, 32], valid[:, :, 32, 32])
                for j, seq_idx in enumerate(ids):
                    day = str(days[seq_idx])
                    block = blocks.setdefault(day, {scope: {method: blank_bucket() for method in names} for scope in scopes})
                    add_sample_stats(total["patch"][name], all_stats, j)
                    add_sample_stats(block["patch"][name], all_stats, j)
                    add_sample_stats(total["station"][name], station_stats, j)
                    add_sample_stats(block["station"][name], station_stats, j)
                for lead_idx in range(16):
                    px = per_sample_metrics(field[:, lead_idx], target[:, lead_idx], valid[:, lead_idx])
                    st = per_sample_metrics(field[:, lead_idx, 32, 32][:, None, None], target[:, lead_idx, 32, 32][:, None, None], valid[:, lead_idx, 32, 32][:, None, None])
                    for j in range(len(ids)):
                        add_sample_stats(lead_total["patch"][str(lead_idx + 1)][name], px, j)
                        add_sample_stats(lead_total["station"][str(lead_idx + 1)][name], st, j)
    metrics = {scope: {name: finalize_bucket(x) for name, x in vals.items()} for scope, vals in total.items()}
    lead_metrics = {scope: {k: {name: finalize_bucket(x) for name, x in vals.items()}
                            for k, vals in leads.items()} for scope, leads in lead_total.items()}
    snapshot = {"arm": arm, "update": update, "metrics": metrics, "lead_metrics": lead_metrics, "day_sums": blocks}
    add_bootstrap_bins(snapshot)
    return snapshot


def profile_batch(train, state, limit_gib=70):
    rows = []
    for batch in (8, 16, 32, 64, 128, 256, 512):
        model = HistoryControlNet().cuda(); model.load_state_dict(state)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        ii = np.arange(batch)
        h=train["h"][ii].cuda(); f=train["f"][ii].cuda(); y=train["y"][ii].cuda()
        mask=train["mask"][ii].cuda(); motion=train["motion"][ii].cuda(); adv=train["adv"][ii].cuda()
        p=loss=None
        torch.cuda.reset_peak_memory_stats(); t0=time.time()
        try:
            p=model(history_input(h,f,adv,motion,"full_history"),adv); loss=loss_fn(p,y,mask)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); torch.cuda.synchronize()
            peak=torch.cuda.max_memory_allocated()/1024**3
            rows.append({"batch":batch,"peak_allocated_gib":peak,"step_seconds":time.time()-t0,"status":"ok","eligible":peak<=limit_gib})
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch":batch,"status":"oom","eligible":False}); torch.cuda.empty_cache()
        del model,opt,h,f,y,mask,motion,adv,p,loss; torch.cuda.empty_cache()
        if rows[-1]["status"]=="oom": break
    eligible=[r for r in rows if r.get("eligible")]
    if not eligible: raise RuntimeError("No batch meets the GPU reserve gate")
    return max(eligible,key=lambda r:r["batch"])["batch"],rows


def run_arm(arm, state, train, val, days, output, batch, max_updates, eval_interval=16):
    seed_all(42)
    model=HistoryControlNet().cuda(); model.load_state_dict(state)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    d=output/arm; d.mkdir(parents=True,exist_ok=False)
    step0=aggregate_snapshot(model,val,days,arm,0)
    baseline=float(step0["metrics"]["patch"]["transport"]["thick_rmse"])
    best_score=baseline; selected_update=0; selected="step0_zero_residual_transport"
    best_learned=float("inf"); best_learned_update=None
    torch.save({"model":model.state_dict(),"update":0,"candidate":selected},d/"best.pt")
    (d/"step0.json").write_text(json.dumps(step0["metrics"],indent=2)+"\n")
    snapshots={0:step0}
    (d/"day_metrics.jsonl").write_text(json.dumps(step0,separators=(",",":"))+"\n")
    update=0; stale=0; epoch=0; evals=[]; start=time.time()
    while update<max_updates and stale<16:
        epoch+=1; order=np.random.permutation(512); loss_buf=[]
        for j in range(0,512,batch):
            if update>=max_updates: break
            ids=order[j:j+batch]
            h=train["h"][ids].cuda(); f=train["f"][ids].cuda(); y=train["y"][ids].cuda()
            mask=train["mask"][ids].cuda(); motion=train["motion"][ids].cuda(); adv=train["adv"][ids].cuda()
            pred=model(history_input(h,f,adv,motion,arm),adv); loss=loss_fn(pred,y,mask)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            loss_buf.append(float(loss.detach())); update+=1
            if update%eval_interval==0 or update==max_updates:
                snap=aggregate_snapshot(model,val,days,arm,update); snapshots[update]=snap
                score=float(snap["metrics"]["patch"]["learned"]["thick_rmse"])
                rec={"update":update,"epoch":epoch,"train_loss":float(np.mean(loss_buf)),"metrics":snap["metrics"],"lead_metrics":snap["lead_metrics"],"elapsed_seconds":time.time()-start}
                evals.append(rec); loss_buf=[]
                with (d/"epochs.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(rec)+"\n")
                with (d/"day_metrics.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(snap,separators=(",",":"))+"\n")
                if score<best_learned:
                    best_learned=score;best_learned_update=update
                    torch.save({"model":model.state_dict(),"update":update,"candidate":"best_learned_by_patch_thick_rmse"},d/"best_learned.pt")
                if score<best_score:
                    best_score=score;selected_update=update;selected=f"learned_update_{update}";stale=0
                    torch.save({"model":model.state_dict(),"update":update,"candidate":selected},d/"best.pt")
                else: stale+=1
    selected_snapshot=snapshots[selected_update]
    learned_snapshot=snapshots[best_learned_update] if best_learned_update is not None else step0
    report={"state":"COMPLETE_SILI_REPEAT_LAST_CONTROL_ARM","test_used":False,"arm":arm,
            "updates_run":update,"selected_candidate":selected,"selected_update":selected_update,
            "selected_metrics":selected_snapshot["metrics"],"selected_lead_metrics":selected_snapshot["lead_metrics"],
            "best_learned_update_by_patch_thick_rmse":best_learned_update,
            "best_learned_metrics":learned_snapshot["metrics"],"best_learned_lead_metrics":learned_snapshot["lead_metrics"],
            "step0_patch_thick_rmse":baseline,"profiled_batch":batch,"evals":evals}
    (d/"training_complete.json").write_text(json.dumps(report,indent=2)+"\n")
    del model,opt;torch.cuda.empty_cache()
    return report,selected_snapshot,learned_snapshot


def paired_ci(a,b,scope,regime="thick",reps=5000,seed=20260928,method_a="learned",method_b="learned"):
    days=sorted(set(a["day_sums"])&set(b["day_sums"]))
    rng=np.random.default_rng(seed);ix=rng.integers(0,len(days),(reps,len(days)))
    outs=[]
    for snap,name in ((a,method_a),(b,method_b)):
        sq=[];n=[]
        for day in days:
            q=snap["day_sums"][day][scope][name]["_by_regime"][regime]
            sq.append(q["sq"]);n.append(q["n"])
        sq=np.asarray(sq);n=np.asarray(n)
        outs.append(np.sqrt(sq[ix].sum(1)/n[ix].sum(1)))
    diff=outs[0]-outs[1]
    return {"days":len(days),"replicates":reps,"seed":seed,"ci95_full_minus_repeat_last":[float(x) for x in np.quantile(diff,[.025,.975])]}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--bank",type=Path,required=True);ap.add_argument("--sequence-days",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--max-updates",type=int,default=512);a=ap.parse_args()
    meta=json.loads((a.bank/"complete.json").read_text())
    if meta.get("state")!="COMPLETE_SILI_COT_DYNAMICS_BANK" or meta.get("test_used") is not False: raise RuntimeError("Unexpected/test-contaminated bank")
    side=np.load(a.sequence_days);days=side["init_day_BJT"].astype(str);ids=side["sequence_index"].astype(np.int64)
    if len(days)!=512 or not np.array_equal(ids,np.load(a.bank/"val"/"sequence_indices.npy").astype(np.int64)): raise RuntimeError("Validation day sidecar mismatch")
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    print("PREPARE_TRAIN",flush=True);tr=prepare(a.bank,"train",np.arange(512),"cuda")
    print("PREPARE_VAL",flush=True);va=prepare(a.bank,"val",np.arange(512),"cuda")
    seed_all(42);short=TransportThicknessNet().cuda();short_state={k:v.detach().cpu().clone() for k,v in short.state_dict().items()};init=matched_state(short_state)
    x=va["h"][:2];f=va["f"][:2];adv=va["adv"][:2];motion=va["motion"][:2]
    with torch.no_grad():
        expected=short(torch.cat((adv,f.permute(0,2,1,3,4),x[:,-1].unsqueeze(2).expand(-1,-1,16,-1,-1),(x[:,-1]-x[:,-2]).unsqueeze(2).expand(-1,-1,16,-1,-1),f.permute(0,2,1,3,4)-adv,torch.linspace(1/16,1,16,device=adv.device)[None,None,:,None,None].expand(2,1,16,64,64),(motion/4)[:,:,None,None,None].expand(-1,-1,16,64,64)),dim=1),adv)
        inp_full=history_input(x,f,adv,motion,"full_history");inp_repeat=history_input(x,f,adv,motion,"repeat_last")
        full0=init.cuda()(inp_full,adv);repeat0=init.cuda()(inp_repeat,adv)
        init_full=float((expected-full0).abs().max());init_repeat=float((expected-repeat0).abs().max())
    if max(init_full,init_repeat)>1e-6: raise RuntimeError(f"Init function mismatch: full={init_full}, repeat={init_repeat}")
    batch,profile=profile_batch(tr,init.state_dict())
    (a.output/"batch_profile.json").write_text(json.dumps({"selected_batch":batch,"reserve_gate_gib":70,"candidates":profile},indent=2)+"\n")
    state={k:v.detach().cpu().clone() for k,v in init.state_dict().items()}
    repeat,repeat_selected,repeat_best=run_arm("repeat_last_control",state,tr,va,days,a.output,batch,a.max_updates)
    full,full_selected,full_best=run_arm("full_history",state,tr,va,days,a.output,batch,a.max_updates)
    # Pair best learned checkpoints separately from step-0 selection; these CIs reuse validation for selection.
    contrasts={
        "full_minus_repeat_last_best_learned_patch_thick":paired_ci(full_best,repeat_best,"patch"),
        "full_minus_repeat_last_best_learned_station_thick":paired_ci(full_best,repeat_best,"station"),
        "full_best_learned_minus_transport_patch_thick":paired_ci(full_best,full_best,"patch",method_b="transport"),
        "full_best_learned_minus_transport_station_thick":paired_ci(full_best,full_best,"station",method_b="transport"),
        "repeat_best_learned_minus_transport_patch_thick":paired_ci(repeat_best,repeat_best,"patch",method_b="transport"),
        "repeat_best_learned_minus_transport_station_thick":paired_ci(repeat_best,repeat_best,"station",method_b="transport"),
    }
    report={"state":"COMPLETE_SILI_FULL_VS_REPEAT_LAST_HISTORY_PILOT","test_used":False,
            "bank_complete_sha256":__import__("hashlib").sha256((a.bank/"complete.json").read_bytes()).hexdigest(),
            "initial_function_max_abs_diff":{"full_history":init_full,"repeat_last":init_repeat},
            "same_architecture_and_parameter_count":True,"input_channels":15,
            "only_intended_difference":"all eight true historical COT maps versus eight repeated copies of the last map; both retain the same last-minus-previous trend channel",
            "selected_batch":batch,"batch_profile":profile,"validation_days":len(set(days)),
            "repeat_last_control":repeat,"full_history":full,"best_learned_paired_day_bootstrap_95pct":contrasts,
            "interpretation_limit":"best learned checkpoints are selected on this validation bank; paired intervals are descriptive and post-selection",
            "cpp_reference_not_independent_truth":True}
    (a.output/"training_complete.json").write_text(json.dumps(report,indent=2)+"\n")
    print("SILI_FULL_VS_REPEAT_LAST_COMPLETE",json.dumps({"batch":batch,"repeat_selected":repeat["selected_candidate"],"full_selected":full["selected_candidate"],"full_best_learned_update":full["best_learned_update_by_patch_thick_rmse"],"full_minus_repeat_ci":contrasts}),flush=True)


if __name__=="__main__":main()
