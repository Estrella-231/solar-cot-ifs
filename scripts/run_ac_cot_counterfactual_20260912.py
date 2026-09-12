"""Frozen AC-checkpoint COT counterfactual forwards; validation only, no training."""
import argparse, csv, hashlib, json, time
from pathlib import Path

import numpy as np
import torch

from train_head_ghi_multiseed_core import Head

ROOT = Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
OUT = ROOT / 'runs/ac_cot_counterfactual_20260912_v1'
WORK = ROOT / 'runs/ac_cot_counterfactual_20260912_v1.incomplete'
FEATURES = ['mean_log1p_cot', 'std_log1p_cot', 'p90_log1p_cot', 'center_log1p_cot']
VARIANTS = ['original', 'zero_all', 'sunny_median', 'drop_mean', 'drop_std', 'drop_p90', 'drop_center']
EXPECTED_PACK_AUDIT_SHA = '3de1c9d622bc5838ef11db2347c675d4b7db9b8e1106ce7ac5a478215e7ee08c'
PROTOCOL = 'docs/AC_COT_COUNTERFACTUAL_PROTOCOL_20260912.md'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def coverage_label(counts, n):
    if n < 231:
        return -1
    cloud = counts[1] + counts[2]
    return 0 if cloud <= .2*n else 2 if cloud >= .8*n else 1


def cpu_check():
    assert coverage_label([256, 0, 0], 256) == 0
    torch.manual_seed(42)
    model = Head(True).eval()
    n = 5
    inputs = dict(x=torch.randn(n, 13, 16, 16), c=torch.randn(n, 4),
                  g=torch.randn(n, 3), station=torch.tensor([0, 1, 0, 1, 0]),
                  lead=torch.arange(n))
    with torch.no_grad():
        original = model(*(inputs[k] for k in ('x', 'c', 'g', 'station', 'lead')))
        changed = inputs['c'].clone(); changed[:, 1] = 0
        dropped = model(inputs['x'], changed, inputs['g'], inputs['station'], inputs['lead'])
    assert original.shape == dropped.shape == (n,)
    assert torch.isfinite(original).all() and torch.isfinite(dropped).all()
    assert not torch.equal(original, dropped)
    print('AC_COT_COUNTERFACTUAL_CPU_WITNESS_PASS', flush=True)


def metrics(pred_kt, truth, clear, mask):
    n = int(mask.sum())
    if n == 0:
        return dict(n=0, rmse=None, mae=None, bias=None, mean_predicted_ghi=None)
    pred = pred_kt[mask].astype(np.float64) * clear[mask]
    err = pred - truth[mask]
    return dict(n=n, rmse=float(np.sqrt(np.mean(err**2))),
                mae=float(np.mean(np.abs(err))), bias=float(np.mean(err)),
                mean_predicted_ghi=float(np.mean(pred)))


def intervention(name, c, station, lead, sunny):
    if name == 'original':
        return c
    out = c.copy()
    if name == 'zero_all':
        out.fill(0)
    elif name == 'sunny_median':
        out[:] = sunny[station, lead]
    else:
        index = {'drop_mean': 0, 'drop_std': 1, 'drop_p90': 2, 'drop_center': 3}[name]
        out[:, index] = 0
    return out


def main():
    if OUT.exists() or WORK.exists():
        raise FileExistsError(f'refuse overwrite or ambiguous retry: {OUT} / {WORK}')
    WORK.mkdir(parents=True)
    started = time.monotonic()
    pack = ROOT / 'data/head_pack_trainval_20260908_v1'
    audit_path = pack / 'audit.json'
    audit = json.loads(audit_path.read_text())
    assert sha(audit_path) == EXPECTED_PACK_AUDIT_SHA
    assert audit['state'] == 'COMPLETE' and audit['test_used'] is False
    assert audit['station_order'] == ['sili', 'zhujia']
    assert audit['lead_values'] == '0..15 means +15..+240 min'
    norm_path = pack / 'norm.json'; norm = json.loads(norm_path.read_text())
    assert sha(norm_path) == audit['norm_sha256']
    assert norm['fit_split'] == 'train' and norm['c_features'] == FEATURES

    keys = ('x', 'c', 'g', 'ghi', 'clear', 'station', 'lead', 'split')
    arrays = {}
    for key in keys:
        spec = audit['arrays'][key]; path = pack / spec['name']
        assert sha(path) == spec['sha256'], key
        arrays[key] = np.load(path, mmap_mode='r', allow_pickle=False)
    split = np.asarray(arrays['split']); train = split == 0; val_rows = np.flatnonzero(split == 1)
    assert int(train.sum()) == 538053 and len(val_rows) == 99849

    # Build station x lead sunny reference from original-training rows only.
    labels = ROOT / 'data/weather_clp_trainval_20260912_v1'
    selection = json.loads((labels/'selection.json').read_text())
    collection = json.loads((labels/'collection.json').read_text())
    assert selection['pack_audit_sha256'] == sha(audit_path)
    assert selection['mapping_sha256'] == sha(labels/'row_target_id.npy')
    assert collection['labels_sha256'] == sha(labels/'labels.jsonl')
    mapping = np.load(labels/'row_target_id.npy', allow_pickle=False)
    records = [json.loads(v) for v in (labels/'labels.jsonl').read_text().splitlines()]
    target_y = np.full(len(records), -1, np.int8)
    for i, row in enumerate(records):
        assert row['target_id'] == i
        if row['state'] == 'READ':
            target_y[i] = coverage_label(row['class_counts'], row['valid_pixels'])
    y = target_y[mapping]
    sunny = np.empty((2, 16, 4), np.float32); sunny_counts = np.empty((2, 16), np.int64)
    c_all = arrays['c']; station_all = arrays['station']; lead_all = arrays['lead']
    for station in range(2):
        for lead in range(16):
            mask = train & (y == 0) & (station_all == station) & (lead_all == lead)
            sunny_counts[station, lead] = mask.sum()
            assert sunny_counts[station, lead] >= 100
            sunny[station, lead] = np.median(np.asarray(c_all[mask]), axis=0)
    assert np.isfinite(sunny).all()

    # Common validation payload is copied once and never mutated.
    val = {k: np.asarray(arrays[k][val_rows]) for k in ('x','c','g','ghi','clear','station','lead')}
    assert all(np.isfinite(v).all() for v in val.values())
    weather_run = ROOT/'runs/weather_coverage_head_seed42_20260912_v2'
    weather_done = json.loads((weather_run/'complete.json').read_text())
    weather_path = weather_run/'validation_predictions.npz'
    assert sha(weather_path) == weather_done['predictions_sha256']
    with np.load(weather_path, allow_pickle=False) as w:
        assert np.array_equal(w['pack_row'], val_rows)
        weather = w['predicted_class'].astype(np.int8)
    assert np.isin(weather, [0,1,2]).all()

    device = torch.device('cuda')
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(1); torch.backends.cudnn.benchmark = True
    tensors = {k: torch.from_numpy(v).to(device) for k,v in val.items() if k in ('x','g','station','lead')}
    batch = 2048

    @torch.no_grad()
    def predict(model, c_value):
        model.eval(); parts=[]
        for start in range(0, len(val_rows), batch):
            stop = min(start+batch, len(val_rows)); ix = slice(start, stop)
            tc = torch.from_numpy(c_value[ix]).to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p = model(tensors['x'][ix], tc, tensors['g'][ix], tensors['station'][ix], tensors['lead'][ix])
            parts.append(p.float().cpu().numpy())
        return np.concatenate(parts)

    all_predictions = np.empty((3, len(VARIANTS), len(val_rows)), np.float32)
    rows=[]; sources={}; reproduction=[]
    groups = {'all': np.ones(len(val_rows), bool),
              'predicted_sunny': weather == 0,
              'predicted_partly_cloudy': weather == 1,
              'predicted_overcast': weather == 2}
    for seed_index, seed in enumerate((42,43,44)):
        dirname = 'head_ghi_original_seed42_20260910_v1' if seed == 42 else f'head_ghi_multiseed_original_seed{seed}_20260910_v1'
        run = ROOT/'runs'/dirname/f'AC_lr0.0003_seed{seed}'
        done = json.loads((run/'complete.json').read_text()); ckpt = run/'best.pt'; saved = run/'validation_predictions.npz'
        assert done['test_used'] is False and sha(ckpt) == done['checkpoint_sha256'] and sha(saved) == done['predictions_sha256']
        ck = torch.load(ckpt, map_location='cpu'); assert ck['group'] == 'AC' and ck['seed'] == seed and ck['lr'] == .0003
        model = Head(True).to(device).eval(); model.load_state_dict(ck['model'], strict=True)
        sources[str(seed)] = dict(checkpoint=str(ckpt), checkpoint_sha256=sha(ckpt), saved_predictions_sha256=sha(saved))
        original = None
        for variant_index, variant in enumerate(VARIANTS):
            changed = intervention(variant, val['c'], val['station'], val['lead'], sunny)
            pred = predict(model, changed); assert np.isfinite(pred).all()
            all_predictions[seed_index, variant_index] = pred
            if variant == 'original':
                original = pred
                with np.load(saved, allow_pickle=False) as z:
                    assert np.array_equal(z['pack_row'], val_rows)
                    max_error = float(np.max(np.abs(pred-z['pred_kt'])))
                    ghi_rmse_error = float(abs(np.sqrt(np.mean((pred.astype(np.float64)*val['clear']-val['ghi'])**2)) -
                                                   np.sqrt(np.mean((z['pred_kt'].astype(np.float64)*val['clear']-val['ghi'])**2))))
                assert max_error <= .005 and ghi_rmse_error <= .05
                reproduction.append(dict(seed=seed,max_abs_pred_kt_error=max_error,pooled_ghi_rmse_abs_error=ghi_rmse_error))
            assert original is not None
            shift = (pred.astype(np.float64)-original.astype(np.float64))*val['clear']
            for station, station_name in enumerate(('sili','zhujia')):
                for group_name, group_mask in groups.items():
                    mask = group_mask & (val['station'] == station)
                    score = metrics(pred,val['ghi'],val['clear'],mask)
                    base = metrics(original,val['ghi'],val['clear'],mask)
                    row = dict(seed=seed,station=station_name,group=group_name,variant=variant,**score,
                               rmse_delta_vs_original=(score['rmse']-base['rmse'] if score['rmse'] is not None else None),
                               mae_delta_vs_original=(score['mae']-base['mae'] if score['mae'] is not None else None),
                               mean_ghi_shift_vs_original=float(shift[mask].mean()) if mask.any() else None,
                               fraction_ghi_shift_positive=float((shift[mask]>0).mean()) if mask.any() else None)
                    rows.append(row)
        del model

    predictions_path = WORK/'counterfactual_predictions.npz'
    np.savez_compressed(predictions_path, pack_row=val_rows, seeds=np.array([42,43,44]),
        variants=np.array(VARIANTS), pred_kt=all_predictions, observed_ghi=val['ghi'],
        clear_sky_ghi=val['clear'], station=val['station'], lead=val['lead'],
        predicted_weather=weather, sunny_median_normalized=sunny, sunny_median_counts=sunny_counts)
    fields=list(rows[0]); csv_path=WORK/'metrics.csv'
    with csv_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

    summary=[]
    for station in ('sili','zhujia'):
        for group in groups:
            for variant in VARIANTS:
                selected=[r for r in rows if r['station']==station and r['group']==group and r['variant']==variant]
                item=dict(station=station,group=group,variant=variant,seeds=[42,43,44],n=selected[0]['n'])
                for key in ('rmse','mae','bias','mean_ghi_shift_vs_original','rmse_delta_vs_original','mae_delta_vs_original','fraction_ghi_shift_positive'):
                    values=np.array([r[key] for r in selected],np.float64)
                    item[key]=dict(values=values.tolist(),mean=float(values.mean()),std_ddof1=float(values.std(ddof=1)))
                summary.append(item)
    focus=[r for r in summary if r['station']=='sili' and r['group']=='all' and r['variant'].startswith('drop_')]
    focus_sunny=[r for r in summary if r['station']=='sili' and r['group']=='predicted_sunny' and r['variant'].startswith('drop_')]
    report=dict(state='COMPLETE_AC_COT_COUNTERFACTUAL_FORWARD',test_used=False,
        scope='Hunan validation original cohort; fixed AC checkpoints seeds42/43/44; forecast AGRI; canonical +15...+240 min',
        feature_order=FEATURES,variants=VARIANTS,intervention_semantics={
            'zero_all':'standardized zero equals original-training physical feature mean',
            'sunny_median':'station x lead coordinate median from original-training CLP-derived sunny rows only',
            'drop_*':'one named standardized coordinate set to zero; all other COT and non-COT inputs unchanged'},
        sunny_median_counts=sunny_counts.tolist(),sunny_median_normalized=sunny.tolist(),
        original_prediction_reproduction=reproduction,sources=sources,
        weather_predictions_sha256=sha(weather_path),pack_audit_sha256=sha(audit_path),norm_sha256=sha(norm_path),
        protocol_sha256=sha(ROOT/PROTOCOL),script_sha256=sha(__file__),predictions_sha256=sha(predictions_path),metrics_sha256=sha(csv_path),
        sili_all_single_drop=focus,sili_predicted_sunny_single_drop=focus_sunny,summary=summary,
        caveats=['validation not test','model-input intervention, not physical causal proof','single-feature effects can interact and need not add','weather groups use one causal forecast-AGRI classifier seed42','CPP producer checkpoint binding unresolved'],
        seconds=time.monotonic()-started)
    atomic(WORK/'report.json',report)
    WORK.replace(OUT)
    print('AC_COT_COUNTERFACTUAL_FORWARD_COMPLETE',json.dumps(dict(seconds=report['seconds'],reproduction=reproduction,
        sili_all_single_drop=focus,sili_predicted_sunny_single_drop=focus_sunny)),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cpu-check',action='store_true');args=parser.parse_args()
    if args.cpu_check:cpu_check()
    else:main()
