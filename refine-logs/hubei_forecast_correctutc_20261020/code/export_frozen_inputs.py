"""Export only existing train/val forecast AGRI and fold-R input arrays."""
import argparse, csv, hashlib, json, tarfile
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True,type=Path); p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    r=a.root; inp=r/'manifests/oof_fold_manifests_20261016_v5/input'
    folds={x['date_BJT'].replace('-',''):int(x['heldout_fold']) for x in csv.DictReader((inp/'hubei_oof_train_date_fold_20261016.csv').open())}
    days={x['date_BJT'].replace('-',''):x['split'] for x in csv.DictReader((inp/'hubei_day_assignment_20261016.csv').open())}
    names=['S_oof_fold0_oof_bank_fd201_20261017_raw_v5_attempt2_gpu0','S_oof_fold1_oof_bank_fd201_20261017_raw_v5_attempt2_gpu1','S_oof_fold2_oof_bank_fd201_20261017_raw_v5_attempt2_gpu2','S_oof_fold3_oof_bank_fd201_20261017_raw_v5_tc6000_import_gpu0','S_oof_fold4_oof_bank_fd201_20261017_raw_v5_tc6000_import_gpu0']
    files=[]; groups=[]
    for i in range(6):
        src=r/'oof_training'/names[i] if i<5 else r/'forecast_bank/strict24_same_day_v3_best_epoch10_fulltrainval_20261016'
        keys=list(csv.DictReader((src/'val/keys.csv').open()))
        for k in keys:
            d=k['init_time_bjt'][:8]
            if (i<5 and (days.get(d)!='train' or folds.get(d)!=i)) or (i==5 and days.get(d)!='val'): raise ValueError(('split/fold',i,d))
            if any(t[:8]!=d for t in k['target_times_bjt'].split('|')): raise ValueError('cross-day sequence')
        role=f'fold_{i}' if i<5 else 'validation'
        for n in ('keys.csv','pred_agri_patch.npy','future_geometry_patch.npy'):
            files.append((src/'val'/n,f'forecast/{role}/{n}'))
        files.append((src/'BANK_REPORT.json',f'forecast/{role}/source_BANK_REPORT.json'))
        groups.append({'role':role,'source':str(src),'sequences':len(keys),'keys_sha256':sha(src/'val/keys.csv')})
        if i<5:
            rp=r/f'manifests/oof_fold_manifests_20261016_v5/folds/fold_{i}/r_pack'
            rows=list(csv.DictReader((rp/'rows.csv').open()))
            if any(x['split'] not in ('train','val') or (x['split']=='train' and folds.get(x['time_bjt'][:8])==i) for x in rows): raise ValueError('fold R overlap')
            for n in ('x_norm.npy','rows.csv','norm.json'): files.append((rp/n,f'fold_inputs/fold_{i}/{n}'))
    for n in ('hubei_oof_train_date_fold_20261016.csv','hubei_day_assignment_20261016.csv'): files.append((inp/n,n))
    receipt={'state':'FROZEN_FORECAST_INPUT_EXPORT_TRAINVAL_ONLY','groups':groups,'test_payloads_read':0,'buffer_payloads_read':0,'true_future_agri_exported':False,'old_cpp_target_mask_exported':False,'files':{dst:{'source':str(src),'bytes':src.stat().st_size,'sha256':sha(src)} for src,dst in files}}
    (a.output/'export_receipt.json').write_text(json.dumps(receipt,indent=2))
    with tarfile.open(a.output/'frozen_forecast_inputs.tar','w') as tar:
        for src,dst in files: tar.add(src,arcname=dst,recursive=False)
        tar.add(a.output/'export_receipt.json',arcname='export_receipt.json')
    (a.output/'archive_sha256.txt').write_text(sha(a.output/'frozen_forecast_inputs.tar')+'  frozen_forecast_inputs.tar\n')
    print(json.dumps({'state':receipt['state'],'groups':groups,'archive_bytes':(a.output/'frozen_forecast_inputs.tar').stat().st_size}),flush=True)
if __name__=='__main__': main()
