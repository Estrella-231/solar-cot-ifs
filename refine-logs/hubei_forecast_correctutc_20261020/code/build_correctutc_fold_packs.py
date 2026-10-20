"""Replace wrong-time CPP targets; retain exact fold-specific AGRI normalization."""
import argparse,csv,hashlib,json,shutil
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=list(csv.DictReader((a.source/'rows.csv').open())); by={x['time_bjt']:i for i,x in enumerate(rows)}
    status=json.loads((a.source/'pack_status.json').read_text())
    if status.get('state')!='HUBEI_R_TRAIN_VAL_CORRECT_UTC_PACK_BUILT_TEST_CLOSED' or status.get('time_rule')!='CPP UTC = BJT - 8 hours': raise ValueError('not corrected UTC')
    targets=np.load(a.source/'target.npy',allow_pickle=False); masks=np.load(a.source/'mask.npy',allow_pickle=False)
    folds={x['date_BJT'].replace('-',''):int(x['heldout_fold']) for x in csv.DictReader((a.inputs/'hubei_oof_train_date_fold_20261016.csv').open())}
    a.output.mkdir(parents=True,exist_ok=False); records=[]
    for fold in range(5):
        src=a.inputs/f'fold_inputs/fold_{fold}'; old=list(csv.DictReader((src/'rows.csv').open())); norm=json.loads((src/'norm.json').read_text())
        if norm.get('fit_split')!='train only' or norm.get('test_used') is not False: raise ValueError('normalization')
        take=[]; new=[]
        for i,r in enumerate(old):
            k=r['time_bjt']; j=by[k]; q=dict(rows[j]); assert q['split']==r['split']
            if r['split']=='train' and folds[k[:8]]==fold: raise ValueError('heldout date used for training')
            bjt=datetime.strptime(k,'%Y%m%d%H%M'); utc=bjt-timedelta(hours=8)
            stamp=q['time_cpp_utc'].replace('-','').replace(':','').replace(' ','')[:12]
            if stamp!=utc.strftime('%Y%m%d%H%M'): raise ValueError(('UTC',q))
            q['index']=i; new.append(q); take.append(j)
        if norm.get('train_samples')!=sum(x['split']=='train' for x in old): raise ValueError('fold norm count')
        out=a.output/f'fold_{fold}';out.mkdir()
        for n in ('x_norm.npy','norm.json'): shutil.copyfile(src/n,out/n)
        np.save(out/'target.npy',targets[take],allow_pickle=False);np.save(out/'mask.npy',masks[take],allow_pickle=False)
        with (out/'rows.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(new[0]));w.writeheader();w.writerows(new)
        counts={s:{'rows':sum(r['split']==s for r in old),'supervised_rows':sum(r['split']==s and masks[j].any() for r,j in zip(old,take))} for s in ('train','val')}
        rec={'state':'HUBEI_R_TRAIN_VAL_CORRECT_UTC_PACK_BUILT_TEST_CLOSED','fold':fold,'rows':len(new),'counts_by_split':counts,'source_correctutc_pack_sha256':sha(a.source/'pack_status.json'),'normalization':'identical original fold train-only AGRI input normalization','heldout_train_date_overlap':0,'time_rule':status['time_rule'],'test_payloads_read':0,'buffer_payloads_read':0}
        (out/'pack_status.json').write_text(json.dumps(rec,indent=2));(out/'SHA256.json').write_text(json.dumps({n:sha(out/n) for n in ('rows.csv','x_norm.npy','norm.json','target.npy','mask.npy')},indent=2))
        records.append(rec)
    (a.output/'fold_pack_report.json').write_text(json.dumps({'state':'FIVE_CORRECTUTC_FOLD_R_PACKS_COMPLETE','folds':records,'test_payloads_read':0,'buffer_payloads_read':0},indent=2));print(json.dumps(records),flush=True)
if __name__=='__main__':main()
