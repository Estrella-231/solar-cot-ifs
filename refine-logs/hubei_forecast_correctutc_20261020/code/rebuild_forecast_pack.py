"""Apply corrected fold/full R to frozen future AGRI, preserving every GHI row."""
import argparse,csv,hashlib,importlib.util,json,shutil
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
import torch

CENTERS={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser()
    for n in ('inputs','fold-packs','fold-runs','full-r','full-r-pack','model-source','source-pack','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--sequence-batch',type=int,default=32);a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('allocated GPU required')
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if a.output.exists():raise FileExistsError(a.output)
    rows=list(csv.DictReader((a.source_pack/'rows.csv').open()));old=json.loads((a.source_pack/'pack_report.json').read_text())
    if len(rows)!=70171 or old.get('train_forecast_inputs_are_oof') is not True:raise ValueError('source forecast contract')
    if any(r['split'] not in ('train','val') for r in rows):raise ValueError('test/buffer row')
    folds={r['date_BJT'].replace('-',''):int(r['heldout_fold']) for r in csv.DictReader((a.inputs/'hubei_oof_train_date_fold_20261016.csv').open())}
    spec=importlib.util.spec_from_file_location('correctutc_R',a.model_source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    a.output.mkdir(parents=True,exist_ok=False);banks=a.output/'cot_banks';banks.mkdir()
    lookup={};bank_arrays={};sources=[]
    for f in range(6):
        role=f'fold_{f}' if f<5 else 'validation';split='train' if f<5 else 'val';src=a.inputs/'forecast'/role
        rp=a.fold_packs/f'fold_{f}' if f<5 else a.full_r_pack;run=a.fold_runs/f'fold_{f}' if f<5 else a.full_r
        status=json.loads((run/'status.json').read_text());norm=json.loads((rp/'norm.json').read_text());meta=json.loads((run/'run_metadata.json').read_text())
        if status.get('state')!='COMPLETE_VALIDATION_ONLY_TEST_CLOSED' or meta.get('model')!='hubei_R_station16_correctutc_v1':raise ValueError('wrong/incomplete R')
        if meta['pack_status_sha256']!=sha(rp/'pack_status.json') or meta['model_source_sha256']!=sha(a.model_source):raise ValueError('R source mismatch')
        keys=list(csv.DictReader((src/'keys.csv').open()));pred=np.load(src/'pred_agri_patch.npy',mmap_mode='r',allow_pickle=False);geo=np.load(src/'future_geometry_patch.npy',mmap_mode='r',allow_pickle=False)
        if pred.shape!=(len(keys),16,13,16,16) or geo.shape!=(len(keys),16,3,16,16):raise ValueError('forecast tensor shape')
        for i,k in enumerate(keys):
            init=k['init_time_bjt'];day=init[:8]
            if (f<5 and folds.get(day)!=f) or (split,init) in lookup:raise ValueError('OOF fold/unique key')
            times=k['target_times_bjt'].split('|');dt=datetime.strptime(init,'%Y%m%d%H%M')
            if times!=[(dt+timedelta(minutes=15*j)).strftime('%Y%m%d%H%M') for j in range(1,17)]:raise ValueError('future times')
            lookup[split,init]=(role,i)
        model=mod.COTUNet().cuda().eval();ck=torch.load(run/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(ck['model'],strict=True)
        mean=np.asarray(norm['mean'],np.float32)[None,:,None,None];std=np.asarray(norm['std'],np.float32)[None,:,None,None]
        arr=np.lib.format.open_memmap(banks/f'{role}.npy',mode='w+',dtype=np.float32,shape=(len(keys),16,1,16,16))
        for start in range(0,len(keys),a.sequence_batch):
            stop=min(start+a.sequence_batch,len(keys));raw=np.concatenate((np.asarray(pred[start:stop]),np.asarray(geo[start:stop])),axis=2).reshape(-1,16,16,16)
            if not np.isfinite(raw).all():raise ValueError('nonfinite forecast input')
            day=raw[:,15:16]>.5;x=(raw-mean)/std;x[:,15:16]=np.where(day,x[:,15:16],0.)
            if not np.isfinite(x).all():raise ValueError('nonfinite normalized input')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):log=model(torch.from_numpy(x).cuda()).float()
            c=torch.expm1(log).masked_fill(~torch.from_numpy(day).cuda(),0.)
            if not torch.isfinite(c).all() or (c<0).any():raise RuntimeError('invalid COT')
            arr[start:stop]=c.cpu().numpy().reshape(stop-start,16,1,16,16)
        arr.flush();del arr,model,ck
        bank_arrays[role]=(np.load(banks/f'{role}.npy',mmap_mode='r'),pred,geo)
        sources.append({'role':role,'sequences':len(keys),'R_checkpoint_sha256':sha(run/'best.pt'),'R_pack_status_sha256':sha(rp/'pack_status.json'),'R_normalization_sha256':sha(rp/'norm.json'),'forecast_keys_sha256':sha(src/'keys.csv'),'forecast_pred_sha256':sha(src/'pred_agri_patch.npy'),'forecast_geometry_sha256':sha(src/'future_geometry_patch.npy'),'COT_sha256':sha(banks/f'{role}.npy')})
        print('CORRECTUTC_FORECAST_COT_BANK_COMPLETE',role,len(keys),flush=True)
    oldag=np.load(a.source_pack/'agri_raw.npy',mmap_mode='r');oldgeo=np.load(a.source_pack/'geometry.npy',mmap_mode='r')
    out=np.empty((len(rows),1,3,3),np.float32);max_ag=0.;max_ge=0.
    for j,r in enumerate(rows):
        role,i=lookup[r['split'],r['init_time_bjt']];lead=int(r['lead_minutes']);li=lead//15-1
        if lead not in range(15,241,15):raise ValueError('lead')
        if (datetime.strptime(r['init_time_bjt'],'%Y%m%d%H%M')+timedelta(minutes=lead)).strftime('%Y%m%d%H%M')!=r['target_time_bjt']:raise ValueError('GHI key time')
        y,x=CENTERS[r['station']];sl=(slice(y-1,y+2),slice(x-1,x+2));c,ag,ge=bank_arrays[role]
        max_ag=max(max_ag,float(np.max(np.abs(ag[i,li][:,sl[0],sl[1]]-oldag[j]))));max_ge=max(max_ge,float(np.max(np.abs(ge[i,li][:2,sl[0],sl[1]]-oldgeo[j]))))
        out[j]=c[i,li][:,sl[0],sl[1]]
    if max_ag>1e-6 or max_ge>1e-6:raise RuntimeError(('forecast reuse mismatch',max_ag,max_ge))
    np.save(a.output/'cot.npy',out,allow_pickle=False)
    exact={}
    for n in ('rows.csv','agri_raw.npy','geometry.npy','ghi_wm2.npy','clear_sky_wm2.npy'):
        shutil.copyfile(a.source_pack/n,a.output/n);exact[n]={'source_sha256':sha(a.source_pack/n),'new_sha256':sha(a.output/n)}
        if exact[n]['source_sha256']!=exact[n]['new_sha256']:raise RuntimeError('changed common input')
    report={'state':'HUBEI_GHI_FORECAST_HEADS_TRAINVAL_PACK_TEST_CLOSED','input_mode':'forecast','samples':len(rows),'split_sha256':old['split_sha256'],'train_forecast_inputs_are_oof':True,'oof_fold_count':5,'oof_train_bank_report_sha256':sha(a.inputs/'export_receipt.json'),'validation_forecast_source':'frozen_full_train_S_R','correctutc_cot_rebuilt':True,'source_legacy_pack_report_sha256':sha(a.source_pack/'pack_report.json'),'source_correctutc_Rs':sources,'same_rows_and_non_cot_arrays':exact,'forecast_AGRI_reuse_max_abs_difference':max_ag,'future_geometry_reuse_max_abs_difference':max_ge,'arrays_sha256':{n:sha(a.output/n) for n in ('agri_raw.npy','geometry.npy','cot.npy','ghi_wm2.npy','clear_sky_wm2.npy')},'future_real_agri_used_as_input':False,'future_cpp_used_as_input':False,'test_and_buffer_values_read':0,'test_payloads_read':0,'buffer_payloads_read':0}
    (a.output/'pack_report.json').write_text(json.dumps(report,indent=2));print('CORRECTUTC_FORECAST_GHI_PACK_COMPLETE',len(rows),max_ag,max_ge,flush=True)
if __name__=='__main__':main()
