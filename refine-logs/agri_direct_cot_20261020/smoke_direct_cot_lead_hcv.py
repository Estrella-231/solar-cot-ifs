import json, pathlib, numpy as np
import run_direct_cot_lead_hcv as m

HERE=pathlib.Path(__file__).resolve().parent
BANK=pathlib.Path(r'C:\Users\29728\AppData\Local\Temp\direct_cot_rowbank_recovery_20261021_0530\results\direct_cot_forecast_row_bank_v1')
OUT=HERE/'logs'/'lead_hcv_smoke_20261021'
st='sili';d=BANK/st/'train';keys=m.read_jsonl(d/'keys.jsonl')
allkeys={s:m.read_jsonl(BANK/s/'train'/'keys.jsonl') for s in m.STATIONS}
folds=m.assign_folds(allkeys);held={day for day,f in folds.items() if f==0}
blocked={day for day in folds if any(abs((day-h).days)<=1 for h in held)}
dates=[m.bjt_date(k) for k in keys]
vi=np.asarray([i for i,day in enumerate(dates) if day in held]);ti=np.asarray([i for i,day in enumerate(dates) if day not in blocked])
raw={'agri':np.load(d/'agri_raw.npy',mmap_mode='r'),'geometry':np.load(d/'geometry_raw.npy',mmap_mode='r'),
 'ghi':np.load(d/'ghi.npy',mmap_mode='r'),'clear':np.load(d/'clear.npy',mmap_mode='r'),
 'p_xc':np.load(d/'cot_raw'/'p_xc.npy',mmap_mode='r'),
 'lead':np.asarray([int(k['lead_minutes']) for k in keys],dtype=np.float32)}
normed=m.normalize_fold(raw,ti,vi);norm=normed.pop('norm');tr=m.tensor_data(normed,ti);va=m.tensor_data(normed,vi)
OUT.mkdir(parents=True,exist_ok=False)
rec=m.train_one(st,0,'pxc_lead',42,tr,va,OUT,m.sha(BANK/'bank_manifest.json'),norm)
receipt={'state':'PASS_ONE_REAL_H_FOLD_TRAIN_SMOKE','record':rec,'test_payload_opened':False,
 'buffer_payload_opened':False,'future_observed_agri_used_as_input':False,'cpp_used_as_input':False}
(OUT/'smoke_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2),flush=True)
