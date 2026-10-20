"""Authorized solar-repair stages. Every output is versioned and test-closed."""
import argparse,csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
BASE=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021')
OLD=BASE.parent/'hubei_forecast_correctutc_20261020/time_repair_v1/runtime_attempt3'
CODE=Path(__file__).resolve().parent;DATA=BASE/'prepared_attempt12';REPO=BASE/'simvp_code_snapshot';ROLES=['full']+[f'fold_{i}' for i in range(5)]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def records(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def run(cmd,**kw):print('COMMAND',json.dumps(list(map(str,cmd))),flush=True);subprocess.run(list(map(str,cmd)),check=True,**kw)
def readready():
 d=json.loads((DATA/'DATA_READY.json').read_text());assert d['state']=='CORRECTED_SOLAR_TRAIN_VAL_DATA_READY' and d['test_payloads_read']==0 and d['buffer_payloads_read']==0
 audit=json.loads((DATA/'DATA_AUDIT_PASS_attempt12.json').read_text());assert audit['state']=='PASS_PREPARED_ATTEMPT12_TRAIN_VAL_KEYS_LABELS_AGRI_GEOMETRY' and audit['test_payloads_read']==0 and audit['buffer_payloads_read']==0
 return d
def profile():
 ready=readready();out=BASE/'S_profile_attempt12';out.mkdir(exist_ok=False);result=[];env=dict(os.environ,SIMVP_REPO=str(REPO))
 for b in (4,6,8,12,16):
  cmd=[sys.executable,CODE/'profile_S.py','--batch-size',b,'--index-csv',DATA/'indexes/full.csv','--norm-npz',DATA/'norms/full.npz','--data-root',ready['S_cache'],'--report-json',out/f'batch{b}.json','--checkpoint',out/f'batch{b}.pt']
  with (out/f'batch{b}.log').open('w') as log:p=subprocess.run(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT,env=env)
  if p.returncode:
   text=(out/f'batch{b}.log').read_text();assert 'out of memory' in text.lower(),text[-4000:];result.append(dict(batch=b,state='OOM'));continue
  r=json.loads((out/f'batch{b}.json').read_text());result.append(dict(batch=b,state='PASS',peak_GiB=r['peak_allocated_bytes']/1024**3,seconds=r['forward_backward_optimizer_step_seconds'],reload_delta=r['reload_prediction_max_abs']))
 feasible={x['batch'] for x in result if x['state']=='PASS' and x['peak_GiB']<=76.0};assert 4 in feasible;choice={}
 for role in ROLES:
  target=16 if role=='full' else 12
  if target in feasible:choice[role]=dict(gpus=1,micro_batch=target,global_batch=target)
  elif target//2 in feasible:choice[role]=dict(gpus=2,micro_batch=target//2,global_batch=target)
  else:raise RuntimeError(('No shape preserves original global batch',role,feasible))
 (BASE/'S_PROFILE_READY_attempt12.json').write_text(json.dumps(dict(state='CORRECTED_SOLAR_S_PROFILE_COMPLETE',results=result,choice=choice,model='unchanged SimVP gSTA 8-to-16 256x256 13+3',test_payloads_read=0),indent=2));print('S_PROFILE_READY',flush=True)
def R():
 readready();root=BASE/'R_runs_attempt12';root.mkdir(exist_ok=False)
 for role in ROLES:run([sys.executable,CODE/'train_corrected_r.py','--pack',DATA/'R_packs'/role,'--model-source',CODE/'train_cot_repaired.py','--output',root/role,'--seed',42,'--batch',128,'--epochs',50,'--patience',8,'--allow-variable-support'])
 (BASE/'R_READY_attempt12.json').write_text(json.dumps(dict(state='CORRECTED_SOLAR_SIX_R_COMPLETE',roles=ROLES,test_payloads_read=0)));print('SIX_R_READY',flush=True)
def S(role):
 ready=readready();shape=json.loads((BASE/'S_PROFILE_READY_attempt12.json').read_text())['choice'][role];ng=shape['gpus'];physical=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(physical)==ng
 cfg=BASE/'configs'/f'{role}.py';cfg.parent.mkdir(exist_ok=True);args=dict(method='SimVP',model_type='gSTA',dataname='hbagri',pre_seq_length=8,aft_seq_length=16,total_length=24,temporal_stride=1,in_shape=[8,16,256,256],out_channels=13,metrics=['mse','mae'],hid_S=32,hid_T=256,N_T=8,N_S=4,spatio_kernel_enc=3,spatio_kernel_dec=3,drop_path=.1,lr=1e-3,opt='adamw',weight_decay=1e-4,sched='cosine',min_lr=1e-5,warmup_epoch=5,warmup_lr=1e-5,filter_bias_and_bn=False,early_stop_patience=5,metric_for_bestckpt='val_loss',no_display_method_info=True,batch_size=shape['micro_batch'],val_batch_size=shape['micro_batch'],num_workers=4,drop_last=False,gpus=list(range(ng)),seed=42,epoch=50,data_root=ready['S_cache'],index_csv=str(DATA/'indexes'/f'{role}.csv'))
 cfg.write_text('\n'.join(f'{k} = {v!r}' for k,v in args.items())+'\n');out=BASE/'S_runs_attempt12'/role;out.mkdir(parents=True,exist_ok=False);(out/'cwd').mkdir();env=dict(os.environ,SIMVP_REPO=str(REPO),HUBEI_SIMVP_CONFIG=str(cfg),HUBEI_SIMVP_NORM=str(DATA/'norms'/f'{role}.npz'),HUBEI_SIMVP_RUN_CWD=str(out/'cwd'),HUBEI_SOLAR_RUN_DIR=str(out),HUBEI_SOLAR_ROLE=role,HUBEI_SOLAR_DATA_READY=str(DATA/'DATA_READY.json'))
 cmd=[sys.executable]
 if ng>1:cmd+=['-m','torch.distributed.run','--standalone','--nnodes=1',f'--nproc_per_node={ng}']
 cmd+=[CODE/'run_corrected_simvp.py']
 if ng>1:cmd+=['--dist']
 cmd+=['--gpus']+list(range(ng))+['--res_dir',BASE/'S_runs_attempt12','--ex_name',role];run(cmd,env=env)
 meta=json.loads((out/'run_metadata.json').read_text());assert meta['state']=='CORRECTED_SOLAR_SIMVP_TRAINING_COMPLETE'
 # No future-real fields are deployed; stored truth is for validation diagnosis only.
 index=DATA/'indexes'/('full.csv' if role=='full' else f'{role}_oof.csv')
 bank=BASE/'S_banks_attempt12'/role
 run([sys.executable,CODE/'infer_S.py','--repo',REPO,'--index',index,'--norm',DATA/'norms'/f'{role}.npz','--checkpoint',meta['best_checkpoint'],'--output',bank,'--batch-size',16,'--workers',4,'--splits','val'],env=env)
 (out/'S_BANK_READY.json').write_text(json.dumps(dict(state='CORRECTED_SOLAR_S_BANK_COMPLETE',role=role,checkpoint_sha256=meta['best_checkpoint_sha256'],bank=str(bank/'val'),source_index_sha256=sha(index),normalization_sha256=sha(DATA/'norms'/f'{role}.npz'),solar_geometry_corrected=True,test_payloads_read=0)));print('S_BANK_READY '+role,flush=True)
def GHI():
 import torch
 from train_cot_repaired import COTUNet
 readready();assert json.loads((BASE/'R_READY_attempt12.json').read_text())['state']=='CORRECTED_SOLAR_SIX_R_COMPLETE';pack=BASE/'GHI_pack_attempt12';pack.mkdir(exist_ok=False);cbank=BASE/'COT_banks_attempt12';cbank.mkdir();saved={};source=[];device=torch.device('cuda');torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 for role in ROLES:
  meta=json.loads((BASE/'S_runs_attempt12'/role/'S_BANK_READY.json').read_text());assert meta['solar_geometry_corrected'];bank=Path(meta['bank']);pred=np.load(bank/'pred_agri_patch.npy');geo=np.load(bank/'future_geometry_patch.npy');keys=records(bank/'keys.csv');n=len(keys);expected=407 if role=='full' else [823,797,850,809,807][int(role[-1])];assert n==expected and pred.shape==(n,16,13,16,16) and geo.shape==(n,16,3,16,16)
  sm=json.loads((BASE/'S_runs_attempt12'/role/'run_metadata.json').read_text());assert sm['solar_geometry_corrected'] and sm['random_initialization'] and sm['input_normalization_sha256']==sha(DATA/'norms'/f'{role}.npz') and sm['index_sha256']==sha(DATA/'indexes'/f'{role}.csv') and sm['best_checkpoint_sha256']==sha(sm['best_checkpoint'])==meta['checkpoint_sha256']
  rmeta=json.loads((BASE/'R_runs_attempt12'/role/'run_metadata.json').read_text());rstatus=json.loads((BASE/'R_runs_attempt12'/role/'status.json').read_text());assert rstatus['state']=='COMPLETE_VALIDATION_ONLY_TEST_CLOSED' and rmeta['pack_status_sha256']==sha(DATA/'R_packs'/role/'pack_status.json') and rmeta['model_source_sha256']==sha(CODE/'train_cot_repaired.py')
  rp=DATA/'R_packs'/role;rn=json.loads((rp/'norm.json').read_text());mu=np.array(rn['mean'],np.float32)[None,:,None,None];sd=np.array(rn['std'],np.float32)[None,:,None,None];rout=BASE/'R_runs_attempt12'/role;model=COTUNet().to(device).eval();model.load_state_dict(torch.load(rout/'best.pt',map_location=device)['model']);raw=np.concatenate((pred,geo),axis=2).reshape(-1,16,16,16);answer=[]
  for j in range(0,len(raw),2048):
   x=raw[j:j+2048]
   day=x[:,15:16]>.5
   valid=day & (x[:,13:14]>np.cos(np.deg2rad(80.0)))
   # Match train_corrected_r.py exactly: all 16 channels, including day_mask,
   # stay train-normalized. Replacing this channel with binary 0/1 here shifts
   # it away from the representation used to fit and validate the R model.
   z=(x-mu)/sd
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
    v=model(torch.from_numpy(z).to(device)).float()
    v=torch.expm1(v).masked_fill(~torch.from_numpy(valid).to(device),0.)
   answer.append(v.cpu().numpy())
  cot=np.concatenate(answer).reshape(n,16,1,16,16);assert np.isfinite(cot).all() and (cot>=0).all();name='validation' if role=='full' else role;np.save(cbank/f'{name}.npy',cot);saved[name]=(pred,geo,cot,keys);source.append(dict(role=name,corrected_S_checkpoint_sha256=meta['checkpoint_sha256'],R_checkpoint_sha256=sha(rout/'best.pt'),R_normalization_sha256=sha(rp/'norm.json'),R_inference_code_sha256=sha(Path(__file__)),R_input_preprocessing='train-fitted per-channel z-score for all 16 channels including day_mask; no binary override',R_output_validity='day_mask>0.5 and cosSOZ>cos(80deg)',forecast_pred_sha256=sha(bank/'pred_agri_patch.npy'),forecast_geometry_sha256=sha(bank/'future_geometry_patch.npy'),COT_sha256=sha(cbank/f'{name}.npy')))
 (cbank/'source_manifest.json').write_text(json.dumps(dict(state='CORRECTED_SOLAR_FORECAST_COT_BANKS_TEST_CLOSED',source_roles=source,test_payloads_read=0,buffer_payloads_read=0),indent=2))
 # Required diagnostics must finish before any downstream GHI-head training.
 cot_eval=BASE/'forecast_COT_eval_attempt12'
 run([sys.executable,CODE/'evaluate_forecast_cot.py','--cot-bank',cbank/'validation.npy','--simvp-keys',BASE/'S_banks_attempt12'/'full'/'val'/'keys.csv','--r-validation',BASE/'R_runs_attempt12'/'full'/'val','--output',cot_eval,'--bootstrap-draws',2000])
 eval_meta=json.loads((cot_eval/'forecast_cot_metrics.json').read_text());assert eval_meta['state']=='VALIDATION_ONLY_FORECAST_COT_DIAGNOSTIC_CPP_RETRIEVAL_REFERENCE' and eval_meta['test_payloads_read']==eval_meta['buffer_payloads_read']==0
 rows=records(OLD/'pack/rows.csv');centers={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)};ag=[];gg=[];cc=[]
 for r in rows:
  pred,geo,cot,keys=saved[r['source_role']];i=int(r['source_sequence_index']);l=int(r['lead_minutes'])//15-1;k=keys[i];assert k['init_time_bjt']==r['init_time_bjt'] and k['target_times_bjt'].split('|')[l]==r['target_time_bjt'];y,x=centers[r['station']];sl=(slice(y-1,y+2),slice(x-1,x+2));ag.append(pred[i,l][:,sl[0],sl[1]]);gg.append(geo[i,l][:2,sl[0],sl[1]]);cc.append(cot[i,l][:,sl[0],sl[1]])
 assert len(rows)==168790
 for n,x in [('agri_raw.npy',ag),('geometry.npy',gg),('cot.npy',cc)]:np.save(pack/n,np.asarray(x,np.float32))
 import shutil
 for n in ('rows.csv','ghi_wm2.npy','clear_sky_wm2.npy'):shutil.copyfile(OLD/'pack'/n,pack/n);assert sha(OLD/'pack'/n)==sha(pack/n)
 oof_report=BASE/'OOF_TRAIN_BANK_REPORT_attempt12.json'
 run([sys.executable,CODE/'audit_oof_bank_lineage.py','--base',BASE,'--old-pack',OLD/'pack','--output',oof_report])
 oof=json.loads(oof_report.read_text());assert oof['state']=='PASS_OOF_TRAIN_BANK_LINEAGE_AND_GHI_ROW_BINDING'
 report=json.loads((OLD/'pack/pack_report.json').read_text());report.update(solar_geometry_corrected=True,corrected_S_and_R_retrained_from_random=True,cot_values_reused_without_change=False,source_correctutc_Rs=source,source_time_only_pack_sha256=sha(OLD/'pack/pack_report.json'),source_corrected_data_ready_sha256=sha(DATA/'DATA_READY.json'),forecast_cot_evaluation_path=str(cot_eval/'forecast_cot_metrics.json'),forecast_cot_evaluation_sha256=sha(cot_eval/'forecast_cot_metrics.json'),oof_train_bank_report_path=str(oof_report),oof_train_bank_report_sha256=sha(oof_report),future_real_agri_used_as_input=False,future_cpp_used_as_input=False,known_solar_limitation=None,arrays_sha256={p.name:sha(p) for p in pack.glob('*.npy')});(pack/'pack_report.json').write_text(json.dumps(report,indent=2));outputs=BASE/'GHI_runs_attempt12';outputs.mkdir(exist_ok=False)
 for seed in (42,43,44):run([sys.executable,CODE/'train_paired_forecast.py','--pack',pack,'--output',outputs/f'paired_seed{seed}_attempt1','--input-mode','forecast','--arms','FA','FAC','--epochs',30,'--batch-size',1536,'--seed',seed])
 run([sys.executable,CODE/'score_predictions.py','--experiment',outputs,'--pack',pack,'--output',BASE/'scored_attempt12']);(BASE/'ALL_PAIRED_COMPLETE_attempt12.json').write_text(json.dumps(dict(state='CORRECTED_SOLAR_PAIRED_GHI_COMPLETE',test_payloads_read=0,buffer_payloads_read=0)));print('ALL_CORRECTED_SOLAR_GHI_PAIRED_COMPLETE',flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['profile','R','S','GHI']);p.add_argument('--role',choices=ROLES,default='full');a=p.parse_args()
 {'profile':profile,'R':R,'S':lambda:S(a.role),'GHI':GHI}[a.mode]()
if __name__=='__main__':main()
