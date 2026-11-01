#!/usr/bin/env python3
"""Preflight, shard five frozen R fold inferences over four allocated GPUs, and audit."""
from __future__ import annotations
import argparse, concurrent.futures, csv, hashlib, json, os, queue, subprocess, sys, time
from pathlib import Path

STAGED_SHA='8b1f95c2a2cf1d66cabd11539b975fe3dccd486146bf3bec17fb18e537f72183'
FOLD_SHA='c86bd6713d5afea1d6864cc4ef6048a93b596051ea4358e7954f39c32b054122'
MODEL_SHA='8af2d0e101b492bf2096223fe788b7f3a8a9b5ece2544db45903b95397620e05'
CKPT_SHA=[
 '4ffb7c841b8f278fcdcd6143062d5af8e9ce515d7139044c02946229795e5a68',
 '8525ab7e4a0a1a57db38221f53545f316af6bea84990e4d944e661c31208d558',
 '8029465a066bd5651272a2297102605fa5a36a6857248507dc74563fd212f2f3',
 'b4f8c035511e4f12d6867adb24b93250ba94be68d0646ea669587984390bf5aa',
 '91a98712546aba04f45685467c133aa74e728fba7d5b45311f307f2df82a6f52',
]
CHANNELS=["C01","C02","C03","C04","C05","C06","C09","C10","C11","C12","C13","C14","C15","cosSOZ","cosRAA","day_mask"]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def json_read(p):return json.loads(Path(p).read_text())

def main():
 p=argparse.ArgumentParser()
 for name in ('staged-rows','fold-map','agri-root','fold-pack-root','fold-run-root','model-source','full-bank','script-dir','output','logs'):
  p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--gpu-ids',required=True);p.add_argument('--batch',type=int,default=256);p.add_argument('--python',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 start=time.time();gpus=[int(x) for x in a.gpu_ids.split(',') if x.strip()]
 if len(gpus)!=4 or len(set(gpus))!=4:raise ValueError(f'exactly four unique allocated GPUs required, got {gpus}')
 if a.output.exists():raise FileExistsError(f'refusing existing output {a.output}')
 if sha(a.staged_rows)!=STAGED_SHA or sha(a.fold_map)!=FOLD_SHA:raise ValueError('frozen staged-index/fold-map hash mismatch')
 if sha(a.model_source)!=MODEL_SHA:raise ValueError('frozen R model source hash mismatch')
 staged=read(a.staged_rows);foldrows=read(a.fold_map)
 if len(staged)!=22061 or sum(r['split']=='train' for r in staged)!=20206 or sum(r['split']=='val' for r in staged)!=1855:raise ValueError('staged train/val key counts changed')
 if any(r['split'] not in ('train','val') for r in staged):raise ValueError('unexpected split key in staged manifest')
 staged_keys={(r['time_bjt'],r['split']) for r in staged}
 if len(staged_keys)!=len(staged):raise ValueError('duplicate staged key')
 assign={r['date_BJT'].replace('-',''):int(r['heldout_fold']) for r in foldrows}
 if len(assign)!=250 or set(assign.values())!=set(range(5)):raise ValueError('fold map structure changed')
 train_dates={r['time_bjt'][:8] for r in staged if r['split']=='train'}
 val_dates={r['time_bjt'][:8] for r in staged if r['split']=='val'}
 if train_dates-set(assign) or val_dates&set(assign):raise ValueError('train/validation date-to-fold boundary changed')
 counts=[sum(1 for r in staged if r['split']=='train' and assign.get(r['time_bjt'][:8])==f) for f in range(5)]
 if counts!=[4174,4052,4008,4031,3941]:raise ValueError(f'fold row counts changed: {counts}')
 report=json_read(a.full_bank/'BANK_REPORT.json')
 bankrows=read(a.full_bank/'rows.csv')
 if report.get('test_payloads_read')!=0 or report.get('train_predictions_are_oof') is not False:raise ValueError('full-R comparison bank receipt mismatch')
 if {(r['time_bjt'],r['split']) for r in bankrows}!=staged_keys or len(bankrows)!=len(staged):raise ValueError('full-R bank does not match staged key support')
 for f in range(5):
  pack=a.fold_pack_root/f'fold_{f}';run=a.fold_run_root/f'fold_{f}'
  status=json_read(run/'status.json');meta=json_read(run/'run_metadata.json');ps=json_read(pack/'pack_status.json');norm=json_read(pack/'norm.json'); hashes=json_read(pack/'SHA256.json')
  if status.get('state')!='COMPLETE_VALIDATION_ONLY_TEST_CLOSED' or status.get('test_payloads_read')!=0:raise ValueError(f'fold {f} run receipt failed')
  if meta.get('model_source_sha256')!=MODEL_SHA or meta.get('normalization',{}).get('fit_split')!='train only' or meta.get('test_used') is not False:raise ValueError(f'fold {f} provenance mismatch')
  if ps.get('state')!='HUBEI_R_TRAIN_VAL_CORRECT_UTC_PACK_BUILT_TEST_CLOSED' or ps.get('test_payloads_read')!=0 or ps.get('buffer_payloads_read')!=0 or ps.get('heldout_train_date_overlap')!=0:raise ValueError(f'fold {f} input pack receipt failed')
  if norm.get('input_channels')!=CHANNELS or norm.get('fit_split')!='train only' or norm.get('test_used') is not False:raise ValueError(f'fold {f} normalization contract failed')
  if hashes.get('norm.json')!=sha(pack/'norm.json'):raise ValueError(f'fold {f} norm hash failed')
  if sha(run/'best.pt')!=CKPT_SHA[f]:raise ValueError(f'fold {f} checkpoint hash mismatch')
 if a.agri_root.resolve().is_dir() is False:raise FileNotFoundError(a.agri_root)
 if len(gpus)!=4:raise RuntimeError('four-GPU allocation check failed')
 if a.preflight_only:
  print(json.dumps({'state':'PASS_OOF_HISTORY_INFERENCE_PREFLIGHT','fold_counts':counts,'staged_rows':len(staged),
   'train_rows':20206,'val_rows':1855,'gpu_allocation_checked':False,
   'test_payloads_read':0,'buffer_payloads_read':0,'cpp_target_or_quality_payloads_read':0,'ghi_labels_read':0,
   'fold_checkpoint_sha256':CKPT_SHA,'model_source_sha256':MODEL_SHA},indent=2),flush=True)
  return
 a.output.mkdir(parents=True,exist_ok=False);a.logs.mkdir(parents=True,exist_ok=True)
 worker=a.script_dir/'infer_oof_history_fold.py';auditor=a.script_dir/'audit_oof_history_bank.py'
 for s in (worker,auditor):
  if not s.is_file():raise FileNotFoundError(s)
  subprocess.run([a.python,'-m','py_compile',str(s)],check=True)
 env_base=os.environ.copy();env_base.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'})
 def run_fold(f,gpu):
  env=env_base.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
  out=a.output/f'fold_{f}'
  log=a.logs/f'fold_{f}.log'
  cmd=[a.python,str(worker),'--fold',str(f),'--staged-rows',str(a.staged_rows),'--fold-map',str(a.fold_map),
   '--agri-root',str(a.agri_root),'--fold-pack',str(a.fold_pack_root/f'fold_{f}'),'--checkpoint',str(a.fold_run_root/f'fold_{f}'/'best.pt'),
   '--model-source',str(a.model_source),'--output',str(out),'--batch',str(a.batch)]
  with log.open('w') as fp:subprocess.run(cmd,env=env,stdout=fp,stderr=subprocess.STDOUT,check=True)
  return f,gpu,log.stat().st_size
 jobs=queue.Queue()
 for f in range(5):jobs.put(f)
 worker_map={}
 def gpu_worker(gpu):
  while True:
   try:f=jobs.get_nowait()
   except queue.Empty:return
   result=run_fold(f,gpu);worker_map[str(result[0])]={'allocated_gpu':result[1],'log_bytes':result[2]}
   print(f'FOLD_COMPLETE fold={result[0]} gpu={result[1]}',flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(gpu_worker,gpu) for gpu in gpus]
  for fut in futures:fut.result()
 subprocess.run([a.python,str(auditor),'--staged-rows',str(a.staged_rows),'--fold-map',str(a.fold_map),
  '--full-bank',str(a.full_bank),'--oof-root',str(a.output),'--out',str(a.output/'evidence'/'summary.json')],check=True)
 receipt={'state':'COMPLETE_OOF_HISTORY_INPUT_DISTRIBUTION_DIAGNOSTIC','pbs_job_id':os.environ.get('PBS_JOBID'),
  'node':os.uname().nodename,'gpu_ids':gpus,'gpu_workers':worker_map,'parallelism':4,'batch_per_worker':a.batch,
  'train_rows':20206,'val_rows_inferred':0,'test_payloads_read':0,'buffer_payloads_read':0,'cpp_target_or_quality_payloads_read':0,'ghi_labels_read':0,
  'elapsed_seconds':time.time()-start,'staged_rows_sha256':sha(a.staged_rows),'fold_map_sha256':sha(a.fold_map),
  'model_source_sha256':sha(a.model_source),'comparison_bank_sha256':sha(a.full_bank/'cot_pred_trainval_full.npy')}
 rp=a.output/'run_receipt.json';rp.write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
