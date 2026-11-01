"""Frozen old heads on new stored validation inputs; CPU-only bounded replay."""
import csv,json,sys,time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
B=Path(__file__).resolve().parent
P=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
H=P/'experiments/cloud_probability_input_20261020'
O=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanDyDiff_20261028/hunan_full_validation_20261030/evaluation_attempt1')
sys.path.insert(0,str(H))
from train_four_arm import FourArmIntervalHead,EXPECTED_COT_MEAN,EXPECTED_COT_STD
def key(r):return(r['station'],str(datetime.fromisoformat(r['init_time_utc'])),str(datetime.fromisoformat(r['target_time_utc'])))
def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 torch.backends.mkldnn.enabled=True
 old=np.load(O/'ghi_predictions.npz',allow_pickle=False);ors=list(csv.DictReader((O/'ghi_rows.csv').open()));oi={key(r):i for i,r in enumerate(ors)}
 nr=list(csv.DictReader((B/'pack/val/keys.csv').open()));ni={key(r):i for i,r in enumerate(nr)}
 norm=json.loads((P/'experiments/cot_interval_history_20261015_v1/data_attempt2/pack_manifest.json').read_text())
 ag=np.load(B/'pack/val/agri_raw.npy',mmap_mode='r');geo=np.load(B/'pack/val/geometry.npy',mmap_mode='r');cot=np.load(B/'pack/val/cot_log1p.npy',mmap_mode='r');clear=np.load(B/'pack/val/clear.npy',mmap_mode='r')
 result=[];start=time.monotonic()
 for st in ('sili','zhujia'):
  ks=sorted(k for k in oi if k[0]==st and k in ni);ix=np.array([oi[k] for k in ks]);jx=np.array([ni[k] for k in ks])
  x=np.concatenate([(ag[jx]-np.array(norm['AGRI_norm_mean'],np.float32)[None,None,:,None,None])/np.array(norm['AGRI_norm_std'],np.float32)[None,None,:,None,None],geo[jx],(cot[jx]-EXPECTED_COT_MEAN)/EXPECTED_COT_STD,np.zeros((len(jx),3,1,3,3),np.float32)],axis=2).astype(np.float32)
  assert np.max(abs(clear[jx]-old['clear'][ix]))<.001
  for seed in (42,43,44):
   for arm in ('M00','M10'):
    cp=H/'results/four_arm_paired_v1_attempt1'/f'seed{seed}'/st/arm/'best.pt';payload=torch.load(cp,map_location='cpu',weights_only=False);m=FourArmIntervalHead().eval();m.load_state_dict(payload['model']);xx=x.copy()
    if arm=='M00':xx[:,:,15]=0
    pred=[]
    with torch.inference_mode():
     for lo in range(0,len(xx),1792):pred.append(m(torch.from_numpy(xx[lo:lo+1792])).numpy()*old['clear'][ix[lo:lo+1792]])
    pred=np.concatenate(pred).astype(float);ref=old[f'{st}_seed{seed}_{arm}_dydiff'][ix];y=old['truth'][ix]
    rec={'station':st,'seed':seed,'arm':arm,'rows':len(ix),'old_original_rmse':float(np.sqrt(np.mean((ref-y)**2))),'old_head_new_inputs_rmse':float(np.sqrt(np.mean((pred-y)**2))),'prediction_change_mae':float(abs(pred-ref).mean()),'prediction_change_p99':float(np.quantile(abs(pred-ref),.99)),'prediction_change_max':float(abs(pred-ref).max())};result.append(rec);print(json.dumps(rec),flush=True)
 source=json.loads((B/'common_sample_diagnosis/TRAINING_METADATA.json').read_text());setup=source['old_setup']
 trains=list(csv.DictReader((B/'pack/train/keys.csv').open()));summary={'old_train_observed_target_rows':setup['train'],'new_train_forecast_instances':len(trains),'new_train_unique_station_target_times':len({(r['station'],r['target_time_utc']) for r in trains}),'old_parameter_count':setup['parameters_each_arm'],'old_batch':setup['batch'],'old_epochs':setup['epochs'],'old_selection':setup['selection'],'old_cot_scaler':[EXPECTED_COT_MEAN,EXPECTED_COT_STD],'new_cot_scaler':[source['new_pack']['COT_log1p_mean'],source['new_pack']['COT_log1p_std']]}
 (B/'common_sample_diagnosis/CROSS_INPUT_REPLAY.json').write_text(json.dumps({'runs':result,'training_summary':summary,'elapsed_seconds':time.monotonic()-start},indent=2))
 print('SUMMARY',json.dumps(summary))
if __name__=='__main__':main()
