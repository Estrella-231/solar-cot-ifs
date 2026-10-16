"""CPU-only immutable checkpoint probes and exact real-cloud input extraction."""
import argparse,csv,json,hashlib,sys
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 exp=a.root/'experiments/r_geometry_downstream_ghi_20261016';run=exp/'results/paired_three_arm_v1';pack=a.root/'experiments/cot_interval_history_20261015_v1/data_attempt2'
 sys.path.insert(0,str(exp));from train_three_arm_ghi import IntervalHead
 sys.path.insert(0,str(a.root/'experiments/r_solar_geometry_corrected_20261015'));from solar_geometry_corrected import compute_solar_angles
 sel=json.loads(a.cases.read_text());assert sel['test_payloads_read']==0 and sha(run/'analysis/paired_validation_scoring.npz')==sel['source_predictions_sha256']
 rows=list(csv.DictReader((run/'validation_rows.csv').open()));assert len(rows)==4263 and {r['split'] for r in rows}=={'val'}
 z=np.load(run/'validation_inputs.npz');saved=np.load(run/'analysis/paired_validation_scoring.npz');assert np.array_equal(z['pack_index'],saved['pack_index'])
 setup=json.loads((run/'setup.json').read_text());norm=json.loads((pack/'pack_manifest.json').read_text());cm=setup['COT_pool_train_mean'];cs=setup['COT_pool_train_std'];torch.set_num_threads(2)
 x=np.where(z['agri_valid'],(z['agri_raw']-np.asarray(norm['AGRI_norm_mean'],np.float32)[None,None,:,None,None])/np.asarray(norm['AGRI_norm_std'],np.float32)[None,None,:,None,None],0).astype(np.float32)
 g=z['geometry'].copy();c=((z['cot_corrected_log1p']-cm)/cs).astype(np.float32);cl=np.array([float(r['clear']) for r in rows],np.float32);st=np.array([r['station'] for r in rows]);n=len(rows)
 grads=np.full((3,n,3,3,3),np.nan,np.float32);actual=np.full((3,n),np.nan);zero=np.full_like(actual,np.nan);repeat=np.full_like(actual,np.nan);receipts=[]
 for si,seed in enumerate((42,43,44)):
  for station in ('sili','zhujia'):
   ids=np.flatnonzero(st==station);folder=run/f'seed{seed}'/station/'AC_history_corrected_R';ck=folder/'best.pt';before=sha(ck);model=IntervalHead();model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['model']);model.eval().requires_grad_(False)
   X=torch.tensor(x[ids]);G=torch.tensor(g[ids]);C=torch.tensor(c[ids],requires_grad=True);CL=torch.tensor(cl[ids]);pr=model(X,G,C)*CL;grad=torch.autograd.grad(pr.sum(),C)[0].detach().numpy()[:,:,0]/cs;grads[si,ids]=grad;actual[si,ids]=pr.detach().numpy()
   with torch.inference_mode():
    zero[si,ids]=(model(X,G,torch.zeros_like(C))*CL).numpy();repeat[si,ids]=(model(X,G,C[:,-1:].repeat(1,3,1,1,1))*CL).numpy()
   expected=np.load(folder/'validation_predictions.npz')['cpu_prediction'];err=float(np.max(np.abs(actual[si,ids]-expected)));assert err<.001,(seed,station,err);assert sha(ck)==before
   receipts.append(dict(seed=seed,station=station,checkpoint_sha256=before,reload_max_abs_GHI=err,parameters_frozen=all(not q.requires_grad for q in model.parameters()),checkpoint_unchanged=True));print('CPU_PROBE',receipts[-1],flush=True)
 assert np.isfinite(grads).all() and np.isfinite(actual).all()
 np.savez_compressed(a.output/'frozen_probe_allval.npz',gradient_dGHI_dlog1pCOT=grads,actual=actual,zero_standardized_cot=zero,repeat_current_cot=repeat,pack_index=z['pack_index'])
 data=Path('/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed');gridpath=data/'grid_static.npz';grid=np.load(gridpath);meta=a.root/'experiments/monthly_R_station16_20261013_v1/interval_alignment_20261015/manifest_full_v1/scan_offsets.json';offsets=json.loads(meta.read_text());time_records=[]
 for i,r in enumerate(rows):
  ts=[r['earlier_time_utc'],r['previous_time_utc'],r['target_time_utc']];si=int(r['station_index']);off=[float(offsets[t][si]) for t in ts];time_records.append(dict(validation_index=i,station=r['station'],utc=r['target_time_utc'],earlier_scan_s_relative_target=-1800+off[0],previous_scan_s_relative_target=-900+off[1],current_scan_s_relative_target=off[2],three_frames_bracket_label=int(-1800+off[0]<=-900 and off[2]>=0)))
 with (a.output/'scan_timing.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(time_records[0]));w.writeheader();w.writerows(time_records)
 casearrays={};frames=[];indices=[]
 for case in sel['cases']:
  ci=case['case_index'];i=case['validation_index'];r=rows[i];assert (r['station'],r['target_time_utc'])==(case['station'],case['utc']);indices.append(i);rr,cc=int(r['row']),int(r['col']);sl=np.s_[rr-8:rr+8,cc-8:cc+8]
  casearrays[f'c{ci}_cot_old']=np.expm1(z['cot_old_log1p'][i,:,0]);casearrays[f'c{ci}_cot_corrected']=np.expm1(z['cot_corrected_log1p'][i,:,0]);casearrays[f'c{ci}_gradient']=grads[:,i]
  for j,key in enumerate(('earlier_time_utc','previous_time_utc','target_time_utc')):
   t=datetime.fromisoformat(r[key]);b=t+timedelta(hours=8);assert b.date()==(datetime.fromisoformat(r['target_time_utc'])+timedelta(hours=8)).date();path=data/'data'/t.strftime('%Y/%Y%m/%Y%m%d')/(t.strftime('%Y%m%d%H%M')+'.npy');raw=np.load(path,mmap_mode='r');auxfiles=sorted((data/'aux_256'/b.strftime('%Y%m')).glob(b.strftime('%Y%m%d%H%M')+'_*_aux.npz'));assert len(auxfiles)==1
   casearrays[f'c{ci}_t{j}_rgb']=np.asarray(raw[[2,1,0],sl[0],sl[1]]).copy();casearrays[f'c{ci}_t{j}_C13_BT']=np.asarray(raw[12,sl[0],sl[1]]).copy()
   channels=np.array([0,1,2,3,4,5,8,9,10,11,12,13,14]);patch=np.asarray(raw[channels,rr-1:rr+2,cc-1:cc+2]);valid=z['agri_valid'][i,j];err=float(np.max(np.abs(patch[valid]-z['agri_raw'][i,j][valid])));assert err<2e-5
   with np.load(auxfiles[0],allow_pickle=False) as aux:
    scan=float(aux['obs_time_offset_s'][rr,cc]);assert abs(scan-float(offsets[r[key]][int(r['station_index'])]))<.02
    casearrays[f'c{ci}_t{j}_scanoffset3']=np.asarray(aux['obs_time_offset_s'][rr-1:rr+2,cc-1:cc+2]).copy()
   lat=grid['lat'][rr-1:rr+2,cc-1:cc+2];lon=grid['lon'][rr-1:rr+2,cc-1:cc+2];saa=grid['saa'][rr-1:rr+2,cc-1:cc+2];zen,az=compute_solar_angles(lat,lon,t);geo=np.stack([np.cos(np.deg2rad(zen)),np.cos(np.deg2rad(az-saa))]);ge=float(np.max(np.abs(geo-z['geometry'][i,j])));assert ge<2e-5,ge
   act=t+timedelta(seconds=scan);za,aa=compute_solar_angles(lat,lon,act);geoact=np.stack([np.cos(np.deg2rad(za)),np.cos(np.deg2rad(aa-saa))]);casearrays[f'c{ci}_t{j}_geometry_nominal']=geo;casearrays[f'c{ci}_t{j}_geometry_actual_scan']=geoact
   fr=dict(case_index=ci,frame=j,station=r['station'],nominal_utc=r[key],actual_station_scan_utc=act.isoformat(sep=' '),scan_offset_s=scan,source_agri=str(path),source_aux=str(auxfiles[0]),HDF_row=int(grid['row_index'][rr,cc]),HDF_col=int(grid['col_index'][rr,cc]),station_lat=float(lat[1,1]),station_lon=float(lon[1,1]),AGRI_3x3_max_abs_error=err,nominal_geometry_max_abs_error=ge,scan_minus_nominal_zenith_deg=float(za[1,1]-zen[1,1]),scan_minus_nominal_cosSOZ=float(geoact[0,1,1]-geo[0,1,1]),scan_minus_nominal_cosRAA=float(geoact[1,1,1]-geo[1,1,1]));frames.append(fr)
  print('CASE_EXTRACTED',ci,r['station'],r['target_time_utc'],flush=True)
 np.savez_compressed(a.output/'real_cloud_cases_and_gradients.npz',**casearrays)
 ix=np.array(indices);np.savez_compressed(a.output/'selected_frozen_probes.npz',validation_index=ix,original=actual[:,ix],zero_standardized_cot=zero[:,ix],repeat_current_cot=repeat[:,ix],gradient_dGHI_dlog1pCOT=grads[:,ix])
 report=dict(state='PASS_FROZEN_CPU_AND_REAL_INPUT_AUDIT',validation_rows=n,case_count=len(indices),models=receipts,frames=frames,source_grid_sha256=sha(gridpath),source_scan_offsets_sha256=sha(meta),source_cases_sha256=sha(a.cases),probe_definition='dGHI/dlog1p COT at the exact input; model weights frozen; zero and repeated-COT inputs are OOD sensitivities only',allval_probes_sha256=sha(a.output/'frozen_probe_allval.npz'),cloud_payload_sha256=sha(a.output/'real_cloud_cases_and_gradients.npz'),test_payloads_read=0,qsub_used=False)
 (a.output/'extraction_receipt.json').write_text(json.dumps(report,indent=2)+'\n');print('COMPLETE_FIXED_MODEL_FORensics',flush=True)
if __name__=='__main__':main()
