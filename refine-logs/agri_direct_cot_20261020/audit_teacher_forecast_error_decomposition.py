"""Offline V reference audit. Observed future AGRI is never a deployable input."""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
import torch
from train_r_u import COTUNet
from solar_geometry_corrected import compute_solar_angles
from score_direct_cot_cpp_v1 import validation_month_blocks

ARMS=('cot_last','cot_simvp_r','cot_p_c','cot_p_xc')
STATIONS=('sili','zhujia')
SCALES={'patch16':(slice(0,16),slice(0,16)),'station3':(slice(7,10),slice(7,10)),'center1':(slice(8,9),slice(8,9))}
COS80=float(np.cos(np.deg2rad(80)))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvout(p,rs):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def met(p,y,m):
    a,b=p[m].astype(np.float64),y[m].astype(np.float64)
    if len(a)==0:return {'n':0,'rmse':None,'mae':None,'bias':None,'pred_sd':None,'ref_sd':None,'corr':None,'thick_pod':None,'thick_far':None,'thick_csi':None}
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite on shared support')
    e=a-b;tp=int(((a>=30)&(b>=30)).sum());fp=int(((a>=30)&(b<30)).sum());fn=int(((a<30)&(b>=30)).sum())
    sd0,sd1=float(a.std()),float(b.std())
    return {'n':len(a),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(np.abs(e))),'bias':float(e.mean()),'pred_sd':sd0,'ref_sd':sd1,'corr':float(np.corrcoef(a,b)[0,1]) if min(sd0,sd1)>0 else None,'thick_pod':tp/(tp+fn) if tp+fn else None,'thick_far':fp/(tp+fp) if tp+fp else None,'thick_csi':tp/(tp+fp+fn) if tp+fp+fn else None}
def decomp(f,r,y,m):
    a=(r[m].astype(np.float64)-y[m]);b=(f[m].astype(np.float64)-r[m]);e=a+b
    if not len(e):return None
    teacher=float(np.mean(a*a));forecast=float(np.mean(b*b));cross=float(2*np.mean(a*b));total=float(np.mean(e*e))
    if abs(teacher+forecast+cross-total)>1e-8*max(1,total):raise ValueError('SSE identity failed')
    return {'n':len(e),'teacher_mse':teacher,'forecast_vs_teacher_mse':forecast,'cross_term':cross,'total_mse':total,'identity_abs_error':abs(teacher+forecast+cross-total)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--source-pack',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    base=a.base;r_run=base/'data/r_u_frozen_model_20261021';cpp=base/'data/cpp_score_inputs_v1/cpp_pack';inf=base/'data/cpp_score_inputs_v1/inference';support=base/'data/cpp_score_inputs_v1/support';grid=base/'data/grid_static.npz'
    status,meta,norm=js(r_run/'status.json'),js(r_run/'run_metadata.json'),js(r_run/'normalization.json');im=js(inf/'inference_manifest.json');ca=js(cpp/'pack_audit.json');pa=js(cpp/'pack_status.json');ua=js(base/'data/r_u_pack_v2/pack_audit.json')
    if status['state']!='COMPLETE_R_U_FIXED_EPOCH_50' or status['last_epoch']!=50:raise ValueError('R training receipt')
    for file,key in [('epoch_050.pt','checkpoint_sha256'),('normalization.json','normalization_sha256'),('run_metadata.json','run_metadata_sha256')]:
        if sha(r_run/file)!=status[key]:raise ValueError(f'R source SHA: {file}')
    if sha(r_run/'epoch_050.pt')!=im['contract']['r_checkpoint_sha256'] or sha(r_run/'normalization.json')!=im['contract']['r_norm_sha256']:raise ValueError('R differs from causal inference R')
    if sha(Path(__file__).with_name('train_r_u.py'))!=meta['trainer_source_sha256']:raise ValueError('R architecture source changed')
    if sha(Path(__file__).with_name('solar_geometry_corrected.py'))!=im['contract']['geometry_source_sha256']:raise ValueError('geometry source changed')
    if sha(grid)!=pa['grid_static_sha256'] or sha(grid)!=ua['grid_static_sha256']:raise ValueError('grid SHA')
    if meta['h_rows_used'] or meta['outer_validation_rows_used'] or status['test_payloads_read'] or status['buffer_payloads_read']:raise ValueError('R train isolation')
    for key in ('future_observed_agri_read','cpp_read','ghi_labels_read','test_payload_opened','buffer_payload_opened'):
        if im[key] is not False:raise ValueError('causal forecast isolation')
    for name in ('x_raw','cot','mask'):
        p=(a.source_pack if name=='x_raw' else cpp)/(name+'.npy')
        if sha(p)!=ca['arrays'][name]['sha256']:raise ValueError('CPP source '+name)
    if sha(a.source_pack/'rows.csv')!=sha(cpp/'rows.csv'):raise ValueError('raw/reference row mismatch')
    selected=[r for r in rows(cpp/'rows.csv') if r['split']=='val' and r['state']=='SUPERVISED']
    if len(selected)!=1398:raise ValueError('unexpected fixed V reference count')
    center={r['station']:tuple(map(int,r['station_grid_rc'].split(','))) for r in rows(base/'data/r_u_pack_v2/rows_u_supervised.csv')}
    gs=np.load(grid,allow_pickle=False);xsrc=np.load(a.source_pack/'x_raw.npy',mmap_mode='r',allow_pickle=False);ysrc=np.load(cpp/'cot.npy',mmap_mode='r',allow_pickle=False);msrc=np.load(cpp/'mask.npy',mmap_mode='r',allow_pickle=False)
    indices=np.asarray([int(r['array_index']) for r in selected]);x=np.asarray(xsrc[indices],np.float32).copy();truth=np.asarray(ysrc[indices,0],np.float32);old=np.asarray(msrc[indices,0],bool);geom=np.empty((len(selected),3,16,16),np.float32)
    for i,r in enumerate(selected):
        t=datetime.strptime(r['target_time_utc'],'%Y-%m-%d %H:%M:%S');rr,cc=center[r['station']];sl=np.s_[rr-8:rr+8,cc-8:cc+8]
        if Path(r['agri_path']).stem!=t.strftime('%Y%m%d%H%M') or r['bjt']!=(t+timedelta(hours=8)).strftime('%Y%m%d%H%M'):raise ValueError('UTC/BJT')
        soz,soa=compute_solar_angles(gs['lat'][sl],gs['lon'][sl],t);soa=np.where(soz>=90,0,soa)
        geom[i]=np.stack((np.cos(np.deg2rad(soz)),np.cos(np.deg2rad(soa-gs['saa'][sl])),(soz<85).astype(np.float32)))
    common=old&(geom[:,0]>COS80)&(geom[:,2]>.5);x[:,13:]=geom
    raw_c03=x[:,2].copy();x=(x-np.asarray(norm['mean'],np.float32)[None,:,None,None])/np.asarray(norm['std'],np.float32)[None,:,None,None];x[~np.isfinite(x)]=0
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(4);model=COTUNet().to(device);ck=torch.load(r_run/'epoch_050.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True);model.eval();pred=[];pred16=[]
    with torch.inference_mode():
        for lo in range(0,len(x),128):
            tin=torch.from_numpy(x[lo:lo+128]).to(device);z=model(tin).float();pred.append(torch.expm1(z).cpu().numpy())
            if device.type=='cuda':
                with torch.autocast('cuda',dtype=torch.float16):z16=model(tin).float()
                pred16.append(torch.expm1(z16).cpu().numpy())
    observed=np.concatenate(pred)[:,0];fp16=np.concatenate(pred16)[:,0] if pred16 else observed.copy()
    if not np.isfinite(observed).all() or (observed<0).any():raise ValueError('reference retrieval output')
    refidx={(r['station'],r['target_time_utc']):i for i,r in enumerate(selected)}
    if len(refidx)!=len(selected):raise ValueError('duplicate reference')
    vindex=rows(inf/'V/index.csv');seqmap={r['seq_id']:r for r in rows(support/'v_sequences.csv')};bydate=validation_month_blocks(base/'data/cpp_score_inputs_v1/day_assignment.csv');items=[]
    forecasts={}
    for arm in ARMS:
        p=inf/'V'/(arm+'.npy')
        if sha(p)!=im['artifact_sha256']['V/'+arm+'.npy']:raise ValueError('forecast source '+arm)
        forecasts[arm]=np.load(p,mmap_mode='r',allow_pickle=False)
    for si,row in enumerate(vindex):
        if int(row['sequence_index_within_split'])!=si:raise ValueError('sequence index')
        paths=seqmap[row['seq_id']]['data_relpaths'].split('|');init=datetime.strptime(Path(paths[7]).stem,'%Y%m%d%H%M')
        for st_i,st in enumerate(STATIONS):
            for li in range(16):
                t=datetime.strptime(Path(paths[8+li]).stem,'%Y%m%d%H%M');key=(st,t.strftime('%Y-%m-%d %H:%M:%S'))
                if key not in refidx:continue
                if t-init!=timedelta(minutes=15*(li+1)):raise ValueError('lead')
                items.append({'sequence_index':si,'station_index':st_i,'station':st,'lead_index':li,'lead_minutes':15*(li+1),'init_time_utc':init.isoformat(' '),'target_time_utc':key[1],'block_id':bydate[(t+timedelta(hours=8)).date().isoformat()],'reference_index':refidx[key]})
    if len(items)!=20500:raise ValueError('original CPP forecast coverage changed')
    ri=np.asarray([r['reference_index'] for r in items]);sq=np.asarray([r['sequence_index'] for r in items]);st=np.asarray([r['station_index'] for r in items]);li=np.asarray([r['lead_index'] for r in items]);ys=truth[ri];ro=observed[ri];masks={'archived_cpp':old[ri],'corrected_common':common[ri]};fs={k:np.asarray(v[sq,st,li,0],np.float32) for k,v in forecasts.items()};fs['observed_agri_r_u']=ro
    if not all(np.isfinite(p).all() for p in fs.values()):raise ValueError('nonfinite forecast')
    a.output.mkdir(parents=True);np.save(a.output/'observed_agri_r_u_cot.npy',observed[:,None]);np.save(a.output/'observed_agri_r_u_cot_fp16.npy',fp16[:,None]);np.save(a.output/'corrected_common_reference_mask.npy',common[:,None]);csvout(a.output/'reference_rows.csv',[dict(r,reference_index=i) for i,r in enumerate(selected)]);csvout(a.output/'matched_forecast_keys.csv',items)
    metricrows=[];decomprows=[];leadvalues=np.asarray([r['lead_minutes'] for r in items])
    for maskname,mask in masks.items():
        for scale,(rs,cs) in SCALES.items():
            mm=mask[:,rs,cs];yy=ys[:,rs,cs];rr=ro[:,rs,cs]
            for station in ('both',)+STATIONS:
                sm=np.ones(len(items),bool) if station=='both' else st==STATIONS.index(station)
                for lead in (0,)+tuple(range(15,241,15)):
                    use=sm if lead==0 else sm&(leadvalues==lead)
                    for arm,pp in fs.items():
                        p=pp[:,rs,cs][use];y=yy[use];r=rr[use];m=mm[use]
                        metricrows.append({'support':maskname,'scale':scale,'station':station,'lead_minutes':lead,'arm':arm,'reference':'CPP',**met(p,y,m)})
                        if arm!='observed_agri_r_u':
                            metricrows.append({'support':maskname,'scale':scale,'station':station,'lead_minutes':lead,'arm':arm,'reference':'observed_AGRI_R_U',**met(p,r,m)})
                            dc=decomp(p,r,y,m)
                            if dc:decomprows.append({'support':maskname,'scale':scale,'station':station,'lead_minutes':lead,'arm':arm,**dc})
    csvout(a.output/'metrics_by_support_scale_station_lead.csv',metricrows);csvout(a.output/'algebraic_mse_decomposition.csv',decomprows)
    blocks=sorted(set(bydate.values()));blockids=np.asarray([r['block_id'] for r in items]);draws=np.random.default_rng(20261020).multinomial(len(blocks),np.full(len(blocks),1/len(blocks)),size=5000);cis={}
    for arm in ARMS:
        stationpoints=[];stationdraws=[]
        for sti in range(2):
            ss0=[];ss1=[];nn=[]
            for b in blocks:
                use=(st==sti)&(blockids==b);mask=masks['corrected_common'][use,7:10,7:10];y=ys[use,7:10,7:10];p=fs[arm][use,7:10,7:10];r=ro[use,7:10,7:10]
                ss0.append(float(np.sum((p[mask].astype(np.float64)-y[mask])**2)));ss1.append(float(np.sum((r[mask].astype(np.float64)-y[mask])**2)));nn.append(int(mask.sum()))
            ss0,ss1,nn=map(np.asarray,(ss0,ss1,nn));den=draws@nn
            if (den<=0).any():raise ValueError('empty bootstrap support')
            stationpoints.append(float(np.sqrt(ss0.sum()/nn.sum())-np.sqrt(ss1.sum()/nn.sum())));stationdraws.append(np.sqrt((draws@ss0)/den)-np.sqrt((draws@ss1)/den))
        dd=np.mean(stationdraws,axis=0);cis[arm+'_minus_observed_r']={'delta_rmse_station_equal':float(np.mean(stationpoints)),'ci95':[float(np.quantile(dd,.025)),float(np.quantile(dd,.975))],'station_deltas':dict(zip(STATIONS,stationpoints))}
    cases=[]
    for sti,station in enumerate(STATIONS):
        eligible=np.flatnonzero((st==sti)&(masks['corrected_common'].sum((1,2))>=200));m=masks['corrected_common'][eligible];cnt=m.sum((1,2));fr=np.where(m,(fs['cot_p_xc'][eligible].astype(np.float64)-ro[eligible])**2,0).sum((1,2))/cnt;tr=np.where(m,(ro[eligible].astype(np.float64)-ys[eligible])**2,0).sum((1,2))/cnt
        for kind,errors in [('forecast_vs_teacher',fr),('teacher_vs_cpp',tr)]:
            maximum=float(errors.max());ties=eligible[np.flatnonzero(errors==maximum)];pick=min(ties,key=lambda j:items[j]['lead_minutes']);cases.append(dict(items[pick],selection=kind,selection_mse=maximum,item_index=int(pick)))
    csvout(a.output/'posthoc_case_selection.csv',cases)
    np.savez_compressed(a.output/'posthoc_case_maps.npz',c03=np.stack([raw_c03[c['reference_index']] for c in cases]),cpp=np.stack([ys[c['item_index']] for c in cases]),observed_r=np.stack([ro[c['item_index']] for c in cases]),p_xc=np.stack([fs['cot_p_xc'][c['item_index']] for c in cases]),simvp_r=np.stack([fs['cot_simvp_r'][c['item_index']] for c in cases]),last=np.stack([fs['cot_last'][c['item_index']] for c in cases]),mask=np.stack([masks['corrected_common'][c['item_index']] for c in cases]))
    source={'script_sha256':sha(__file__),'inference_manifest_sha256':sha(inf/'inference_manifest.json'),'r_checkpoint_sha256':sha(r_run/'epoch_050.pt'),'r_normalization_sha256':sha(r_run/'normalization.json'),'raw_agri_sha256':sha(a.source_pack/'x_raw.npy'),'cpp_audit_sha256':sha(cpp/'pack_audit.json'),'grid_sha256':sha(grid)}
    summary={'state':'COMPLETE_TEACHER_FORECAST_ERROR_DECOMPOSITION','scope':'seen V development-only reference diagnosis; no GHI improvement or independent confirmation','reference_rows':len(selected),'forecast_items':len(items),'unique_station_target_keys':len(set(ri.tolist())),'matched_blocks':len(set(blockids.tolist())),'frozen_blocks':len(blocks),'original_mask_pixels':int(old.sum()),'corrected_common_pixels':int(common.sum()),'fp16_reference_drift':met(fp16,observed,common),'fp16_max_abs_drift_on_common':float(np.max(np.abs(fp16[common]-observed[common]))),'bootstrap_station3_corrected_common':cis,'primary_metrics':[r for r in metricrows if r['support']=='corrected_common' and r['scale']=='station3' and r['station']=='both' and r['lead_minutes']==0],'primary_decomposition':[r for r in decomprows if r['support']=='corrected_common' and r['scale']=='station3' and r['station']=='both' and r['lead_minutes']==0],'source':source,'test_payload_opened':False,'buffer_payload_opened':False,'observed_future_agri_read_for_retrieval_reference':True,'observed_future_agri_used_by_causal_forecaster':False,'observed_future_agri_used_by_ghi_head':False,'ghi_training_or_inference_performed':False,'training_started':False,'reference_role':'observed AGRI -> frozen R_U, only as offline reference; CPP is retrieval reference, not independent truth'}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2,allow_nan=False),flush=True)
if __name__=='__main__':main()
