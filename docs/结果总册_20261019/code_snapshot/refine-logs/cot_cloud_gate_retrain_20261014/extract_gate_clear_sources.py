"""Bounded validation-only source reads for timing and cloud-shadow diagnostics."""
import argparse,csv,json,importlib.util
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['hpc','fd'],required=True);ap.add_argument('--cases',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--grid',type=Path);ap.add_argument('--workbook-extractor',type=Path)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);cases=json.loads(a.cases.read_text())
    if a.mode=='hpc':
        root=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/cot_value_first_20261013')
        src=root/'data/ghi_kt_trainval_v1/ghi_kt_train_val.csv';rows=list(csv.DictReader(src.open()));rows=[r for r in rows if r['new_split']=='val']
        with (a.output/'val_labels_5min.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        allowed={(r['station'],r['target_time_utc']) for r in rows}
        data=Path('/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed');grid=dict(np.load(data/'grid_static.npz'));coords={'sili':(163,148),'zhujia':(73,150)}
        traces=[];arrays={};errors=[]
        for ci,c in enumerate(cases):
            st=c['station'];t=datetime.fromisoformat(c['utc']);r,col=coords[st]
            for dt in (-30,-15,0,15,30):
                tt=t+timedelta(minutes=dt);stamp=tt.strftime('%Y%m%d%H%M');key=(st,tt.strftime('%Y-%m-%d %H:%M:%S'))
                if key not in allowed:continue
                try:
                    af=list((data/'aux_256'/((tt+timedelta(hours=8)).strftime('%Y%m'))).glob((tt+timedelta(hours=8)).strftime('%Y%m%d%H%M')+'_*_aux.npz'))
                    assert len(af)==1
                    zz=dict(np.load(af[0]));x=np.load(data/'data'/tt.strftime('%Y/%Y%m/%Y%m%d')/(stamp+'.npy'),mmap_mode='r')
                    sl=np.s_[r-8:r+8,col-8:col+8];tag=f'c{ci}_m{dt}'
                    arrays[tag+'_channels']=np.asarray(x[[0,1,2,12,15,16,17,18,19],r-8:r+8,col-8:col+8])
                    arrays[tag+'_lat']=grid['lat'][sl];arrays[tag+'_lon']=grid['lon'][sl]
                    rec=dict(case=ci,station=st,utc=key[1],offset_min=dt,aux_fields=list(zz),array_tag=tag,
                        hdf_row=int(grid['row_index'][r,col]),hdf_col=int(grid['col_index'][r,col]),
                        solar_source_degrees=[float(x[k,r,col]) for k in (15,16,17,18)])
                    for k,v in zz.items():
                        if 'scan' in k.lower() or 'obs' in k.lower():
                            vv=np.asarray(v);rec[k]=float(vv[r,col]) if vv.shape==(256,256) else (vv.tolist() if vv.size<10 else str(vv.shape))
                    traces.append(rec)
                except Exception as e:errors.append(dict(case=ci,utc=key[1],error=str(e)))
        np.savez_compressed(a.output/'source_agri_geometry.npz',**arrays)
        (a.output/'source_geometry_metadata.json').write_text(json.dumps(dict(records=traces,errors=errors,label_rows=len(rows),test_payloads_read=0),indent=2))
        np.savez_compressed(a.output/'grid_static.npz',**grid)
        print(json.dumps(dict(label_rows=len(rows),records=len(traces),errors=errors)))
    else:
        import h5py
        grid=dict(np.load(a.grid));coords={'sili':(163,148),'zhujia':(73,150)};records=[];arrays={}
        for ci,c in enumerate(cases):
            st=c['station'];t=datetime.fromisoformat(c['utc']);r,col=coords[st];sl=np.s_[r-8:r+8,col-8:col+8]
            rec=dict(case=ci,station=st,utc=c['utc'],errors=[])
            hp=Path('/home/Data_Pool/data/FY/FY4B/AGRI/4KM')/t.strftime('%Y/%Y%m%d')
            matches=list(hp.glob('*_'+t.strftime('%Y%m%d%H%M')+'00_*4000M*.HDF'))
            try:
                assert len(matches)==1
                with h5py.File(matches[0]) as h:
                    rr=int(grid['row_index'][r,col]);cc=int(grid['col_index'][r,col]);v=h['NOMObs/NOMObsTime'];raw=v[rr].tolist()
                    mid=[datetime.strptime(str(int(q)),'%Y%m%d%H%M%S%f') for q in raw];tm=mid[0]+(mid[1]-mid[0])/2
                    rec.update(hdf=str(matches[0]),true_HDF_rc=[rr,cc],NOMObsTime_raw=raw,scan_midpoint_utc=tm.isoformat(),scan_offset_seconds=(tm-t).total_seconds())
            except Exception as e:rec['errors'].append('HDF: '+str(e))
            cp=Path('/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP')/t.strftime('%Y/%Y%m%d')/('FY4B_AGRI_'+t.strftime('%Y%m%d%H%M')+'00.nc')
            try:
                with h5py.File(cp) as h:
                    rec['CPP_path']=str(cp);rec['CPP_variables']={k:list(v.shape) for k,v in h.items() if isinstance(v,h5py.Dataset)}
                    rec['CPP_global_attributes']={k:str(v) for k,v in h.attrs.items() if any(q in k.lower() for q in ['time','date','source','history'])}
                    lat=np.asarray(h['LAT'][:,0] if h['LAT'].ndim==2 else h['LAT'][:]);lon=np.asarray(h['LON'][0,:] if h['LON'].ndim==2 else h['LON'][:])
                    la=grid['lat'][sl];lo=grid['lon'][sl];rr=np.abs(lat[:,None]-la[:,8]).argmin(0);cc=np.abs(lon[:,None]-lo[8]).argmin(0)
                    rec['CPP_max_grid_error']=float(max(np.abs(la-lat[rr,None]).max(),np.abs(lo-lon[None,cc]).max()))
                    rec['CPP_variable_attributes']={}
                    for k,v in h.items():
                        if isinstance(v,h5py.Dataset) and v.shape==(len(lat),len(lon)) and k not in ['LAT','LON']:
                            cut=np.asarray(v[int(rr.min()):int(rr.max())+1,int(cc.min()):int(cc.max())+1]);cut=cut[np.ix_(rr-rr.min(),cc-cc.min())]
                            arrays[f'c{ci}_{k}']=cut;rec['CPP_variable_attributes'][k]={ak:str(av) for ak,av in v.attrs.items() if ak in ['units','long_name','_FillValue','scale_factor','add_offset']}
            except Exception as e:rec['errors'].append('CPP: '+str(e))
            records.append(rec)
        np.savez_compressed(a.output/'raw_cpp_case_fields.npz',**arrays)
        (a.output/'raw_time_cpp_audit.json').write_text(json.dumps(dict(records=records,test_payloads_read=0),indent=2))
        if a.workbook_extractor:
            from collections import defaultdict
            spec=importlib.util.spec_from_file_location('workbook',a.workbook_extractor);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            original={}
            for st in ('sili','zhujia'):
                needed=defaultdict(dict)
                for c in cases:
                    if c['station']!=st:continue
                    t=datetime.fromisoformat(c['utc'])+timedelta(hours=8)
                    for k in range(-90,96,5):
                        tt=t+timedelta(minutes=k);assert tt.date()==t.date()
                        needed[tt.date()][tt.strftime('%H%M')]=tt
                path=Path('/home/Data_Pool_3/chenyi/GHI/data/hunan/raw/stations')/('四里光伏.xlsx' if st=='sili' else '竺家光伏.xlsx')
                vals,meta=mod.extract_requested(path,needed)
                original[st]=dict(values={k.isoformat(sep=' '):v for k,v in vals.items()},source=str(path),metadata=meta)
            (a.output/'original_workbook_5min.json').write_text(json.dumps(original,ensure_ascii=False,indent=2))
        print(json.dumps(records,indent=2))

if __name__=='__main__':main()
