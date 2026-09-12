"""Read-only coordinate binding for the already frozen 64-target diagnostic."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import numpy as np

SELECTION_SHA = '8a06a7f06cf3339cd8f067111d5397c173ba63868598c6904348362994b08318'
ABS_DEG = 1e-5

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def coords_hash(lat, lon):
    return hashlib.sha256(np.asarray(lat,dtype='<f4').tobytes()+np.asarray(lon,dtype='<f4').tobytes()).hexdigest()

def compare(a,b):
    assert a.shape==b.shape==(16,16)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    error=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(max_abs_degrees=float(error.max()), exact_equal=bool(np.array_equal(a,b)),
                within_fixed_tolerance=bool(error.max()<=ABS_DEG))

def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--grid-reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists(),'refuse overwrite'
    inv=json.loads(a.inventory.read_text());ref=json.loads(a.grid_reference.read_text())
    assert inv['selection_sha256']==SELECTION_SHA and len(inv['selection'])==64
    assert hashlib.sha256(json.dumps(dict(rule=inv['selection_rule'],targets=inv['selection']),
        sort_keys=True,separators=(',',':')).encode()).hexdigest()==SELECTION_SHA
    assert ref['grid_sha256']==inv['forecast']['grid_sha256'] and ref['absolute_tolerance_degrees']==ABS_DEG
    report=dict(state='IN_PROGRESS',started_utc=datetime.now(timezone.utc).isoformat(),
        inventory_sha256=sha(a.inventory),grid_reference_sha256=sha(a.grid_reference),
        selection_sha256=SELECTION_SHA,script_sha256=sha(__file__),absolute_tolerance_degrees=ABS_DEG,
        source_arrays_decoded=['grid_lat','grid_lon','station_id','timestamp_utc','original_sha256','schema_version'],
        source_files_hashed_in_full=True,GHI_CPP_AGRI_values_decoded=False,test_used=False,
        selected_targets=64,matched_targets=0,unmatched_targets=[],targets=[])
    try:
        for item in inv['selection']:
            if not item['pairs']:
                report['unmatched_targets'].append(dict(station=item['station'],target_time_utc=item['target_time_utc']))
                continue
            s=item['station'];expected=ref['stations'][s]
            assert expected['bounds']==inv['forecast']['station_patches'][s]
            lat_ref=np.array(expected['lat'],np.float32);lon_ref=np.array(expected['lon'],np.float32)
            assert coords_hash(lat_ref,lon_ref)==expected['coordinate_sha256']
            values=[];identities=[]
            for kind in ('original','sidecar'):
                path=Path(item[kind+'_path']);raw=path.read_bytes()
                assert hashlib.sha256(raw).hexdigest()==item[kind+'_sha256'],(kind,str(path),'SHA mismatch')
                with np.load(io.BytesIO(raw),allow_pickle=False) as z:
                    assert str(z['station_id'].item())==s
                    timestamp=str(z['timestamp_utc'].item())
                    stamp=datetime.fromisoformat(timestamp.replace('Z','+00:00'))
                    if stamp.tzinfo is None:stamp=stamp.replace(tzinfo=timezone.utc)
                    assert stamp.astimezone(timezone.utc)==datetime.fromisoformat(item['target_time_utc'])
                    lat=z['grid_lat'].copy();lon=z['grid_lon'].copy()
                    if kind=='sidecar':
                        assert str(z['original_sha256'].item())==item['original_sha256']
                        assert str(z['schema_version'].item())=='cpp_aligned_sidecar_v1'
                    identities.append(dict(kind=kind,path=str(path),sha256=item[kind+'_sha256'],
                        latitude=compare(lat,lat_ref),longitude=compare(lon,lon_ref),coordinate_sha256=coords_hash(lat,lon)))
                    values.append((lat,lon))
            original_sidecar=dict(latitude=compare(values[0][0],values[1][0]),longitude=compare(values[0][1],values[1][1]))
            row=dict(station=s,target_time_utc=item['target_time_utc'],r_pack_index=item['r_pack_index'],
                pairs=len(item['pairs']),S_patch_bounds=expected['bounds'],sources=identities,
                original_sidecar=original_sidecar)
            report['targets'].append(row);report['matched_targets']+=1
            print('COORDINATES_CHECKED',s,item['target_time_utc'],flush=True)
        assert report['matched_targets']==58 and len(report['unmatched_targets'])==6
        checks=[v for row in report['targets'] for src in row['sources'] for v in (src['latitude'],src['longitude'])]
        checks += [v for row in report['targets'] for v in row['original_sidecar'].values()]
        report['max_abs_degrees']=max(v['max_abs_degrees'] for v in checks)
        report['all_coordinates_exact_equal']=all(v['exact_equal'] for v in checks)
        report['state']='PASS_58_TARGET_COORDINATE_BINDINGS' if all(v['within_fixed_tolerance'] for v in checks) else 'COMPLETE_WITH_COORDINATE_MISMATCH'
    except Exception as error:
        report.update(state='ERROR_INCOMPLETE_COORDINATE_BINDING',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2,allow_nan=False))
        print(report['state'],flush=True)

if __name__=='__main__':main()
