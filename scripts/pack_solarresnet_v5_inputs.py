"""Build the test-free SolarResNet-v5 image and metadata sidecar.

Rows remain keyed by the frozen head pack.  The script reads only its frozen
forecast cache: 13 forecast AGRI channels and deterministic geometry maps.
"""
import argparse, csv, hashlib, json
from datetime import datetime
from pathlib import Path
import numpy as np

TOTAL, TRAIN, VAL = 637902, 538053, 99849

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''): h.update(chunk)
    return h.hexdigest()

def atomic(path, data):
    tmp = Path(path).with_suffix('.tmp'); tmp.write_text(json.dumps(data, indent=2)); tmp.replace(path)

def metadata(rows):
    value = np.empty((len(rows), 5), np.float32)
    for i, row in enumerate(rows):
        t = datetime.fromisoformat(row['target_time_bjt'])
        hour, doy = t.hour + t.minute / 60, t.timetuple().tm_yday
        value[i] = (float(row['solar_cosine']), np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                    np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25))
    return value

def main(root, out):
    root, out = Path(root), Path(out)
    if out.exists(): raise RuntimeError('refuse to overwrite v5 sidecar')
    base, cache = root/'data/head_pack_trainval_20260908_v1', root/'data/frozen_forecast_trainval_20260907_v1'
    audit = json.loads((base/'audit.json').read_text())
    if audit['test_used'] or not audit['S_frozen'] or not audit['R_frozen']: raise RuntimeError('invalid frozen source')
    split, sequence, station, lead = (np.load(base/(k+'.npy'), mmap_mode='r') for k in ('split','sequence','station','lead'))
    if (len(split), int((split==0).sum()), int((split==1).sum())) != (TOTAL, TRAIN, VAL): raise RuntimeError('row contract drift')
    with (base/'rows.csv').open(newline='', encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    if len(rows) != TOTAL: raise RuntimeError('rows.csv length drift')
    out.mkdir(parents=True); image=np.lib.format.open_memmap(out/'image.npy','w+',np.float32,(TOTAL,16,16,16)); image[:] = np.nan
    meta=np.lib.format.open_memmap(out/'metadata.npy','w+',np.float32,(TOTAL,5)); meta[:] = metadata(rows); meta.flush()
    lookup={}
    for name, flag, slots in (('train',0,27675),('val',1,4258)):
        table=np.full((slots,2,16),-1,np.int64); ids=np.where(split==flag)[0]
        table[sequence[ids],station[ids],lead[ids]]=ids
        if (table[sequence[ids],station[ids],lead[ids]] != ids).any(): raise RuntimeError('duplicate sequence key')
        lookup[name]=table
    specs=json.loads((base/'source_shards.json').read_text())
    for spec in specs:
        path=cache/spec['split']/spec['name']
        if sha(path)!=spec['sha256']: raise RuntimeError('source hash drift: '+str(path))
        with np.load(path, allow_pickle=False) as z:
            ids, agri, geom=z['indices'], z['predicted_agri_physical'], z['geometry']
            if agri.shape[1:] != (2,16,13,16,16) or geom.shape[1:] != (2,16,3,16,16): raise RuntimeError('cache shape drift')
            target=lookup[spec['split']][ids]
            for b in range(len(ids)):
                # The frozen forecast cache contains every sequence/station/lead,
                # while the audited head pack intentionally omits invalid rows.
                # Write only rows retained by the head pack; missing entries are
                # expected and must not be treated as cache corruption.
                for station in range(2):
                    for lead in range(16):
                        row = int(target[b, station, lead])
                        if row >= 0:
                            image[row] = np.concatenate((agri[b, station, lead],
                                                         geom[b, station, lead]), axis=0)
    if not np.isfinite(image).all(): raise RuntimeError('incomplete image sidecar')
    train_ids=np.where(split==0)[0]; sums=np.zeros(16,np.float64); squares=np.zeros(16,np.float64); count=0
    for ids in np.array_split(train_ids, max(1,len(train_ids)//1024)):
        x=image[ids].astype(np.float64); sums+=x.sum((0,2,3)); squares+=(x*x).sum((0,2,3)); count+=x.shape[0]*256
    mean, std=sums/count, np.sqrt(squares/count-(sums/count)**2)
    if not np.isfinite(std).all() or (std[:15]<=0).any(): raise RuntimeError('bad train statistics')
    for ids in np.array_split(np.arange(TOTAL), max(1,TOTAL//1024)):
        image[ids,:15]=(image[ids,:15]-mean[:15,None,None])/std[:15,None,None]
    image.flush()
    atomic(out/'audit.json', {'state':'COMPLETE_SOLARRESNET_V5_INPUT_SIDECAR','test_used':False,
      'image_layout':'[row,13 AGRI + cosSOZ + cosRAA + day_mask,16,16]','metadata_layout':'[row,cosSZA,hour_sin,hour_cos,doy_sin,doy_cos]',
      'image_train_mean':mean.tolist(),'image_train_std':std.tolist(),'day_mask_unchanged':True,
      'source_head_pack_audit_sha256':sha(base/'audit.json'),'image_sha256':sha(out/'image.npy'),'metadata_sha256':sha(out/'metadata.npy')})
    print('COMPLETE_SOLARRESNET_V5_INPUT_SIDECAR')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); main(a.root,a.out)
