"""Pack geometry24 forecast images for paired train/val GHI-head comparison.

Rows remain keyed by the frozen head pack.  Each source bank contains
forecast 13-channel AGRI plus the correctly aligned 16 target-time geometry
maps; future real AGRI is absent from this pack.
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

def main(root, train_bank, val_bank, out):
    root, train_bank, val_bank, out = map(Path, (root, train_bank, val_bank, out))
    if out.exists(): raise RuntimeError('refuse to overwrite v5 sidecar')
    base = root/'data/head_pack_trainval_20260908_v1'
    source_config = json.loads((root/'configs/s_frozen_hunan_seed42.json').read_text())
    if sha(source_config['manifest']) != source_config['manifest_sha256']:
        raise RuntimeError('manifest hash drift')
    manifest_ids = {'train':[], 'val':[]}
    with Path(source_config['manifest']).open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['split'] in manifest_ids:
                manifest_ids[row['split']].append(row['seq_id'])
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
    banks = {'train': train_bank, 'val': val_bank}
    bank_sha = {}
    for name, directory in banks.items():
        contract = json.loads((directory/'complete.json').read_text())
        if (contract['state'] != 'COMPLETE_GEOMETRY24_FORECAST_COT' or contract['test_used'] or
                contract['split'] != name or contract['geometry_contract'] != 'history_8_then_target_16_v2' or
                contract['geometry_forward_indices'] != list(range(8,24))):
            raise RuntimeError('invalid geometry24 source bank: '+name)
        path = directory/'forecast_image_physical.npy'
        if sha(path) != contract['forecast_image_sha256']:
            raise RuntimeError('forecast image hash drift: '+str(path))
        index_path = directory/'index.csv'
        if sha(index_path) != contract['index_sha256']:
            raise RuntimeError('forecast sequence-index hash drift: '+name)
        with index_path.open(newline='') as stream:
            bank_ids = list(csv.DictReader(stream))
        if (len(bank_ids) != len(manifest_ids[name]) or
                any(int(item['index']) != i or item['seq_id'] != manifest_ids[name][i] or item['split'] != name
                    for i,item in enumerate(bank_ids))):
            raise RuntimeError('forecast bank sequence order mismatch: '+name)
        source = np.load(path, mmap_mode='r', allow_pickle=False)
        if source.shape != (lookup[name].shape[0],2,16,16,16,16):
            raise RuntimeError('forecast image shape drift: '+name)
        for start in range(0, len(source), 32):
            stop = min(start+32, len(source))
            targets = lookup[name][start:stop].reshape(-1)
            available = targets >= 0
            images = source[start:stop].reshape(-1,16,16,16)
            if not np.isfinite(images[available]).all():
                raise RuntimeError('nonfinite geometry24 source: '+name)
            image[targets[available]] = images[available]
        bank_sha[name] = sha(directory/'complete.json')
    if not np.isfinite(image).all(): raise RuntimeError('incomplete image sidecar')
    train_ids=np.where(split==0)[0]; sums=np.zeros(16,np.float64); squares=np.zeros(16,np.float64); count=0
    for ids in np.array_split(train_ids, max(1,len(train_ids)//1024)):
        x=image[ids].astype(np.float64); sums+=x.sum((0,2,3)); squares+=(x*x).sum((0,2,3)); count+=x.shape[0]*256
    mean, std=sums/count, np.sqrt(squares/count-(sums/count)**2)
    if not np.isfinite(std).all() or (std[:15]<=0).any(): raise RuntimeError('bad train statistics')
    for ids in np.array_split(np.arange(TOTAL), max(1,TOTAL//1024)):
        image[ids,:15]=(image[ids,:15]-mean[:15,None,None])/std[:15,None,None]
    image.flush()
    atomic(out/'audit.json', {'state':'COMPLETE_GEOMETRY24_SOLARRESNET_V5_INPUT_SIDECAR','test_used':False,
      'image_layout':'[row,13 AGRI + cosSOZ + cosRAA + day_mask,16,16]','metadata_layout':'[row,cosSZA,hour_sin,hour_cos,doy_sin,doy_cos]',
      'image_train_mean':mean.tolist(),'image_train_std':std.tolist(),'day_mask_unchanged':True,
      'geometry_contract':'history_8_then_target_16_v2','bank_complete_sha256':bank_sha,
      'source_head_pack_audit_sha256':sha(base/'audit.json'),'image_sha256':sha(out/'image.npy'),'metadata_sha256':sha(out/'metadata.npy')})
    print('COMPLETE_GEOMETRY24_SOLARRESNET_V5_INPUT_SIDECAR')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--train-bank',type=Path,required=True); p.add_argument('--val-bank',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); main(a.root,a.train_bank,a.val_bank,a.out)
