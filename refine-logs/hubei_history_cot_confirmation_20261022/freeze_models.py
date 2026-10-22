"""Bind existing S/R/GHI weights and date metadata, without loading models."""
import csv,hashlib,json,re
from datetime import datetime,timedelta
from pathlib import Path
ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments')
BASE=ROOT/'hubei_solar_geometry_repair_20261021'
HEAD=ROOT/'hubei_forecast_cot_increment_20261022'
OUT=ROOT/'hubei_history_cot_confirmation_20261022'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def main():
    OUT.mkdir(exist_ok=True);meta=json.loads((BASE/'S_runs_attempt12/full/run_metadata.json').read_text())
    assert sha(meta['best_checkpoint'])==meta['best_checkpoint_sha256']
    index=BASE/'prepared_attempt12/indexes/full.csv';assert sha(index)==meta['index_sha256']
    stamps=[]
    for r in rows(index):
        assert r['split'] in ['train','val']
        for p in r['data_paths'].split('|'):
            s=Path(p).stem;assert re.fullmatch(r'\d{12}',s)
            stamps.append(datetime.strptime(s,'%Y%m%d%H%M')+timedelta(hours=8))
    assert max(stamps)<datetime(2026,7,15)
    rp=BASE/'prepared_attempt12/R_packs/full';rmeta=json.loads((BASE/'R_runs_attempt12/full/run_metadata.json').read_text())
    assert sha(rp/'rows.csv')==rmeta['source_pack_hashes']['rows.csv']
    rtimes=[datetime.strptime(r['time_bjt'],'%Y%m%d%H%M') for r in rows(rp/'rows.csv')]
    assert max(rtimes)<datetime(2026,7,15)
    records=[]
    for station in ['Niushou','Wangzhai','Mizhuang']:
        rr=rows(HEAD/'prepared_attempt1'/station/'rows.csv')
        assert all(datetime.strptime(r['target_time_bjt'],'%Y%m%d%H%M')<datetime(2026,7,15) for r in rr)
        for seed in [42,43,44]:
            d=HEAD/'training_attempt1/pairs'/f'{station}_seed{seed}';m=json.loads((d/'COMPLETE.json').read_text())
            for arm in ['G0','G1']:
                p=d/(arm+'_best.pt');digest=sha(p);assert digest==m['arms'][arm]['checkpoint_sha256']
                records.append(dict(station=station,seed=seed,arm=arm,path=str(p),sha256=digest,
                    chosen_epoch=m['arms'][arm]['best_epoch'],norm_source=str(d/'run_config.json'),
                    norm_source_sha256=sha(d/'run_config.json')))
    report=dict(state='FROZEN_WEIGHTS_DATES_PRIOR_TO_NEW_PERIOD_NO_NEW_SCORES',S=meta,
        S_date_range_BJT=[min(stamps).isoformat(' '),max(stamps).isoformat(' ')],
        S_index=str(index),S_index_sha256=sha(index),S_norm_path=str(BASE/'prepared_attempt12/norms/full.npz'),
        S_norm_sha256=sha(BASE/'prepared_attempt12/norms/full.npz'),
        R_checkpoint=str(BASE/'R_runs_attempt12/full/best.pt'),R_checkpoint_sha256=sha(BASE/'R_runs_attempt12/full/best.pt'),
        R_norm_path=str(rp/'norm.json'),R_norm_sha256=sha(rp/'norm.json'),
        R_date_range_BJT=[min(rtimes).isoformat(' '),max(rtimes).isoformat(' ')],
        GHI_heads=records,old_test_payloads_read=0,old_buffer_payloads_read=0,new_period_scores_computed=0,
        prior_contact_note='This binds actual current source dates; full historical project contact inventory remains separate.')
    (OUT/'MODEL_FREEZE.json').write_text(json.dumps(report,indent=2)+'\n');print(report['state'])
if __name__=='__main__':main()
