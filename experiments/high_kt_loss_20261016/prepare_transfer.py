"""Export only named, previously accepted train/validation arrays."""
import hashlib,json,tarfile
from pathlib import Path

ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
OUT=ROOT/'experiments/high_kt_loss_20261016_transfer_v1'
pack=ROOT/'experiments/cot_interval_history_20261015_v1/data_attempt2'
bank=ROOT/'experiments/r_geometry_downstream_ghi_20261016/data/paired_r_cot_bank_v1'
m=json.loads((pack/'pack_manifest.json').read_text());b=json.loads((bank/'bank_manifest.json').read_text())
assert m['split_id']=='hunan_monthly_block_2d_buffer1d_seed42_v1' and m['test_payload_opened'] is False
assert b['state']=='COMPLETE_PAIRED_R_COT_HISTORY_BANK' and b['test_payloads_read']==0
OUT.mkdir(exist_ok=False)
records=[]
with tarfile.open(OUT/'trainval_inputs.tar.gz','w:gz',compresslevel=1) as archive:
    for folder,names,prefix in [(pack,['pack_manifest.json','rows.csv','agri_raw.npy','agri_valid.npy','geometry.npy'],'pack'),(bank,['bank_manifest.json','cot_old_log1p.npy','cot_corrected_log1p.npy'],'bank')]:
        for name in names:
            p=folder/name;key=prefix+'/'+name
            records.append(dict(path=key,source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
            archive.add(p,arcname=key,recursive=False)
(OUT/'transfer_receipt.json').write_text(json.dumps(dict(files=records,test_payloads_read=0),indent=2)+'\n')
print(json.dumps(dict(output=str(OUT),files=len(records),archive_bytes=(OUT/'trainval_inputs.tar.gz').stat().st_size)))
