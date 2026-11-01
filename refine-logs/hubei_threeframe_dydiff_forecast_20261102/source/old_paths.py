from pathlib import Path
import hashlib,json
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
EXPS=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments')
OLD=EXPS/'hubei_solar_geometry_repair_20261021'
HEADS=EXPS/'hubei_training_input_source_20261022_v1'
PACK=OLD/'GHI_pack_attempt12'
BANK=OLD/'S_banks_attempt12/full/val'
PREP=BASE/'prepared_v2'
RAW=Path('/public/home/slfu/ttzhou/Auxiliary_data/HuBei_AGRI_Preprocessed/data')
STATIONS=('Niushou','Wangzhai','Mizhuang')
CENTERS={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)}
SEEDS=(42,43,44)
ARMS=('REAL','REAL_COT')
CHANNELS=(0,1,2,3,4,5,8,9,10,11,12,13,14)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
