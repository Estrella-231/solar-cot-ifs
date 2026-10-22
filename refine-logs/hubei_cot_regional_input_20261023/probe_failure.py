import csv,sys,traceback
from pathlib import Path
P=Path('/home/Data_Pool/zjnu/hubei_cot_regional_input_code_20261023');sys.path.insert(0,str(P))
import extract_cpp_region as m
rows=list(csv.DictReader((P/'selected_cpp_agri.csv').open()));done=Path('/tmp/hubei_cot_regional_input_20261023_attempt1/payload')
missing=[r for r in rows[:70] if not (done/r['split']/(r['time_bjt']+'.npz')).exists()]
print('missing early keys',[r['time_bjt'] for r in missing],flush=True)
if missing:
 o=Path('/tmp/hubei_cot_regional_probe_20261023');(o/'payload/train').mkdir(parents=True,exist_ok=True);(o/'payload/val').mkdir(exist_ok=True)
 m.init(P/'grid_static256.npz',o)
 try:r=m.work(missing[0]);print('PROBE_PASS',r['time_bjt'])
 except Exception:traceback.print_exc()
