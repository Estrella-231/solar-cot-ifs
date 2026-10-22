"""Three frozen-forward workers, one per station, no GPU optimization."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--gpus',required=True);a=ap.parse_args()
gpus=a.gpus.split(',');assert len(gpus)==3 and len(set(gpus))==3
for gpu in gpus:
    fields=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert int(fields[0])<500 and int(fields[1])<5,(gpu,fields)

out=Path(__file__).resolve().parent/'replay';out.mkdir(exist_ok=False);tasks=[]
for st,gpu in zip(['Niushou','Wangzhai','Mizhuang'],gpus):
    log=(out/f'{st}.log').open('w')
    proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('train_station.py')),'--station',st,'--out',str(out/st)],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT)
    tasks.append((proc,log,st))
failures=[]
for proc,log,st in tasks:
    rc=proc.wait();log.close()
    if rc:failures.append(dict(station=st,exit=rc))
assert not failures,failures

(out/'COMPLETE.json').write_text(json.dumps(dict(state='CALIBRATION_COVERAGE_FOUR_ARM_COMPLETE',test_buffer_payloads_read=0),indent=2))
