"""Continue the approved frozen protocol when the data transfer completes."""
import json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
REMOTE='/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_history_cot_confirmation_20261022'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','zjnu-hpc']
SCP=['scp','-o','BatchMode=yes','-o','ConnectTimeout=12']
def run(cmd):subprocess.run(list(map(str,cmd)),check=True)
def main():
    assert not (HERE/'AUTO_SUBMISSION.json').exists()
    started=time.monotonic()
    while not (HERE/'TRANSFER_REPORT.json').exists():
        if time.monotonic()-started>3600:raise TimeoutError('Transfer not finished; no GPU task submitted')
        time.sleep(15)
    transfer=json.loads((HERE/'TRANSFER_REPORT.json').read_text());assert transfer['state']=='TRANSFER_COMPLETE'
    files=[HERE/p for p in ['MODEL_FREEZE.json','TRANSFER_REPORT.json']]
    files += [HERE/'final_inputs_attempt2/INPUT_SELECTION.json',HERE/'final_inputs_attempt2/keys.csv',HERE/'labels_attempt1/labels_with_clear.csv']
    run(SCP+files+['zjnu-hpc:'+REMOTE+'/'])
    for name in ['infer_new_period.py','replay_heads.py','run_inference.py','score_new_period.py','submit_inference.py','infer.pbs']:
        run(SCP+[HERE/name,'zjnu-hpc:'+REMOTE+'/code_attempt1/'])
    run(SSH+[f'cp /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021/grid_static256.npz {REMOTE}/grid_static256.npz'])
    cmd='/public/home/slfu/miniconda3/envs/swc/bin/python '+REMOTE+'/code_attempt1/submit_inference.py --root '+REMOTE
    # A busy account is a resource wait, never a reason to exceed the limits.
    for _ in range(30):
        reply=subprocess.run(SSH+[cmd],text=True,capture_output=True)
        (HERE/'auto_submission_last_stdout.log').write_text(reply.stdout)
        (HERE/'auto_submission_last_stderr.log').write_text(reply.stderr)
        if reply.returncode==0:
            report=json.loads(reply.stdout.strip().splitlines()[-1]);(HERE/'AUTO_SUBMISSION.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report),flush=True);return
        if 'AssertionError: [' not in reply.stderr:raise RuntimeError(reply.stderr)
        time.sleep(60)
    raise TimeoutError('Account resource cap still reached; no submission')
if __name__=='__main__':main()
