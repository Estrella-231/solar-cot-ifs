import importlib.util,json,subprocess,time
from pathlib import Path
BASE=Path("/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021")
spec=importlib.util.spec_from_file_location("queue_controller",BASE/"code_pipeline_attempt12/queue_controller.py")
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
script=BASE/"prepare_data_attempt12.pbs"
assert script.is_file() and not (BASE/"prepared_attempt12").exists()
assert not (BASE/"PREP_SUBMISSION_attempt12.json").exists()
before=q.queue()
print("QUEUE_BEFORE",json.dumps(before,sort_keys=True),flush=True)
if len(before)>=3:raise RuntimeError("three unfinished slfu jobs; refusing submission")
if any(v.get("name")=="hb_solar_data12" for v in before.values()):raise RuntimeError("attempt12 data job already active")
p=subprocess.run([q.QSUB,str(script)],capture_output=True,text=True)
receipt={"submitted_unix":time.time(),"returncode":p.returncode,"stdout":p.stdout.strip(),"stderr":p.stderr.strip(),"queue_before":before}
print("QSUB",json.dumps(receipt,sort_keys=True),flush=True)
if p.returncode:raise SystemExit(p.returncode)
jid=p.stdout.strip()
if not jid or not jid[0].isdigit():raise RuntimeError("unrecognized qsub output; inspect queue before any retry")
after=q.queue();receipt["job_id"]=jid;receipt["queue_after"]=after
(BASE/"PREP_SUBMISSION_attempt12.json").write_text(json.dumps(receipt,indent=2))
print("QUEUE_AFTER",json.dumps(after,sort_keys=True),flush=True)
if jid not in after:print("JOB_NOT_IN_IMMEDIATE_QSTAT; inspect qstat -f before retry",flush=True)
