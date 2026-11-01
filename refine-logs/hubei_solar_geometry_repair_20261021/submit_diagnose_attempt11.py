import importlib.util,json,subprocess,time
from pathlib import Path
B=Path("/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021 ".strip())
spec=importlib.util.spec_from_file_location("qc",B/"code_pipeline_attempt10/queue_controller.py");q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
before=q.queue();print("QUEUE_BEFORE",json.dumps(before,sort_keys=True),flush=True)
if len(before)>=3:raise RuntimeError("three unfinished slfu jobs; refusing diagnostic")
p=subprocess.run([q.QSUB,str(B/"diagnose_attempt11.pbs")],capture_output=True,text=True);print("QSUB",p.returncode,p.stdout.strip(),p.stderr.strip(),flush=True)
if p.returncode:raise SystemExit(p.returncode)
jid=p.stdout.strip();assert jid and jid[0].isdigit()
after=q.queue();print("QUEUE_AFTER",json.dumps(after,sort_keys=True),flush=True)
(B/"DIAG_SUBMISSION_attempt11.json").write_text(json.dumps({"job_id":jid,"submitted_unix":time.time(),"queue_before":before,"queue_after":after},indent=2))
