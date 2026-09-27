"""Submit one reviewed pilot after live cross-project PBS/GPU cap checks."""
import json
import re
import subprocess
from pathlib import Path

ROOT=Path("/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907")
EXP=ROOT/"experiments/regional_station_cot_20260925"
PBS=Path("/opt/gridview/pbs/dispatcher/bin")


def main():
    marker=EXP/"regional_ghi_pilot_20260927_v1_submission.json"
    if marker.exists() or (EXP/"regional_ghi_pilot_bank_20260927_v1").exists():
        raise RuntimeError("existing submission/output; inspect rather than duplicate")
    listing=subprocess.check_output([str(PBS/"qstat"),"-u","slfu"],text=True)
    print(listing,flush=True)
    jobs=[]
    for line in listing.splitlines():
        fields=line.split()
        if fields and re.match(r"^\d+\.",fields[0]) and fields[-2] != "C":
            jobs.append(fields[0])
    if len(jobs)>=3:
        raise RuntimeError("3 unfinished PBS jobs already present")
    gpus=0
    for job in jobs:
        detail=subprocess.check_output([str(PBS/"qstat"),"-f",job],text=True)
        node_line=re.search(r"Resource_List.nodes\s*=\s*([^\n]+)",detail)
        if not node_line:
            raise RuntimeError("cannot conservatively determine GPU allocation for "+job)
        gpus+=sum(int(v) for v in re.findall(r"gpus=(\d+)",node_line[1]))
    if gpus+1>8:
        raise RuntimeError("new job exceeds 8-GPU account cap")
    script=EXP/"scripts/run_regional_cot_ghi_pilot_20260927_node21.pbs"
    subprocess.run(["bash","-n",str(script)],check=True)
    job=subprocess.check_output([str(PBS/"qsub"),str(script)],text=True).strip()
    marker.write_text(json.dumps({"job_id":job,"existing_jobs":jobs,"existing_gpus":gpus,
                                "new_gpus":1,"test_used":False},indent=2)+"\n")
    print("SUBMITTED",job,flush=True)
    print(subprocess.check_output([str(PBS/"qstat"),"-u","slfu"],text=True),flush=True)


if __name__=="__main__":main()
