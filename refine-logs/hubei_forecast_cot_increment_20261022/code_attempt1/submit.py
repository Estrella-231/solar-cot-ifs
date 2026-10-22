import argparse,json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
def queue():
    raw=subprocess.check_output(['qstat','-fx','-u','slfu'],text=True)
    return [{'id':j.findtext('Job_Id'),'state':j.findtext('job_state')} for j in ET.fromstring(raw).findall('Job') if j.findtext('job_state') not in ['C','F']]
a=argparse.ArgumentParser();a.add_argument('script');a.add_argument('receipt');x=a.parse_args()
before=queue();assert len(before)<3, before
job=subprocess.check_output(['qsub',x.script],text=True).strip()
result={'job_id':job,'queue_before':before,'queue_after':queue(),'script':x.script}
Path(x.receipt).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
