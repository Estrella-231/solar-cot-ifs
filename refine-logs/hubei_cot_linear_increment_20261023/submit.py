import subprocess,json,xml.etree.ElementTree as ET,re
from pathlib import Path
E=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cot_linear_increment_20261023')
def queue():
    s=subprocess.check_output(['qstat','-fx','-u','slfu'],text=True);return [dict(id=j.findtext('Job_Id'),state=j.findtext('job_state'),nodes=j.findtext('Resource_List/nodes') or '') for j in ET.fromstring(s).findall('Job') if j.findtext('job_state') not in ['C','F']]
before=queue();assert len(before)<3 and sum(sum(map(int,re.findall(r'gpus=(\d+)',x['nodes']))) for x in before)+1<=8
job=subprocess.check_output(['qsub',str(E/'screen.pbs')],text=True).strip();d=dict(job=job,before=before,after=queue(),purpose='frozen baseline C-date GPU forward plus 27-parameter CPU ridge screen; no upstream training',gpus=1,short_job=True)
(E/'SUBMISSION.json').write_text(json.dumps(d,indent=2));print(d)
