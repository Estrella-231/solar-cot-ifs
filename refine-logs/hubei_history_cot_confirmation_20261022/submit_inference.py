"""One qsub with live whole-account task and GPU-request limits."""
import argparse,json,re,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
def queue():
    raw=subprocess.check_output(['qstat','-fx','-u','slfu'],text=True)
    answer=[]
    for j in ET.fromstring(raw).findall('Job'):
        if j.findtext('job_state') in ['C','F']:continue
        spec=j.findtext('Resource_List/nodes') or ''
        gpu=sum(int(n) for n in re.findall(r'gpus=(\d+)',spec))
        answer.append(dict(id=j.findtext('Job_Id'),state=j.findtext('job_state'),nodes=spec,requested_gpus=gpu))
    return answer
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--attempt',type=int,default=1);a=p.parse_args()
    receipt=a.root/('SUBMISSION.json' if a.attempt==1 else f'SUBMISSION_attempt{a.attempt}.json');assert not receipt.exists()
    transfer=json.loads((a.root/'TRANSFER_REPORT.json').read_text());assert transfer['state']=='TRANSFER_COMPLETE'
    inp=json.loads((a.root/'INPUT_SELECTION.json').read_text());assert inp['state']=='FROZEN_NEW_PERIOD_INPUT_KEYS_NO_SCORES'
    before=queue();assert len(before)<3 and sum(j['requested_gpus'] for j in before)+4<=8,before
    resource=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','node23','nvidia-smi',
        '--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True)
    (a.root/f'resource_before_submission_attempt{a.attempt}.txt').write_text(resource)
    script=a.root/f'code_attempt{a.attempt}/infer.pbs'
    subprocess.run(['bash','-n',str(script)],check=True)
    job=subprocess.check_output(['qsub',str(script)],text=True).strip()
    report=dict(job_id=job,queue_before=before,queue_after=queue(),requested_node='node23',requested_gpus=4,
        gpu_type='A100-SXM4-40GB',purpose='frozen-model new-period inference, no training',
        old_test_payloads_read=0,old_buffer_payloads_read=0)
    receipt.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
