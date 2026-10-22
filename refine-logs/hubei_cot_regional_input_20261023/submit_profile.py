"""Submit one prepared 4-GPU profile after actual G2 and resource checks."""
import argparse,json,re,subprocess,xml.etree.ElementTree as ET
from pathlib import Path


def queue():
    raw=subprocess.check_output(['qstat','-fx','-u','slfu'],text=True)
    answer=[]
    for j in ET.fromstring(raw).findall('Job'):
        if j.findtext('job_state') in ('C','F'):continue
        spec=j.findtext('Resource_List/nodes') or ''
        gpu=0
        for part in spec.split('+'):
            mm=re.search(r'(?:^|:)gpus=(\d+)',part)
            if mm:
                prefix=part.split(':')[0];gpu+=int(mm.group(1))*(int(prefix) if prefix.isdigit() else 1)
        ng=j.findtext('Resource_List/ngpus')
        if ng is not None:gpu=max(gpu,int(ng))
        answer.append(dict(id=j.findtext('Job_Id'),state=j.findtext('job_state'),nodes=spec,requested_gpus=gpu))
    return answer


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--node',choices=['node22','node23'],default='node22');a=ap.parse_args()
    receipt=a.root/'PROFILE_SUBMISSION_attempt1.json';intent=a.root/'PROFILE_SUBMISSION_INTENT_attempt1.json'
    assert not receipt.exists() and not intent.exists(), 'Inspect any earlier submission; never submit twice'
    pack='/dev/shm/hubei_regional_r_pack_20261023_attempt2'
    state=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',a.node,f'cat {pack}/REPORT.json'],text=True))
    assert state['state']=='REGIONAL_R_PACK_BUILT_NOT_TRAINED' and state['frames']==3124
    fixed=json.loads((a.root/'G2_ATTEMPT2_COMPLETE.json').read_text());assert fixed['state']=='PASS_BINARY_ONLY_PACK_REPAIR' and fixed['pack']==pack
    if a.node=='node23':assert json.loads((a.root/'LOCAL_STAGE_REPORT_node23.json').read_text())['state']=='PASS_NODE_LOCAL_PACK_COPY'
    assert json.loads((a.root/'source_ready_attempt1/DESTINATION_AUDIT.json').read_text())['state']=='PASS_INDEPENDENT_PACKAGE_AND_TRAIN_STATISTICS'
    before=queue();assert len(before)<3 and sum(r['requested_gpus'] for r in before)+4<=8,before
    resource=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',a.node,'nvidia-smi',
                 '--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    gpus=[]
    for line in resource.strip().splitlines():
        index,name,used,total,util=[s.strip() for s in line.split(',')]
        gpus.append(dict(index=int(index),name=name,memory_used_MiB=int(used),memory_total_MiB=int(total),utilization=int(util)))
    assert sum(g['memory_used_MiB']<500 and g['utilization']<5 for g in gpus)>=4, 'Fewer than4 physically idle GPUs'
    script=a.root/f'profile_{a.node}_attempt1.pbs'
    text=(a.root/'profile_regional_r.pbs').read_text().replace('nodes=node22:',f'nodes={a.node}:').replace('node22-gpu/',f'{a.node}-gpu/')
    script.write_text(text);subprocess.run(['bash','-n',str(script)],check=True)
    intent.write_text(json.dumps(dict(queue_before=before,gpus=gpus,purpose='2/4 GPU correctness, checkpoint replay and throughput only'),indent=2))
    job=subprocess.check_output(['qsub',str(script)],text=True).strip()
    report=dict(job_id=job,queue_before=before,queue_after=queue(),physical_gpus_before=gpus,node=a.node,
                server='ZJNU college, not Fudan',requested_gpus=4,purpose='profile; no formal R/GHI training',test_payloads_read=0)
    receipt.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
