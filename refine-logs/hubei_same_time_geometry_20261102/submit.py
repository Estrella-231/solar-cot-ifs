from pathlib import Path
import subprocess,json,re
P=Path(__file__).parent
def queue():
 s=subprocess.check_output(['qstat','-u','slfu'],text=True)
 n=sum(bool(re.match(r'^\d',x)) and x.split()[4]!='C' for x in s.splitlines())
 assert n<3,(n,s)
 return s
before=queue();a=subprocess.check_output(['qsub',str(P/'prepare.pbs')],text=True).strip();after_prepare=queue()
b=subprocess.check_output(['qsub','-W','depend=afterok:'+a,str(P/'run.pbs')],text=True).strip()
after=subprocess.check_output(['qstat','-u','slfu'],text=True)
d={'prepare_job':a,'paired_job':b,'before':before,'after_prepare':after_prepare,'after_submit':after,'training_gpu_node':'node21','gpu_count':2,'batch512_unchanged':True,'test_payloads_read':0}
(P/'LAUNCH.json').write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2))
