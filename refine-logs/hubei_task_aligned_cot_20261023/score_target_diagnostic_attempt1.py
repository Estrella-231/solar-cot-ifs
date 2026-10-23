"""Independent full-teacher reference innovation replay and date-paired scores."""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parent;Q=P.parent/'hubei_cot_conditional_information_20261023';O=P/'target_diagnostic_attempt1'
sites=['Niushou','Wangzhai','Mizhuang'];names=['G1','reference_innovation_OFFLINE','physical','task','direct_AGRI']
mapping=json.loads((Q/'probe_attempt1/DATE_FOLDS.json').read_text())['date_to_fold'];days=sorted(mapping)
acc=np.zeros((len(days),3,3,5,2));errmax=0.;provenance=[]
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
for si,st in enumerate(sites):
    source=Q/'inputs_attempt2'/f'{st}.npz';d=load(source)
    for sj,seed in enumerate([42,43,44]):
        file=O/f'{st}_seed{seed}_OOF.npz';v=load(file)
        for key in ['truth','init_time','target_time','lead','available']:assert np.array_equal(v[key],d[key])
        assert v['arms'].tolist()==names
        x=np.column_stack([d['features'],d['bases'][sj]/d['clear']]);reference=x.copy();reference[:,279:297]=d['observed_COT_features']
        for fold in range(3):
            m=load(Q/'probe_attempt1'/st/f'seed{seed}_observed_R_OFFLINE_fold{fold}.npz')
            held=np.array([mapping[str(t)[:8]]==fold for t in d['target_time']])
            raw=1000*(m['intercept']+(x[held]-m['mean'])/m['scale']@m['coefficient'])
            actual=1000*(m['intercept']+(reference[held]-m['mean'])/m['scale']@m['coefficient'])
            answer=np.where(d['available'][held],np.maximum(0,d['bases'][sj,held]+actual-raw),d['bases'][sj,held])
            error=float(abs(answer-v['prediction'][1,held]).max());errmax=max(errmax,error);assert error<1e-8
        for di,day in enumerate(days):
            take=v['date']==day
            for ai in range(5):
                residual=v['prediction'][ai,take]-d['truth'][take]
                acc[di,si,sj,ai]=[math.fsum(float(x)**2 for x in residual),len(residual)]
        provenance.append(dict(station=st,seed=seed,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
total=acc.sum(0);per=np.sqrt(total[...,0]/total[...,1]);mean=per.mean((0,1))
w=np.random.default_rng(20261023).multinomial(len(days),np.ones(len(days))/len(days),5000)
flat=acc.reshape(len(days),-1,2);boot=np.sqrt((w@flat[:,:,0])/(w@flat[:,:,1])).reshape(5000,3,3,5).mean((1,2))
comp={}
for ai in range(1,5):comp[names[ai]]=dict(delta_rmse=float(mean[ai]-mean[0]),ci95=np.quantile(boot[:,ai]-boot[:,0],[.025,.975]).tolist(),relative_change_percent=float((mean[ai]/mean[0]-1)*100))
result=dict(state='FULL_TEACHER_REPLAY_AND_FP64_DATE_SCORE_PASS',OOF_rmse=dict(zip(names,mean.tolist())),comparisons_to_G1=comp,
    by_station={st:dict(zip(names,per[si].mean(0).tolist())) for si,st in enumerate(sites)},full_teacher_replay_max=errmax,
    dates=days,n_rows=32838,provenance=provenance,oracle_deployable=False,new_model_fits=0,test_payloads_read=0)
(O/'INDEPENDENT_SCORE.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:result[k] for k in ['OOF_rmse','comparisons_to_G1','by_station','full_teacher_replay_max']},indent=2))
