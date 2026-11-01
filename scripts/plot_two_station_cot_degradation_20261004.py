"""Recompute historical matched GHI pilot comparison from saved predictions."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results/regional_station_cot_20260927/ghi_pilot'
OUT=ROOT/'results/two_station_cot_degradation_20261004'
OUT.mkdir(exist_ok=False)
arms=['zero','forecast','oracle']
a={k:dict(np.load(SRC/f'{k}_validation_predictions.npz')) for k in arms}
for k in arms[1:]:
    for f in ['row_ids','true_ghi','station','lead']:
        assert np.array_equal(a['zero'][f],a[k][f]),(k,f)
assert np.unique(a['zero']['row_ids']).size==len(a['zero']['row_ids'])
assert set(a['zero']['lead'])==set(range(16))
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
colors=['#666666','#0072B2','#D55E00']
labels=['无 COT','预测 COT','真实未来 AGRI→R（oracle）']
stats={};rows=[]
for s,name in [(0,'四里'),(1,'竺家')]:
    m=a['zero']['station']==s
    stats[name]={k:{'n':int(m.sum()),'rmse':float(np.sqrt(np.mean((a[k]['pred_ghi'][m]-a[k]['true_ghi'][m])**2)))} for k in arms}
    for i in range(16):
        q=m & (a['zero']['lead']==i)
        assert q.sum()>0
        row={'station':name,'lead_minutes':15*(i+1),'n':int(q.sum())}
        for k in arms:
            row[k]=float(np.sqrt(np.mean((a[k]['pred_ghi'][q]-a[k]['true_ghi'][q])**2)))
        row['forecast_delta']=row['forecast']-row['zero']
        row['oracle_delta']=row['oracle']-row['zero']
        rows.append(row)
with (OUT/'lead_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
xs=np.arange(2)
for j,k in enumerate(arms):
    values=[stats[n][k]['rmse'] for n in stats]
    bars=ax[0,0].bar(xs+(j-1)*.24,values,width=.23,color=colors[j],label=labels[j])
    ax[0,0].bar_label(bars,fmt='%.2f',fontsize=9,padding=3)
ax[0,0].set(xticks=xs,xticklabels=['四里（N=1632）','竺家（N=1624）'],ylabel='GHI RMSE（W/m²）',ylim=(0,195),title='全16时效：四里改善，竺家退化')
ax[0,0].legend(fontsize=8,loc='upper right')
for s,name in enumerate(stats):
    rr=[r for r in rows if r['station']==name]
    ax[0,1].plot([r['lead_minutes'] for r in rr],[r['forecast_delta'] for r in rr],'-o',color=['#009E73','#CC79A7'][s],label=name)
    for j,k in enumerate(arms):
        ax[1,s].plot([r['lead_minutes'] for r in rr],[r[k] for r in rr],['-o','-s','--^'][j],color=colors[j],markersize=4,label=labels[j])
    ax[1,s].set(title=name+'：逐时效 RMSE',xlabel='距历史末帧的时效（分钟）',ylabel='GHI RMSE（W/m²）',ylim=(95,205),xticks=[15,60,120,180,240])
    ax[1,s].legend(fontsize=8)
ax[0,1].axhline(0,color='black',lw=.8)
ax[0,1].set(title='预测COT组 − 无COT组：正值代表退化',xlabel='距历史末帧的时效（分钟）',ylabel='ΔRMSE（W/m²）',xticks=[15,60,120,180,240])
ax[0,1].legend()
for axis in ax.flat:axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
fig.suptitle('2026-09-27旧验证pilot：四里与竺家 COT→GHI 配对对照',fontsize=15)
fig.supxlabel('各组行键、真值与有效样本一致；独立训练的头，非纯输入因果消融。单种子；未使用test。',fontsize=9)
fig.savefig(OUT/'two_station_comparison.png',dpi=200)
fig.savefig(OUT/'two_station_comparison.svg')
plt.close(fig)
manifest={'overall':stats,'test_used':False,'source_hashes':{k:hashlib.sha256((SRC/f'{k}_validation_predictions.npz').read_bytes()).hexdigest() for k in arms},'scope':'historical validation pilot, independent trained heads; station mapping 0=Sili 1=Zhujia; matched row ids/truth/station/lead asserted; finite masked cohort already saved; no confidence intervals or smoothing','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'summary.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'README.md').write_text('旧验证逐样本输出重算；不代表新按月划分成绩。两站各自组间同样本，不要求不同站点的样本相同。各lead支持数不同。\n\n绘图流程使用 scientific-visualization 技能。Kassis et al. (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents, https://doi.org/10.48550/arXiv.2609.00065 。流程引用不支持性能结论。\n',encoding='utf-8')
print(json.dumps(stats,ensure_ascii=False))
