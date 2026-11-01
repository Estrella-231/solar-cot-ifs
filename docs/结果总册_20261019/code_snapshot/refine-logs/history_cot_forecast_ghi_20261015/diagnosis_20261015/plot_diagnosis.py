from pathlib import Path
from datetime import datetime,timedelta
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

D=Path(__file__).resolve().parent;R=D.parent/'results_attempt2'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
colors={'A':'#0072B2','transport':'#D55E00','hybrid':'#009E73'}
labels={'A':'无 COT','transport':'输运 COT','hybrid':'混合修正 COT'}
cases=json.loads((D/'selected_cases.json').read_text())
raw={arm:np.stack([np.load(R/f'{arm}_seed{s}'/'predictions.npz')['pred_ghi'] for s in (42,43,44)]) for arm in colors}
ref=np.load(R/'A_seed42/predictions.npz');pack=np.load(R/'paired_pack.npz');early=np.load(D/'early_selected_predictions.npz')
rgbs=np.concatenate([np.load(D/f"case_{c['sequence']}.npz")['rgb'].reshape(-1,3) for c in cases])
low=np.nanpercentile(rgbs,2,axis=0);high=np.nanpercentile(rgbs,98,axis=0)
(D/'rgb_stretch.json').write_text(json.dumps(dict(channels='C03/C02/C01 physical reflectance proxy',low=low.tolist(),high=high.tolist(),gamma=.65,shared_across_all_cases=True),indent=2))
def save(fig,name):
    fig.savefig(D/(name+'.png'),dpi=165,bbox_inches='tight');fig.savefig(D/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def rgb(v):return np.clip((v-low)/(high-low),0,1)**.65
for c in cases:
    seq=c['sequence'];z=np.load(D/f'case_{seq}.npz');mask=ref['init_time']==c['init'];ix=np.flatnonzero(mask)
    assert np.array_equal(ref['row_ids'][ix],z['row_ids']) and np.array_equal(ref['lead_minutes'][ix],np.arange(15,241,15))
    t=ref['lead_minutes'][ix];init=datetime.fromisoformat(c['init'])
    fig=plt.figure(figsize=(13,13.2));g=GridSpec(6,4,figure=fig,height_ratios=[1,1,1,1,1,1.65],hspace=.19,wspace=.03)
    rowlabels=['观测 AGRI\n伪彩色','预测 AGRI\n同通道/同拉伸','观测图 → R\n事后诊断参考','历史 COT 输运','混合修正 COT']
    for col,ld in enumerate((3,7,11,15)):
        panels=[rgb(np.where(z['rgb_valid'][ld],z['rgb'][ld],np.nan)),rgb(z['forecast_rgb'][ld]),z['oracle_cot'][ld],np.where(z['transport_support'][ld],z['transport'][ld],np.nan),np.where(z['transport_support'][ld],z['hybrid'][ld],np.nan)]
        for row,v in enumerate(panels):
            ax=fig.add_subplot(g[row,col]);ax.set_facecolor('#dddddd')
            if row<2:ax.imshow(v,interpolation='nearest')
            else:im=ax.imshow(v,cmap='magma',vmin=0,vmax=70,interpolation='nearest')
            ax.plot(32,32,'+',color='cyan',ms=8,mew=1.3);ax.set_xticks([]);ax.set_yticks([])
            if col==0:ax.set_ylabel(rowlabels[row],fontsize=10)
            if row==0:ax.set_xlabel(f"+{(ld+1)*15} min / {(init+timedelta(minutes=(ld+1)*15)).strftime('%H:%M')}");ax.xaxis.set_label_position('top')
    ca=fig.add_axes([.925,.34,.012,.27]);fig.colorbar(im,cax=ca,label='检索/模型 COT（无量纲）',extend='max')
    ax=fig.add_subplot(g[5,:2]);ax.plot(t,z['oracle_cot'][:,32,32],'-o',color='#222222',ms=3,label='观测图 → R（事后诊断）')
    cpp=np.where(z['cpp_valid'][:,32,32],z['cpp'][:,32,32],np.nan);ax.plot(t,cpp,':',color='#777777',lw=2,label='CPP 检索参考')
    for arm in ('transport','hybrid'):ax.plot(t,np.where(z['transport_support'][:,32,32],z[arm][:,32,32],np.nan),'-s',color=colors[arm],ms=3,label=labels[arm])
    ax.set_xlabel('未来时效（分钟）');ax.set_ylabel('站点中心 COT（检索/模型）');ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False);ax.set_xticks([15,60,120,180,240])
    ax=fig.add_subplot(g[5,2:]);ax.plot(t,ref['true_ghi'][ix],'k-o',lw=2,ms=4,label='实测 15min 均值')
    ax.plot(t,pack['val_clear'][ix],':',color='#bbbbbb',label='区间晴空 GHI')
    for arm,p in raw.items():
        ax.plot(t,p[:,ix].mean(0),color=colors[arm],label=labels[arm]);ax.fill_between(t,p[:,ix].min(0),p[:,ix].max(0),color=colors[arm],alpha=.10)
    ax.set_xlabel('未来时效（分钟）');ax.set_ylabel('GHI（W/m²）');ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False);ax.set_xticks([15,60,120,180,240])
    fig.text(.5,.025,f"四里站；初始化 {init.strftime('%Y-%m-%d %H:%M')} BJT；青色十字为站点，原生64×64图。\n底部GHI为原检查点的三种子预测均值，阴影为种子范围；云图为目标时刻，GHI为该时刻结束的15min均值。\n未来观测仅用于事后诊断；R/CPP非独立真值；案例按诊断目的选取，不代表整体频率。",ha='center',fontsize=9)
    fig.subplots_adjust(left=.08,right=.915,bottom=.10,top=.97)
    save(fig,f'case_{seq}_real_cloud_cot_ghi')

# Station-crossing space-time slices. x is fixed image-column offset, not a tracked cloud object.
fig,axes=plt.subplots(len(cases),3,figsize=(13,10),sharex=True,sharey=True)
for row,c in enumerate(cases):
    z=np.load(D/f"case_{c['sequence']}.npz")
    maps=[np.concatenate((z['oracle_history'],z['oracle_cot'])),np.concatenate((z['history'],np.where(z['transport_support'],z['transport'],np.nan))),np.concatenate((z['history'],np.where(z['transport_support'],z['hybrid'],np.nan)))]
    for col,v in enumerate(maps):
        ax=axes[row,col];ax.set_facecolor('#ddd');im=ax.imshow(v[:,32,:],aspect='auto',origin='lower',extent=(-32.5,31.5,-112.5,247.5),cmap='magma',vmin=0,vmax=70,interpolation='nearest')
        ax.axvline(0,color='cyan',ls='--',lw=1);ax.axhline(0,color='white',ls=':',lw=1);ax.set_yticks([0,60,120,180,240])
        if col==0:ax.set_ylabel(c['init'][:16].replace('T',' ')+'\n相对初始化时间（分钟）')
        if row==len(cases)-1:ax.set_xlabel(['观测图 → R（事后诊断）','历史 COT 输运','混合修正 COT'][col]+'\n相对站点列偏移（像元）')
fig.subplots_adjust(left=.12,right=.91,bottom=.09,top=.98,hspace=.12,wspace=.08);ca=fig.add_axes([.93,.25,.013,.5]);fig.colorbar(im,cax=ca,label='检索/模型 COT',extend='max')
fig.text(.5,.015,'沿站点所在行的时空剖面；青线为站点，白线为初始化时刻，灰色为输运无来源；未来观测仅用于事后诊断，R非独立真值；这不是独立云团追踪。',ha='center',fontsize=9)
save(fig,'station_crossing_space_time')

fig,axes=plt.subplots(1,3,figsize=(15,4.7),gridspec_kw={'width_ratios':[1.1,1,1.1]})
for arm in colors:
    cs=[json.loads((R/f'{arm}_seed{s}/curve.json').read_text()) for s in (42,43,44)]
    step=[v['step'] for v in cs[0]];v=np.array([[d['rmse'] for d in c] for c in cs]);tr=1000*np.sqrt(np.array([[d['loss'] for d in c] for c in cs]))
    axes[0].plot(step,v.mean(0),color=colors[arm],label=labels[arm]+'：验证');axes[0].plot(step,tr.mean(0),':',color=colors[arm],label=labels[arm]+'：训练批次')
    earlycurves=[json.loads((D/'early_replay_attempt1'/f'{arm}_seed{s}/curve.json').read_text()) for s in (42,43,44)]
    et=[x['step'] for x in earlycurves[0]];vv=np.array([[x['rmse'] for x in c] for c in earlycurves]);axes[1].plot(et,vv.mean(0),'-o',color=colors[arm],ms=3,label=labels[arm]);axes[1].fill_between(et,vv.min(0),vv.max(0),alpha=.1,color=colors[arm])
axes[0].set_xlabel('原完整训练：更新步数');axes[0].set_ylabel('GHI RMSE（W/m²）');axes[0].legend(fontsize=8,frameon=False)
axes[1].set_xlabel('原样重放：前100次更新');axes[1].set_ylabel('验证 GHI RMSE（W/m²）');axes[1].legend(frameon=False,fontsize=9)
original=json.loads((R/'independent_score.json').read_text())['results'];diag=json.loads((D/'early_selection_diagnostic.json').read_text())['results']
for y,(arm,new) in enumerate([('transport',False),('transport',True),('hybrid',False),('hybrid',True)]):
    v=diag[arm] if new else original[arm]['0'];point=v['delta'] if new else v['AC_minus_A'];ci=v['ci95'] if new else v['delta_ci95']
    axes[2].errorbar(point,y,xerr=[[point-ci[0]],[ci[1]-point]],fmt='s' if new else 'o',color=colors[arm],capsize=3);axes[2].text(point,y+.19,f'{point:+.2f}',ha='center',fontsize=9,zorder=10,bbox=dict(facecolor='white',edgecolor='none',pad=1))
axes[2].axvline(0,color='#888',lw=1);axes[2].set_yticks(range(4),['输运：原检查点','输运：补早期检查','混合：原检查点','混合：补早期检查']);axes[2].invert_yaxis();axes[2].set_xlabel('加 COT − 无 COT：RMSE 差\n（W/m²；负值表示改善）')
for ax in axes:ax.grid(alpha=.2)
fig.subplots_adjust(left=.06,right=.99,bottom=.24,top=.97,wspace=.5)
fig.text(.5,.025,'4041条训练记录，batch8192；原每100步验证一次，间隔约203遍数据处理量。重放第100步九组RMSE均复现（差0）。\n左图虚线为当步训练batch RMSE，中图阴影为三种子范围。\n右图为同一验证集的事后选模诊断，误差线为日期配对95%区间；两组区间仍跨零，不能确认独立增益。',ha='center',fontsize=10)
save(fig,'training_and_early_checkpoint_diagnosis')
print('FIGURES_COMPLETE')
