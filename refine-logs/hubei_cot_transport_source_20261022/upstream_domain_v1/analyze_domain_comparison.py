"""FP64 paired rescore, date-block uncertainty, coverage and actual trajectories."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args();out=a.directory
    rows=list(csv.DictReader(open(out/'paired_pixels.csv')))
    names=['persistence','domain16','domain64','domain256'];sites=['Niushou','Wangzhai','Mizhuang']
    y=np.array([float(r['reference']) for r in rows],np.float64)
    p=np.array([[float(r[n]) for n in names] for r in rows],np.float64)
    day=np.array([int(r['day']) for r in rows],bool)
    center=np.array([r['dr']=='0' and r['dc']=='0' for r in rows])
    st=np.array([r['station'] for r in rows]);lead=np.array([int(r['lead_min']) for r in rows]);dates=np.array([r['date'] for r in rows])
    valid=day&np.isfinite(y); common=valid&np.all(np.isfinite(p),axis=1)
    maxdiff=float(np.max(np.ptp(p[common,1:],axis=1))) if common.any() else None
    assert maxdiff is None or maxdiff<1e-6, 'Changing domain changed common-supported transport values'
    fb=np.where(np.isfinite(p),p,p[:,0,None]);paired=valid&np.isfinite(p[:,0])
    def metric(mask,j):
        if not mask.any():return {'n':0}
        e=fb[mask,j]-y[mask];thick=mask&(y>=30)
        return {'n':int(mask.sum()),'dates':len(set(dates[mask])),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(abs(e))),'bias':float(np.mean(e)),
            'thick_n':int(thick.sum()),'thick_miss_rate':float(np.mean(fb[thick,j]<30)) if thick.any() else None}
    report={'scope':'CPP-history diagnostic; exploratory validation; no frozen-R or GHI gain',
        'max_transport_difference_on_common_support':maxdiff,'test_payloads_read':0,'buffer_payloads_read':0,'views':{}}
    for view,v in [('station_pixel',center),('neighborhood3',np.ones(len(rows),bool))]:
        rec={}
        for si in sites:
            mask=valid&v&(st==si);full=paired&v&(st==si)
            rec[si]={'reference_n':int(mask.sum()),'coverage':{n:float(np.mean(np.isfinite(p[mask,j]))) if mask.any() else None for j,n in enumerate(names)},
                'fallback_metrics':{n:metric(full,j) for j,n in enumerate(names)},
                'recovered_vs16':int((mask&~np.isfinite(p[:,1])&np.isfinite(p[:,3])).sum())}
        report['views'][view]=rec
    # Date block resampling of paired squared errors. Equal site, then equal lead.
    ds=sorted(set(dates[paired&center]));ss=np.zeros((len(ds),3,16,4));nn=np.zeros((len(ds),3,16),int)
    for di,d in enumerate(ds):
        for si,s in enumerate(sites):
            for li,l in enumerate(range(15,241,15)):
                m=paired&center&(dates==d)&(st==s)&(lead==l)
                nn[di,si,li]=m.sum();ss[di,si,li]=np.sum((fb[m]-y[m,None])**2,axis=0)
    def score(idx):
        count=nn[idx].sum(0);sums=ss[idx].sum(0)
        with np.errstate(divide='ignore',invalid='ignore'):
            return np.nanmean(np.sqrt(sums/np.where(count[...,None]>0,count[...,None],np.nan)),axis=(0,1))
    point=score(np.arange(len(ds))) if ds else np.full(4,np.nan)
    rng=np.random.default_rng(20261022)
    bs=np.array([score(rng.integers(0,len(ds),len(ds))) for _ in range(2000)]) if ds else np.zeros((0,4))
    report['equal_station_equal_lead_rmse']=dict(zip(names,map(float,point)))
    report['paired_date_block']={'dates':ds,'replicates':2000,'domain256_minus16_rmse_ci95':np.nanpercentile(bs[:,3]-bs[:,1],[2.5,97.5]).tolist() if len(bs) else None,
        'interpretation':'Same exploratory dates; missing source replaced by persistence; no independent confirmation'}
    arrivals=[]
    for init in sorted(set(r['init_bjt'] for r in rows)):
        for s in sites:
            m=np.array([r['init_bjt']==init for r in rows])&center&(st==s)&paired
            ix=np.where(m)[0];ix=ix[np.argsort(lead[ix])]
            if len(ix)!=16:continue
            ref=y[ix];entry={'init':init,'station':s,'reference_has_arrival':False}
            if ref[0]<30 and (ref>=30).any():
                obs=lead[ix][np.where(ref>=30)[0][0]];entry.update(reference_has_arrival=True,reference_arrival_minutes=int(obs))
                for j,n in enumerate(names):
                    vals=fb[ix,j];hits=np.where(vals>=30)[0]
                    entry[n+'_arrival_error_minutes']=int(lead[ix][hits[0]]-obs) if vals[0]<30 and len(hits) else None
            arrivals.append(entry)
    report['arrival']={'threshold':30,'complete_sequences':len(arrivals),'reference_arrivals':sum(x['reference_has_arrival'] for x in arrivals),'records':arrivals}
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    colors=['#777777','#e69f00','#009e73','#0072b2'];fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for j,n in enumerate(names[1:],1):
        cov=[];err=[]
        for l in range(15,241,15):
            m=valid&center&(lead==l);cov.append(np.mean(np.isfinite(p[m,j]))*100 if m.any() else np.nan)
            m=paired&center&(lead==l);err.append(np.sqrt(np.mean((fb[m,j]-y[m])**2)) if m.any() else np.nan)
        axes[0].plot(np.arange(15,241,15)/60,cov,label=n.replace('domain','')+' × '+n.replace('domain',''),color=colors[j])
        axes[1].plot(np.arange(15,241,15)/60,err,label=n.replace('domain','')+' × '+n.replace('domain',''),color=colors[j])
    axes[0].set(ylabel='Valid upstream source coverage (%)',xlabel='Forecast lead (h)',ylim=(0,102))
    axes[1].set(ylabel='COT RMSE (persistence fallback)',xlabel='Forecast lead (h)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.suptitle('Hubei: historical CPP transport source-domain diagnostic')
    fig.savefig(out/'source_domain_comparison.png');fig.savefig(out/'source_domain_comparison.pdf');plt.close(fig)
    # Choose an explanatory case by recovered-source count, not favorable error.
    candidates=[]
    for path in sorted(out.glob('20*.npz')):
        z=np.load(path);pred=z['prediction'];ref=z['reference'];dm=z['day_mask']
        recovered=np.isfinite(ref)&dm&~np.isfinite(pred[:,1])&np.isfinite(pred[:,3])
        candidates.append((int(recovered.sum()),path))
    if candidates:
        _,case=max(candidates,key=lambda x:(x[0],x[1].name))
        z=np.load(case);motion=z['motion'];base=z['base_cpp'];pred=z['prediction'];ref=z['reference'];dm=z['day_mask']
        from matplotlib.patches import Rectangle
        fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained')
        im=ax[0,0].imshow(base,origin='upper',vmin=0,vmax=60,cmap='magma',extent=(-.5,255.5,255.5,-.5))
        ax[0,0].set(xlim=(60,175),ylim=(165,55),xlabel='Column in verified AGRI grid',ylabel='Row',title='Actual historical CPP COT + upstream locations')
        for r0,c0,h,color,label in [(103,118,16,colors[1],'16 × 16'),(79,94,64,colors[2],'64 × 64')]:
            ax[0,0].add_patch(Rectangle((c0-.5,r0-.5),h,h,fill=False,edgecolor=color,lw=1.7,label=label))
        for s,(r,c) in zip(sites,[(111,124),(112,126),(110,128)]):
            ax[0,0].scatter(c,r,marker='*',color='white',edgecolors='black',s=70)
            sy=r-np.arange(1,17)*motion[0];sx=c-np.arange(1,17)*motion[1]
            ax[0,0].plot(sx,sy,color='cyan',lw=1,alpha=.7)
        ax[0,0].legend(frameon=False,fontsize=9);fig.colorbar(im,ax=ax[0,0],label='CPP COT')
        for si,s in enumerate(sites):
            aa=[ax[0,1],ax[1,0],ax[1,1]][si]
            yt=np.where(dm[:,si,1,1],ref[:,si,1,1],np.nan)
            pp=pred[:,:,si,1,1];fallback=np.where(np.isfinite(pp),pp,pp[:,0,None])
            aa.plot(np.arange(1,17)/4,yt,color='black',marker='o',ms=3,label='Future CPP reference')
            for j in [1,3]:aa.plot(np.arange(1,17)/4,fallback[:,j],color=colors[j],label=f'{[0,16,64,256][j]} × {[0,16,64,256][j]} + fallback')
            aa.set(title=s,xlabel='Forecast lead (h)',ylabel='Station COT');aa.grid(alpha=.2);aa.legend(frameon=False,fontsize=9)
        fig.suptitle(f'Actual case: {case.stem} BJT; selected by recovered-source count')
        fig.savefig(out/'actual_cpp_source_and_trajectories.png');fig.savefig(out/'actual_cpp_source_and_trajectories.pdf');plt.close(fig)
        report['illustrative_case']={'init_bjt':case.stem,'selection':'largest recovered-source count; tie latest date; no error-based selection'}
    # Independent saved-array (rather than CSV) FP64 station rescore.
    ee={s:[[],[],[],[]] for s in sites}
    for path in sorted(out.glob('20*.npz')):
        z=np.load(path);yy=z['reference'][:,:,1,1].astype(np.float64);pp=z['prediction'][:,:,:,1,1].astype(np.float64)
        for si,s in enumerate(sites):
            vv=z['day_mask'][:,si,1,1]&np.isfinite(yy[:,si])&np.isfinite(pp[:,0,si])
            for j in range(4):
                predj=np.where(np.isfinite(pp[:,j,si]),pp[:,j,si],pp[:,0,si])
                ee[s][j].extend((predj[vv]-yy[vv,si]).tolist())
    independent={s:{n:float(np.sqrt(np.mean(np.square(ee[s][j])))) for j,n in enumerate(names)} for s in sites}
    maxerr=max(abs(independent[s][n]-report['views']['station_pixel'][s]['fallback_metrics'][n]['rmse']) for s in sites for n in names)
    assert maxerr<1e-10
    report['independent_npz_vs_csv_rmse_max_abs_difference']=maxerr
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# 湖北历史CPP来源范围对照结果','', '**范围：验证集诊断，历史CPP输入；不是冻结R或GHI结果。**','',
        f'共同有效位置输运值的最大差：{maxdiff}（应为0）。','',
        '|站点|16覆盖率|64覆盖率|256覆盖率|持平RMSE|16+持平RMSE|256+持平RMSE|','|---|---:|---:|---:|---:|---:|---:|']
    for s in sites:
        r=report['views']['station_pixel'][s];co=r['coverage'];me=r['fallback_metrics']
        if co['domain16'] is None:continue
        lines.append(f"|{s}|{co['domain16']:.1%}|{co['domain64']:.1%}|{co['domain256']:.1%}|{me['persistence'].get('rmse',float('nan')):.3f}|{me['domain16'].get('rmse',float('nan')):.3f}|{me['domain256'].get('rmse',float('nan')):.3f}|")
    lines.extend(['',f"等站点等时效RMSE：{report['equal_station_equal_lead_rmse']}",'',f"256相对16的RMSE差、日期块95%区间：{report['paired_date_block']['domain256_minus16_rmse_ci95']}",'',
        'CPP为检索参考；历史CPP本身与业务冻结R的来源和误差不同。范围扩大仅恢复源支持，不改变共同位置的预测。未读取test/buffer；未训练网络。',
        '样本在CPP读取前按每个验证日期10:00固定选择；缺测日期/时刻见source_inventory.json和completion.json，不以晴空填缺测。',
        '太阳高度>10°筛选；评分仅用参考有效且历史持平有效的共同像元。数值有效性不代表完整产品质量控制。'])
    lines.extend(['',f"站点像元评分覆盖{len(ds)}个目标日期；每站有效记录数为{[report['views']['station_pixel'][s]['reference_n'] for s in sites]}，各时效参考支持数量不同，逐时效曲线不能解释成同一批事件随时间的退化。",
        f"完整16时效有效的站点序列{len(arrivals)}条，其中COT=30阈值的新增厚云到达事件{report['arrival']['reference_arrivals']}条；若为0，本轮不能评价到达时间改善。",
        f"NPZ独立复算与CSV评分RMSE最大差{maxerr}。",
        '结果判断：较大来源域是否恢复有效上游与是否降低误差是两个问题；区间跨零或部分站点退化时，不据此启动昂贵训练。'])
    (out/'RESULT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report['equal_station_equal_lead_rmse']),flush=True)
if __name__=='__main__':main()
