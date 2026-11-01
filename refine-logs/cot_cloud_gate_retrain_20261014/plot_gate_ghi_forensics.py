"""Paired GHI error decomposition, CPP-cloud restoration diagnostic, and observed-image figures."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, BoundaryNorm

def rmse(p,y): return float(np.sqrt(np.mean((p-y)**2)))
def score(z,m):
    y=z['ghi'][m]; p=z['raw_saved'][m]; q=z['gate_saved'][m]
    if not len(y): return dict(n=0)
    return dict(n=int(m.sum()),dates=len(np.unique((z['utc'][m].astype('datetime64[s]')+np.timedelta64(8,'h')).astype('datetime64[D]'))),
        raw_rmse=rmse(p,y),gate_rmse=rmse(q,y),delta_rmse=rmse(q,y)-rmse(p,y),
        delta_sse=float(((q-y)**2-(p-y)**2).sum()),worse_fraction=float((np.abs(q-y)>np.abs(p-y)).mean()))

def boot(p,q,y,dates):
    ud=np.unique(dates); rng=np.random.default_rng(20261015)
    inds=[np.where(dates==d)[0] for d in ud]; n=np.array([len(i) for i in inds])
    s1=np.array([((p[i]-y[i])**2).sum() for i in inds]); s2=np.array([((q[i]-y[i])**2).sum() for i in inds])
    draws=rng.integers(0,len(ud),size=(2000,len(ud))); ns=n[draws].sum(1)
    d=np.sqrt(s2[draws].sum(1)/ns)-np.sqrt(s1[draws].sum(1)/ns)
    return [float(v) for v in np.quantile(d,[.025,.975])]

def format_time(t): return str(t).replace('T',' ')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);a=ap.parse_args()
    out=a.data; z=dict(np.load(out/'validation_forensics.npz',allow_pickle=False)); figs=out/'figures';figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10,'pdf.fonttype':42})
    times=z['utc'].astype('datetime64[s]')+np.timedelta64(8,'h'); dates=times.astype('datetime64[D]'); s=z['station']; y=z['ghi']; p=z['raw_saved'];q=z['gate_saved']
    false3=z['falseclear'][:,7:10,7:10].sum((1,2)); false16=z['falseclear'].sum((1,2)); change3=z['cleared'][:,7:10,7:10].sum((1,2))
    supported=z['known'][:,7:10,7:10].all((1,2)); matched=z['refidx']>=0
    ds=(q-y)**2-(p-y)**2; cpug=z['gatehead_gateinput']; res=z['gatehead_restore_falsecloud']; allres=z['gatehead_rawinput']
    station_reports={}; cases=[]
    for st in ('sili','zhujia'):
        sm=s==st
        groups={name:score(z,sm&m) for name,m in {
            'all':np.ones(len(y),bool),'CPP_joined':matched,'confirmed_falsecloud_cleared_3x3':false3>0,
            'complete_CPP_no_falseclear_3x3':supported&(false3==0),'no_local_COT_change':change3==0,
            'without_CPP_reference':~matched}.items()}
        interventions={}
        for name,m in [('all_validation',sm),('CPP_joined',sm&matched),('confirmed_falseclear_3x3',sm&(false3>0))]:
            if not m.any():continue
            interventions[name]=dict(n=int(m.sum()),gate_cpu_rmse=rmse(cpug[m],y[m]),
                restore_CPP_cloud_rmse=rmse(res[m],y[m]),delta_restore_rmse=rmse(res[m],y[m])-rmse(cpug[m],y[m]),
                restore_all_original_R_rmse=rmse(allres[m],y[m]),delta_restore_all_rmse=rmse(allres[m],y[m])-rmse(cpug[m],y[m]),
                restore_CPPclear_rmse=rmse(z['gatehead_restore_CPPclear'][m],y[m]),
                delta_restore_CPPclear_rmse=rmse(z['gatehead_restore_CPPclear'][m],y[m])-rmse(cpug[m],y[m]),
                restore_CPPknown_rmse=rmse(z['gatehead_restore_CPPknown'][m],y[m]),
                restoration_delta_sse=float(((res[m]-y[m])**2-(cpug[m]-y[m])**2).sum()),
                restoration_mean_abs_prediction_change=float(np.abs(res[m]-cpug[m]).mean()),
                restoration_ci95_date=boot(cpug[m],res[m],y[m],dates[m]))
        station_reports[st]=dict(groups=groups,counterfactuals=interventions,
            confirmed_falseclear_pixels_3x3=int(false3[sm].sum()),confirmed_falseclear_pixels_16x16=int(false16[sm].sum()),
            confirmed_falseclear_ge01_pixels3=int((z['falseclear'][sm,7:10,7:10]&(np.expm1(z['raw16_log'][sm,7:10,7:10])>=.1)).sum()),
            paired_delta_rmse_ci95_date=boot(p[sm],q[sm],y[sm],dates[sm]),
            share_net_delta_sse_falseclear3=float(ds[sm&(false3>0)].sum()/ds[sm].sum()),
            share_net_delta_sse_no_local_change=float(ds[sm&(change3==0)].sum()/ds[sm].sum()))
        cand=np.where(sm&(false3>0))[0]
        if len(cand): cases.append(dict(index=int(cand[np.argmax(ds[cand])]),category='worst_confirmed_falseclear'))
        if len(cand): cases.append(dict(index=int(cand[np.argmin(ds[cand])]),category='improvement_despite_falseclear'))
        cand=np.where(sm&supported&(change3==0))[0]
        if len(cand): cases.append(dict(index=int(cand[np.argmax(ds[cand])]),category='degradation_without_local_clearing'))
    fields=['station','target_time_bjt','target_time_utc','CPP_joined','CPP_3x3_complete','falseclear_pixels3','falseclear_pixels16','cleared_pixels3','ghi','raw_ghi','gate_ghi','delta_squared_error','gate_cpu','restore_CPP_cloud_cpu','restore_all_R_cpu']
    with (out/'paired_sample_diagnostics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i in range(len(y)):
            w.writerow(dict(zip(fields,[s[i],format_time(times[i]),z['utc'][i],int(matched[i]),int(supported[i]),int(false3[i]),int(false16[i]),int(change3[i]),y[i],p[i],q[i],ds[i],cpug[i],res[i],allres[i]])))
    for c in cases:
        i=c['index'];c.update(station=str(s[i]),target_time_bjt=format_time(times[i]),target_time_utc=str(z['utc'][i]),falseclear_pixels3=int(false3[i]),
            raw_ghi=float(p[i]),gate_ghi=float(q[i]),obs_ghi=float(y[i]),delta_squared_error=float(ds[i]),
            gate_cpu=float(cpug[i]),restore_falsecloud_cpu=float(res[i]),restore_all_original_R_cpu=float(allres[i]))
    report=dict(state='COMPLETE_PAIRED_DIAGNOSTIC',mode='observed target-time AGRI, seed42 validation only',stations=station_reports,cases=cases,
        limits=['CPP availability is partial; unmatched samples are UNKNOWN, not clear/no-false-clear.',
                'CPP CLP/COT are reference product categories; not independent atmospheric truth.',
                'Restoration uses original R values, not true physical COT; it cannot repair R underestimation.',
                'Changing inputs with frozen heads is a diagnostic, not a deployable skill estimate.',
                'Raw/gate heads also differ in fitted weights and train-only COT normalization.',
                'CPU intervention comparisons use same CPU run; saved GPU outputs define main paired result.'],
        test_payloads_read=0,bootstrap='2000 paired BJT target-date clusters; seed 20261015')
    (out/'diagnostic_summary.json').write_text(json.dumps(report,indent=2)+'\n')

    # Overview preserves small effect sizes and shows partial reference support explicitly.
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,st in enumerate(('sili','zhujia')):
        m=s==st; names=['Saved raw head','Saved gate head','Fixed gate head','+ restore CPP-cloud','+ restore ALL R']
        vals=[rmse(p[m],y[m]),rmse(q[m],y[m]),rmse(cpug[m],y[m]),rmse(res[m],y[m]),rmse(allres[m],y[m])]
        ax=axs[0,col];ax.plot(vals,np.arange(5),'o',color='#0072B2');ax.set_yticks(np.arange(5),names);ax.invert_yaxis()
        for j,v in enumerate(vals):ax.annotate(f'{v:.4f}',(v,j),xytext=(5,5),textcoords='offset points',fontsize=9)
        ax.set_xlabel('GHI RMSE (W m$^{-2}$); point scale');ax.set_title(st.upper()+': tiny overall effect of restoring known clouds');ax.grid(axis='x',alpha=.25)
        ax=axs[1,col]
        sel=np.where(m&matched)[0]; ax.scatter(false3[sel],ds[sel],s=18,alpha=.65,color='#D55E00')
        ax.axhline(0,c='black',lw=.8);ax.set_xlabel('CPP-cloud pixels actually cleared in center 3x3');ax.set_ylabel('Gate SE - raw SE (W$^2$ m$^{-4}$)')
        ax.set_title(f'{st.upper()}: {int((m&matched).sum())}/{int(m.sum())} GHI samples have CPP')
    fig.suptitle('Actual saved validation + frozen-head input restoration\nCPP restoration is a diagnostic; test remains closed',fontsize=14)
    fig.savefig(figs/'overview.png',dpi=180);fig.savefig(figs/'overview.pdf');plt.close(fig)

    reflookup={(str(st),str(t)):i for i,(st,t) in enumerate(zip(z['ref_station'],z['ref_utc']))}
    val_lookup={(str(st),str(t)):i for i,(st,t) in enumerate(zip(s,z['utc']))}
    def mark(ax):
        ax.add_patch(Rectangle((6.5,6.5),3,3,fill=False,edgecolor='#00FFFF',lw=1.2,linestyle='--'))
        ax.plot(8,8,marker='*',ms=8,mec='black',mfc='white');ax.set_xticks([0,8,15]);ax.set_yticks([0,8,15])
    def line_with_gaps(ax,ids,values,**kw):
        # insert NaN at gaps > 15 minutes rather than connect missing observations.
        tt=times[ids];vv=np.asarray(values)[ids];xx=[];yy=[]
        for j,(t,v) in enumerate(zip(tt,vv)):
            if j and t-tt[j-1]>np.timedelta64(15,'m'):xx.append(t-np.timedelta64(1,'s'));yy.append(np.nan)
            xx.append(t);yy.append(v)
        ax.plot(np.array(xx),np.array(yy),**kw)
    with PdfPages(out/'cloud_images_and_GHI_timeseries.pdf') as pdf:
        for c in cases:
            i=c['index'];st=str(s[i]);t=times[i];ri=z['refidx'][i]
            fig=plt.figure(figsize=(20,12));gs=fig.add_gridspec(4,7,height_ratios=[1,1,1,1.45],left=.055,right=.955,bottom=.10,top=.86,wspace=.40,hspace=.34)
            focal=np.argwhere(z['falseclear'][i,7:10,7:10])+7
            if len(focal):
                fr,fc=focal[np.argmax([z['raw16_log'][i,r,c] for r,c in focal])]
            else:fr,fc=8,8
            c['focal_pixel_rc']=[int(fr),int(fc)]
            cotnorm=matplotlib.colors.Normalize(0,60)
            phasecmap=ListedColormap(['#F0F0F0','#56B4E9','#0072B2']);phasecmap.set_bad('#777777')
            images={}; mapaxes={k:[] for k in range(7)}
            for row,offset in enumerate((-30,0,30)):
                utc=t-np.timedelta64(8,'h')+np.timedelta64(offset,'m');key=(st,format_time(utc))
                vj=val_lookup.get(key);rj=reflookup.get(key)
                for col in range(7):
                    ax=fig.add_subplot(gs[row,col])
                    if rj is None or vj is None:
                        ax.text(.5,.5,'No matched CPP/AGRI\n(no interpolation)',ha='center',va='center',transform=ax.transAxes);ax.axis('off');continue
                    rx=z['real_x'][rj];rv=z['ref_valid'][rj]
                    if col==0:
                        # False colour uses actual measured channels, fixed reflectance stretch.
                        rgb=np.stack([rx[2],rx[1],rx[0]],-1);rgb=np.clip(rgb/0.8,0,1)**(1/2.2)
                        im=ax.imshow(rgb,interpolation='nearest');title='Observed AGRI false colour\nR=C03, G=C02, B=C01'
                        ax.set_ylabel(format_time(times[vj])+' BJT',fontsize=9)
                    elif col==1:
                        im=ax.imshow(np.ma.masked_where(~rv,z['ref_cot'][rj]),norm=cotnorm,cmap='magma',interpolation='nearest');title='CPP reference COT';images['cot']=im
                    elif col==2:
                        im=ax.imshow(np.ma.masked_where(~rv,z['ref_clp'][rj]),cmap=phasecmap,norm=BoundaryNorm([-.5,.5,1.5,2.5],3),interpolation='nearest');title='CPP CLP\n0=clear; 1/2=cloud';images['clp']=im
                    elif col==3:
                        im=ax.imshow(np.expm1(z['raw16_log'][vj]),norm=cotnorm,cmap='magma',interpolation='nearest');title='R: original COT'
                    elif col==4:
                        im=ax.imshow(np.expm1(z['gate16_log'][vj]),norm=cotnorm,cmap='magma',interpolation='nearest');title='R: gated COT'
                    elif col==5:
                        im=ax.imshow(z['prob16'][vj],vmin=0,vmax=1,cmap='viridis',interpolation='nearest');title='P(cloud)\ncut=0.2935';images['prob']=im
                    else:
                        state=np.zeros((16,16),float);state[z['cleared'][vj]&z['known'][vj]]=1;state[z['falseclear'][vj]]=2;state[~z['known'][vj]]=np.nan
                        cm=ListedColormap(['#E5E5E5','#56B4E9','#D55E00']);cm.set_bad('#777777')
                        im=ax.imshow(state,cmap=cm,norm=BoundaryNorm([-.5,.5,1.5,2.5],3),interpolation='nearest');title='Cleared pixels\norange=cloud; blue=clear'
                    mapaxes[col].append(ax);mark(ax);ax.add_patch(Rectangle((fc-.5,fr-.5),1,1,fill=False,edgecolor='#FF00FF',lw=1.2))
                    if row==0:ax.set_title(title,fontsize=9)
            if 'cot' in images:fig.colorbar(images['cot'],cax=fig.add_axes([.966,.43,.008,.36]),label='COT (dimensionless); common scale',extend='max')
            if 'prob' in images:fig.colorbar(images['prob'],cax=fig.add_axes([.966,.32,.008,.06]),ticks=[0,1],label='P(cloud)')
            ax=fig.add_subplot(gs[3,:4]);window=(s==st)&(dates==dates[i])&(np.abs(times-t)<=np.timedelta64(120,'m'));jj=np.where(window)[0];jj=jj[np.argsort(times[jj])]
            for vals,label,color,style in [(y,'Observed 15-min GHI','black','-'),(p,'Raw-COT head','#0072B2','-'),(q,'Gate-COT head','#D55E00','--'),(res,'Fixed gate head + restore CPP-cloud','#009E73',':')]:
                line_with_gaps(ax,jj,vals,label=label,color=color,linestyle=style,marker='.',ms=4)
            ax.axvline(t,c='#CC79A7',lw=1);ax.set_ylabel('GHI (W m$^{-2}$)');ax.set_xlabel('Target interval END time (BJT); gaps retained');ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'));ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2)
            ax=fig.add_subplot(gs[3,4:]);rawcenter=np.expm1(z['raw16_log'][:,fr,fc]);gatecenter=np.expm1(z['gate16_log'][:,fr,fc]);cppcenter=np.full(len(y),np.nan)
            for j in jj:
                rj=z['refidx'][j]
                if rj>=0 and z['ref_valid'][rj,fr,fc]:cppcenter[j]=z['ref_cot'][rj,fr,fc]
            for vals,label,color,style in [(rawcenter,'Original R at focal pixel','#0072B2','-'),(gatecenter,'Gated R at focal pixel','#D55E00','--'),(cppcenter,'CPP at focal pixel (reference)','black',':')]:
                line_with_gaps(ax,jj,vals,label=label,color=color,linestyle=style,marker='.',ms=4)
            ax.axvline(t,c='#CC79A7',lw=1);ax.set_ylabel(f'Focal pixel ({fr},{fc}) COT');ax.set_xlabel('Target time (BJT)');ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'));ax.legend(fontsize=8);ax.grid(alpha=.2)
            fig.suptitle(f"{st.upper()} | {format_time(t)} BJT | {c['category']}\nObserved {y[i]:.2f}; raw {p[i]:.2f}; gate {q[i]:.2f} W/m2 | center-3x3 false-cleared pixels={false3[i]}; restoration change={res[i]-cpug[i]:+.3f} W/m2",fontsize=13)
            fig.text(.5,.015,'Real observed 16x16 pixels; no synthetic clouds or spatial interpolation. Cyan box=center 3x3 read by GHI head; star=station (8,8); magenta cell=focal COT pixel.\nFixed RGB stretch: reflectance / 0.8, gamma 1/2.2. CPP/R COT share 0-60 colour scale (values >60 saturate); gray=invalid/unknown. Same-time validation, not future forecast.',ha='center',fontsize=8)
            name=f"{st}_{str(t).replace('-','').replace(':','').replace('T','_')}_{c['category']}"
            fig.savefig(figs/(name+'.png'),dpi=170);pdf.savefig(fig);plt.close(fig);c['figure']='figures/'+name+'.png'
    # Compact Chinese views for the two cloud-clearing-associated deterioration cases.
    from matplotlib.font_manager import FontProperties
    cn=FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
    with plt.rc_context({'font.family':cn,'axes.unicode_minus':False}):
        for c in cases:
            if c['category']!='worst_confirmed_falseclear':continue
            i=c['index'];ri=z['refidx'][i];fr,fc=c['focal_pixel_rc'];st=str(s[i]);t=times[i]
            fig=plt.figure(figsize=(13,8));gs=fig.add_gridspec(2,4,height_ratios=[1,1.1],left=.07,right=.93,bottom=.16,top=.80,wspace=.35,hspace=.45)
            rx=z['real_x'][ri];rgb=np.clip(np.stack([rx[2],rx[1],rx[0]],-1)/.8,0,1)**(1/2.2)
            for col,(arr,title) in enumerate([(rgb,'真实 AGRI 假彩色'),(np.ma.masked_where(~z['ref_valid'][ri],z['ref_cot'][ri]),'CPP 参考 COT'),(np.expm1(z['raw16_log'][i]),'实际原 R 输出'),(np.expm1(z['gate16_log'][i]),'实际门控后 COT')]):
                ax=fig.add_subplot(gs[0,col]);im=ax.imshow(arr,interpolation='nearest',**({} if col==0 else dict(cmap='magma',vmin=0,vmax=60)));mark(ax)
                ax.add_patch(Rectangle((fc-.5,fr-.5),1,1,fill=False,edgecolor='#FF00FF',lw=1.7));ax.set_title(title)
                if col==3:fig.colorbar(im,cax=fig.add_axes([.95,.56,.012,.22]),label='COT（无量纲）',extend='max')
            jj=np.where((s==st)&(dates==dates[i])&(np.abs(times-t)<=np.timedelta64(90,'m')))[0];jj=jj[np.argsort(times[jj])]
            ax=fig.add_subplot(gs[1,:2])
            for vals,label,color,style in [(y,'实测 GHI','black','-'),(p,'原 COT 训练头','#0072B2','-'),(q,'门控 COT 训练头','#D55E00','--'),(res,'固定门控头：仅恢复误清云','#009E73',':'),(z['gatehead_restore_CPPclear'],'固定门控头：仅恢复 CPP 晴空处 R 值','#CC79A7','-.')]:
                line_with_gaps(ax,jj,vals,label=label,color=color,linestyle=style,marker='.',ms=4)
            ax.axvline(t,c='gray',lw=.8);ax.set_ylabel('GHI（W/m²）');ax.set_xlabel('15 分钟区间结束时刻（北京时间）');ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'));ax.legend(fontsize=7);ax.grid(alpha=.2)
            ax=fig.add_subplot(gs[1,2:]);cpp=np.full(len(y),np.nan)
            for j in jj:
                rj=z['refidx'][j]
                if rj>=0 and z['ref_valid'][rj,fr,fc]:cpp[j]=z['ref_cot'][rj,fr,fc]
            for vals,label,color,style in [(np.expm1(z['raw16_log'][:,fr,fc]),'原 R','#0072B2','-'),(np.expm1(z['gate16_log'][:,fr,fc]),'门控后','#D55E00','--'),(cpp,'CPP 参考','black',':')]:
                line_with_gaps(ax,jj,vals,label=label,color=color,linestyle=style,marker='.',ms=4)
            ax.axvline(t,c='gray',lw=.8);ax.set_ylabel(f'紫色像元 ({fr},{fc}) 的 COT');ax.set_xlabel('目标时刻（北京时间）');ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'));ax.legend(fontsize=8);ax.grid(alpha=.2)
            nm='四里' if st=='sili' else '竺家'
            fig.suptitle(f'{nm} | {format_time(t)} 北京时间\nCPP 有云像元被清零，恢复后只能消除部分 GHI 偏差',fontsize=15,y=.98)
            fig.text(.5,.86,f"紫色像元：CPP COT={z['ref_cot'][ri,fr,fc]:.3f}，原 R={np.expm1(z['raw16_log'][i,fr,fc]):.3f}，门控后=0；\n固定门控头恢复误清云：{cpug[i]:.2f} → {res[i]:.2f} W/m²；实测={y[i]:.2f} W/m²",ha='center',fontsize=10)
            fig.text(.5,.025,'青色框是 GHI 实际读取的中心 3×3；白星是站点；紫框是右下时序对应的误清云像元。\nAGRI 为真实观测 C03/C02/C01 假彩色（固定拉伸），不是自然真彩色或合成云场；恢复 CPP 类别仅作诊断。\n这是同期真实 AGRI 验证，不是未来外推预报；仅单种子，test 未使用。',ha='center',fontsize=9)
            fig.savefig(figs/(st+'_focus_cn.png'),dpi=190);fig.savefig(figs/(st+'_focus_cn.pdf'));plt.close(fig)
    (out/'selected_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    (out/'diagnostic_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
