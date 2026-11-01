"""Actual native-pixel maps and complete saved COT trajectories; no synthetic fields."""
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.dates import DateFormatter
from audit_cot_event_timing import read_csv, csvout, sha, STATIONS

BASE=Path(__file__).resolve().parent
D=BASE/'results/dense_r_field_event_diagnostic_v1_20261021'
OUT=D/'figures_v1'
LABEL={'cot_p_xc':'Historical AGRI + COT → future COT',
       'cot_p_c':'Historical COT → future COT', 'cot_simvp_r':'SimVP AGRI → R',
       'observed_R':'Observed future AGRI → frozen R (offline reference)'}
STYLE={'observed_R':dict(color='black',marker='o',lw=2.0),
       'cot_simvp_r':dict(color='#0072B2',marker='s',ls='--',lw=1.5),
       'cot_p_xc':dict(color='#D55E00',marker='^',lw=1.7),
       'cot_p_c':dict(color='#009E73',marker='.',ls=':',lw=1.4)}

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=170)
    fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig)

def trajectory(ax,rows,title,event=None):
    rows=sorted(rows,key=lambda r:int(r['lead_minutes']))
    times=[datetime.fromisoformat(r['target_time_bjt']) for r in rows]
    for arm in ('observed_R','cot_simvp_r','cot_p_xc','cot_p_c'):
        ax.plot(times,[float(r[arm]) for r in rows],label=LABEL[arm],ms=3,**STYLE[arm])
    ax.axhline(30,color='.5',ls='--',lw=.8,label='Thick-cloud threshold = 30')
    if event:
        ax.axvspan(datetime.fromisoformat(event['start_time_bjt']),
                   datetime.fromisoformat(event['end_time_bjt']),color='.75',alpha=.25)
    ax.set_title(title,fontsize=10)
    ax.set_ylabel('Station-pixel COT (dimensionless)')
    ax.set_ylim(bottom=0);ax.grid(alpha=.2)
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.set_xlabel('Target time (BJT); each forecast lead is +15…+240 min')

def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    metrics=read_csv(D/'field_metrics.csv')
    source=[r for r in metrics if r['scale']=='station3' and r['station']!='both'
            and r['lead_minutes']!='all' and r['arm']!='cot_last']
    csvout(OUT/'lead_figure_source.csv',source)
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),sharex=True,layout='constrained')
    for col,st in enumerate(STATIONS):
        for arm in ('cot_simvp_r','cot_p_xc','cot_p_c'):
            rr=sorted([r for r in source if r['station']==st and r['arm']==arm],key=lambda r:int(r['lead_minutes']))
            xx=[int(r['lead_minutes']) for r in rr]
            for row,key in enumerate(('rmse','thick_csi')):
                axes[row,col].plot(xx,[float(r[key]) for r in rr],label=LABEL[arm],ms=3,**STYLE[arm])
        axes[0,col].set_title('Sili' if st=='sili' else 'Zhujia')
        axes[0,col].set_ylim(0,24);axes[1,col].set_ylim(0,1)
        axes[0,col].set_ylabel('COT RMSE vs learned R reference')
        axes[1,col].set_ylabel('Thick-cloud CSI vs learned R reference')
        axes[1,col].set_xlabel('Forecast lead from history end (min)')
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.set_xticks([15,60,120,180,240])
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=1,fontsize=9)
    fig.suptitle('Dense observed-image R reference: forecast field fidelity\n'
                 'Station 3×3; common QA / SOZ < 80° support; development validation only',fontsize=12)
    save(fig,'leadwise_cot_fidelity')

    timeline=read_csv(D/'actual_event_timelines.csv')
    selected=read_csv(D/'posthoc_daily_selection.csv')
    fig,axs=plt.subplots(2,1,figsize=(11,7.5),layout='constrained')
    for ax,r in zip(axs,selected):
        rr=[z for z in timeline if z['window_id']==r['window_id']]
        trajectory(ax,rr,f"{r['station'].title()} — {r['date_bjt']}; init {rr[0]['init_time_bjt']} BJT")
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',fontsize=9,ncol=2)
    fig.suptitle('Actual saved trajectories: largest daily-window peak error per station\n'
                 'Post-hoc diagnostic examples; observed-image R is not CPP truth',fontsize=12)
    save(fig,'daily_peak_error_examples')
    events=[r for r in read_csv(D/'unique_reference_events.csv') if r['status']=='EVALUABLE']
    with PdfPages(OUT/'all_27_reference_event_timelines.pdf') as pdf:
        for e in events:
            rr=[r for r in timeline if r['window_id']==e['canonical_window']]
            fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
            trajectory(ax,rr,f"{e['event_id']}\nInit {rr[0]['init_time_bjt']} BJT; shaded: reference event",e)
            ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1,1))
            pdf.savefig(fig);plt.close(fig)

    vis=read_csv(BASE/'results/initial_cloud_visibility_v1_20261021/event_input_visibility.csv')
    cases=[r for r in vis if r['arm']=='cot_p_xc' and r['detected']=='False'
           and r['initial_visibility']=='NEIGHBOR_THICK_CENTER_NOT']
    assert len(cases)==5
    csvout(OUT/'visible_neighbor_miss_cases.csv',cases)
    rd=BASE/'results/dense_observed_v_r_reference_v1_20261021'
    raw=BASE/'data/dense_observed_v_reference_patches_v1'
    lookup={r['target_time_utc']:int(r['frame_index']) for r in read_csv(rd/'rows.csv')}
    x=np.load(raw/'x_raw.npy',mmap_mode='r');cot=np.load(rd/'cot.npy')[:,:,0];mask=np.load(rd/'mask.npy')[:,:,0]
    inf=BASE/'data/cpp_score_inputs_v1/inference/V'
    forecasts={a:np.load(inf/(a+'.npy'),mmap_mode='r') for a in ('cot_last','cot_p_xc','cot_simvp_r')}
    cmap=plt.get_cmap('magma').copy();cmap.set_bad('#bdbdbd');cmap.set_over('#00c8c8')
    grey=plt.get_cmap('gray').copy();grey.set_bad('#bdbdbd')
    fig,axs=plt.subplots(5,6,figsize=(17,13),layout='constrained')
    titles=('Observed C03 at init','Retrieved COT at init','Observed C03 at event',
            'Observed AGRI → R\nat event (offline)','Direct future COT\nat event','SimVP AGRI → R\nat event')
    source_arrays={}
    for row,r in enumerate(cases):
        sti=STATIONS.index(r['station']);sq=int(r['sequence_index']);lead=int(r['start_lead_min']);li=lead//15-1
        init=datetime.fromisoformat(r['initial_time_utc']);target=init+timedelta(minutes=lead)
        ii=lookup[init.isoformat(' ')];ti=lookup[target.isoformat(' ')]
        initial=np.where(mask[ii,sti],forecasts['cot_last'][sq,sti,0,0],np.nan)
        observed=np.where(mask[ti,sti],cot[ti,sti],np.nan)
        mm=mask[ti,sti]
        fields=(x[ii,sti,2],initial,x[ti,sti,2],observed,
                np.where(mm,forecasts['cot_p_xc'][sq,sti,li,0],np.nan),
                np.where(mm,forecasts['cot_simvp_r'][sq,sti,li,0],np.nan))
        assert mask[ii,sti].all() and mask[ti,sti,8,8]
        for col,(ax,field) in enumerate(zip(axs[row],fields)):
            optical=col in (0,2)
            im=ax.imshow(field,cmap=grey if optical else cmap,vmin=0,vmax=1 if optical else 100,
                         interpolation='nearest',origin='upper')
            if optical:optical_image=im
            else:cot_image=im
            ax.plot(8,8,'*',color='white',markeredgecolor='black',ms=9)
            ax.add_patch(Rectangle((6.5,6.5),3,3,fill=False,color='white',ls='--',lw=.7))
            ax.set_xticks([0,8,15]);ax.set_yticks([0,8,15]);ax.tick_params(labelsize=7)
            if row==0:ax.set_title(titles[col],fontsize=10)
            if col==0:
                ax.set_ylabel(f"{r['station'].title()} {target+timedelta(hours=8):%Y-%m-%d}\n"
                              f"init {init+timedelta(hours=8):%H:%M} → {target+timedelta(hours=8):%H:%M} BJT\n+{lead} min",fontsize=9)
            if not optical:ax.set_xlabel(f"station COT = {field[8,8]:.1f}",fontsize=8)
            source_arrays[f'case{row}_column{col}']=np.asarray(field)
    fig.colorbar(optical_image,ax=axs[:,[0,2]].ravel().tolist(),shrink=.35,pad=.01,label='Raw C03 reflectance (0–1)')
    fig.colorbar(cot_image,ax=axs[:,[1,3,4,5]].ravel().tolist(),shrink=.35,pad=.01,extend='max',label='COT (dimensionless); gray = invalid')
    fig.suptitle('All five direct-forecast misses with a thick neighbor visible at initialization\n'
                 'Native 16×16 pixels; star = station (8,8), box = 3×3. Same target time across future columns.\n'
                 'Reference events: station COT ≥ 30 for ≥2 consecutive frames; post-hoc mechanism illustration only.',fontsize=12)
    np.savez_compressed(OUT/'visible_neighbor_maps_source.npz',**source_arrays)
    save(fig,'all_visible_neighbor_misses')
    info={'state':'COMPLETE_DENSE_DIAGNOSTIC_FIGURES','native_map_shape':[16,16],
          'image_interpolation':'nearest','cot_limits':[0,100],'c03_limits':[0,1],
          'new_model_training':False,'synthetic_fields':False,
          'reference':'learned frozen R from observed future AGRI; offline diagnostic only',
          'event_timeline_pages':len(events),'visible_neighbor_miss_rows':len(cases),
          'script_sha256':sha(Path(__file__)),
          'source_sha256':{str(p.relative_to(BASE)):sha(p) for p in
              [D/'field_metrics.csv',D/'actual_event_timelines.csv',D/'posthoc_daily_selection.csv',
               D/'unique_reference_events.csv',BASE/'results/initial_cloud_visibility_v1_20261021/event_input_visibility.csv',rd/'complete.json']}}
    (OUT/'figure_provenance.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
