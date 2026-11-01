"""Actual 8-frame histories and 64x64 context for all five fixed misses."""
import json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from audit_cot_event_timing import read_csv,csvout,sha,STATIONS

BASE=Path(__file__).resolve().parent

def main():
    source=BASE/'data/visible_neighbor_context_v1_20261021'
    rec=json.loads((source/'complete.json').read_text())
    assert rec['state']=='COMPLETE_FIVE_CASE_OBSERVED_CONTEXT'
    for n,h in rec['artifact_sha256'].items():assert sha(source/n)==h,n
    cases=read_csv(source/'cases.csv');frames=read_csv(source/'frames.csv')
    context=np.load(source/'channels.npy');valid=np.load(source/'channel_valid.npy')
    rdir=BASE/'results/dense_observed_v_r_reference_v1_20261021'
    refs=np.load(rdir/'cot.npy')[:,:,0];rm=np.load(rdir/'mask.npy')[:,:,0]
    lookup={r['target_time_utc']:int(r['frame_index']) for r in read_csv(rdir/'rows.csv')}
    raw=np.load(BASE/'data/dense_observed_v_reference_patches_v1/x_raw.npy',mmap_mode='r')
    pred={a:np.load(BASE/'data/cpp_score_inputs_v1/inference/V'/(a+'.npy'),mmap_mode='r')
          for a in ('cot_p_xc','cot_simvp_r')}
    out=BASE/'results/visible_neighbor_history_context_diagnostic_v1_20261021'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    cotmap=plt.get_cmap('magma').copy();cotmap.set_bad('#bdbdbd');cotmap.set_over('#00c8c8')
    grey=plt.get_cmap('gray').copy();grey.set_bad('#bdbdbd');grey.set_over('#00c8c8')
    ir=plt.get_cmap('gray_r').copy();ir.set_bad('#bdbdbd');ir.set_over('#00c8c8');ir.set_under('#0072B2')
    anchors=[0,3,6,7,11,15,19,23]
    summaries=[];series=[];consistency=[]
    with PdfPages(out/'all_five_context_and_history.pdf') as pdf:
        for ci,case in enumerate(cases):
            sti=STATIONS.index(case['station']);sq=int(case['sequence_index'])
            rr=[r for r in frames if int(r['case_index'])==ci]
            assert [int(r['frame_index']) for r in rr]==list(range(24))
            times=[datetime.fromisoformat(r['time_bjt']) for r in rr]
            ix=[lookup[r['time_utc']] for r in rr]
            rfields=np.where(rm[ix,sti],refs[ix,sti],np.nan)
            # Native center16 must be identical to original dense reference input.
            maximum=0.;n=0
            for ti,ri in enumerate(ix):
                a=context[ci,ti,:,24:40,24:40];b=raw[ri,sti,[2,10]]
                m=np.isfinite(b)&valid[ci,ti,:,24:40,24:40]
                assert np.isfinite(a[m]).all()
                if m.any():maximum=max(maximum,float(np.abs(a[m]-b[m]).max()));n+=int(m.sum())
            assert maximum==0.
            consistency.append({'event_id':case['event_id'],'common_channel_values':n,'max_abs_context_vs_dense_center16':maximum})
            hist=rfields[:8,8,8];allvalid=rm[ix[:8],sti].all(axis=(1,2))
            summaries.append({'event_id':case['event_id'],'station':case['station'],'history_frames':8,
                'history_full_support_frames':int(allvalid.sum()),'history_center_valid_frames':int(np.isfinite(hist).sum()),
                'history_center_min':float(np.nanmin(hist)),'history_center_max':float(np.nanmax(hist)),
                'history_center_last':float(hist[-1]),'history_center_thick_frames':int((hist>=30).sum()),
                'initial_patch_max':float(np.nanmax(rfields[7])),
                'future_reference_peak':float(np.nanmax(rfields[8:,8,8])),
                'future_pxc_peak':float(pred['cot_p_xc'][sq,sti,:,0,8,8].max()),
                'future_simvp_r_peak':float(pred['cot_simvp_r'][sq,sti,:,0,8,8].max())})
            for ti,r in enumerate(rr):
                series.append({'event_id':case['event_id'],'frame_index':ti,'time_bjt':r['time_bjt'],
                    'lead_minutes':int(r['lead_minutes']),'reference_center_cot':float(rfields[ti,8,8]) if np.isfinite(rfields[ti,8,8]) else '',
                    'reference_valid_fraction':float(rm[ix[ti],sti].mean()),
                    'center_c13_K':float(context[ci,ti,1,32,32]) if valid[ci,ti,1,32,32] else '',
                    'p_xc':float(pred['cot_p_xc'][sq,sti,ti-8,0,8,8]) if ti>=8 else '',
                    'simvp_r':float(pred['cot_simvp_r'][sq,sti,ti-8,0,8,8]) if ti>=8 else ''})
            fig=plt.figure(figsize=(16,10),layout='constrained');gs=fig.add_gridspec(4,8,height_ratios=[1,1,1,.95])
            maps=np.empty((3,8),object)
            for j,ti in enumerate(anchors):
                for row in range(3):
                    ax=fig.add_subplot(gs[row,j]);maps[row,j]=ax
                    if row<2:
                        field=np.where(valid[ci,ti,row],context[ci,ti,row],np.nan)
                        im=ax.imshow(field,cmap=grey if row==0 else ir,vmin=0 if row==0 else 190,
                                     vmax=1 if row==0 else 320,interpolation='nearest')
                        if row==0:c03im=im
                        else:c13im=im
                        ax.plot(32,32,'*',ms=7,color='white',markeredgecolor='black')
                        ax.add_patch(Rectangle((23.5,23.5),16,16,fill=False,ec='#D55E00',lw=1))
                        ax.set_xticks([0,32,63]);ax.set_yticks([0,32,63])
                    else:
                        rimage=ax.imshow(rfields[ti],cmap=cotmap,vmin=0,vmax=100,interpolation='nearest')
                        ax.plot(8,8,'*',ms=7,color='white',markeredgecolor='black')
                        ax.set_xticks([0,8,15]);ax.set_yticks([0,8,15])
                        v=rfields[ti,8,8];ax.set_xlabel(f'station={v:.1f}' if np.isfinite(v) else 'station: unsupported',fontsize=7)
                    ax.tick_params(labelsize=6)
                    if row==0:ax.set_title(f"{int(rr[ti]['lead_minutes']):+d} min\n{times[ti]:%H:%M} BJT",fontsize=9)
                    if j==0:ax.set_ylabel(('C03, native 64×64','C13, native 64×64','Observed R, native 16×16')[row],fontsize=9)
            ax=fig.add_subplot(gs[3,:])
            rel=np.arange(-105,241,15)
            ax.plot(rel,rfields[:,8,8],'ko-',ms=3,label='Observed AGRI → R; offline reference')
            ax.plot(rel[8:],pred['cot_p_xc'][sq,sti,:,0,8,8],'^-',color='#D55E00',ms=3,label='Direct future COT')
            ax.plot(rel[8:],pred['cot_simvp_r'][sq,sti,:,0,8,8],'s--',color='#0072B2',ms=3,label='SimVP AGRI → R')
            ax.axhline(30,color='.5',lw=.8,ls='--');ax.axvline(0,color='black',ls=':',lw=1)
            ax.axvspan(-105,0,color='.8',alpha=.2)
            ax.set(xlabel='Time relative to initialization (min); gray shading = observed history',
                   ylabel='Station COT',ylim=(0,100),xticks=[-105,-60,0,60,120,180,240]);ax.grid(alpha=.2)
            ax.legend(loc='upper center',ncol=3,fontsize=8)
            fig.colorbar(c03im,ax=maps[0].tolist(),shrink=.7,pad=.01,extend='max',label='Reflectance')
            fig.colorbar(c13im,ax=maps[1].tolist(),shrink=.7,pad=.01,extend='both',label='Brightness temperature (K)')
            fig.colorbar(rimage,ax=maps[2].tolist(),shrink=.7,pad=.01,extend='max',label='COT')
            fig.suptitle(f"{case['event_id']} — init {times[7]:%Y-%m-%d %H:%M} BJT\n"
                         'Actual native pixels; orange box = original 16×16 input; gray = invalid/unsupported.\n'
                         'C13 shows cloud-top temperature, not cloud optical thickness; region64 is context only.',fontsize=12)
            fig.savefig(out/(case['event_id']+'_context.png'),dpi=160);pdf.savefig(fig);plt.close(fig)
            # All eight real history frames: retain unsupported geometry as gray in R.
            fig,axs=plt.subplots(3,8,figsize=(15,6),layout='constrained')
            for ti in range(8):
                for row in range(3):
                    ax=axs[row,ti]
                    field=(np.where(valid[ci,ti,row,24:40,24:40],context[ci,ti,row,24:40,24:40],np.nan)
                           if row<2 else rfields[ti])
                    ax.imshow(field,cmap=(grey,ir,cotmap)[row],vmin=(0,190,0)[row],vmax=(1,320,100)[row],interpolation='nearest')
                    ax.plot(8,8,'*',ms=7,color='white',markeredgecolor='black');ax.set_xticks([]);ax.set_yticks([])
                    if row==0:ax.set_title(f"{int(rr[ti]['lead_minutes']):+d} min\n{times[ti]:%H:%M}",fontsize=9)
                    if ti==0:ax.set_ylabel(('C03','C13','Observed R')[row])
            fig.suptitle(f"{case['event_id']}: all eight actual history frames (native 16×16)\n"
                         'Same scales as context page; gray R pixels = no valid QA/solar support.',fontsize=12)
            pdf.savefig(fig);plt.close(fig)
            fig,axs=plt.subplots(3,16,figsize=(18,5),layout='constrained')
            for li in range(16):
                for row,arm in enumerate(('reference','cot_p_xc','cot_simvp_r')):
                    field=rfields[li+8] if row==0 else np.where(rm[ix[li+8],sti],pred[arm][sq,sti,li,0],np.nan)
                    ax=axs[row,li];ax.imshow(field,cmap=cotmap,vmin=0,vmax=100,interpolation='nearest')
                    ax.plot(8,8,'*',ms=5,color='white',markeredgecolor='black');ax.set_xticks([]);ax.set_yticks([])
                    if row==0:ax.set_title(f'+{15*(li+1)}',fontsize=8)
                    if li==0:ax.set_ylabel(('Observed R','Direct COT','SimVP → R')[row],fontsize=9)
            fig.suptitle(f"{case['event_id']}: all 16 future frames, common target support\n"
                         'Native16; COT 0–100; first row offline observed reference, remaining rows causal forecasts.',fontsize=12)
            pdf.savefig(fig);plt.close(fig)
    csvout(out/'history_and_peak_summary.csv',summaries);csvout(out/'all_actual_station_timelines.csv',series)
    csvout(out/'center16_source_consistency.csv',consistency)
    result={'state':'COMPLETE_FIVE_CASE_HISTORY_CONTEXT_DIAGNOSTIC','cases':5,'pdf_pages':15,
        'source_receipt_sha256':sha(source/'complete.json'),'script_sha256':sha(Path(__file__)),
        'context_matches_old_16_inputs':all(r['max_abs_context_vs_dense_center16']==0 for r in consistency),
        'new_training':False,'ghi_gain_established':False,'all_5_cases_displayed':True,
        'scope':'post-hoc actual-image mechanism diagnostic only; not tracking proof or independent confirmation'}
    (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summaries,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
