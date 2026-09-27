"""Plot validation-only AGRI/COT/GHI event evidence for the Zhujia regional pilot.

Uses deployable forecast AGRI/COT for model inputs and real-future-AGRI->R only as
an oracle diagnostic. It does not access test data or claim CPP is physical truth.
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bank',type=Path,required=True)
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--forecast-run',type=Path,required=True)
    ap.add_argument('--diagnosis',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    # Prediction arrays from the original independently-trained zero/direct-fusion pilot.
    zero=np.load(a.forecast_run/'zero_val_predictions.npz')
    cot=np.load(a.forecast_run/'fusion_val_predictions.npz')
    for k in ('row_ids','lead','true_ghi'):
        assert np.array_equal(zero[k],cot[k]), f'unmatched validation key: {k}'
    rows=zero['row_ids']; y=zero['true_ghi']; p0=zero['pred_ghi']; pc=cot['pred_ghi']; lead=zero['lead']
    seq=np.load(a.base/'sequence.npy',mmap_mode='r')
    station=np.load(a.base/'station.npy',mmap_mode='r')
    assert np.all(station[rows]==1), 'validation cohort contains a non-Zhujia station'
    rowseq=seq[rows]
    # Select two large COT-related worsening examples and one example where direct COT helps.
    se0=(p0-y)**2; sec=(pc-y)**2; dse=sec-se0
    selected=[]
    for label, order in [('COT strongly worsens',np.argsort(dse)[::-1]),('COT improves',np.argsort(dse))]:
        for ix in order:
            s=int(rowseq[ix])
            if s not in [x['sequence'] for x in selected]:
                selected.append({'label':label,'row_id':int(rows[ix]),'sequence':s,'lead_idx':int(lead[ix]),
                    'truth':float(y[ix]),'no_cot':float(p0[ix]),'cot_fusion':float(pc[ix]),'delta_se':float(dse[ix])})
                break
    # Add the already discussed high-GHI case if it is distinct.
    for rid in (571443,629178,634527):
        hit=np.flatnonzero(rows==rid)
        if len(hit):
            ix=int(hit[0]); s=int(rowseq[ix])
            if s not in [x['sequence'] for x in selected]:
                selected.append({'label':'previously examined error case','row_id':rid,'sequence':s,'lead_idx':int(lead[ix]),
                    'truth':float(y[ix]),'no_cot':float(p0[ix]),'cot_fusion':float(pc[ix]),'delta_se':float(dse[ix])})
                break
    assert len(selected)>=2
    # Locate the sample index in the bank's sequence-index arrays.
    split=a.bank/'val'; ids=np.load(split/'head_row_ids.npy'); seqidx=np.load(split/'sequence_indices.npy')
    # Bank arrays are memory-mapped; only one sequence is read at a time.
    def mm(name): return np.load(split/(name+'.npy'),mmap_mode='r')
    image=mm('image_local'); cf=mm('cot_local_forecast'); co=mm('cot_local_oracle')
    history=mm('cot_local_history')
    lookup={int(v):i for i,v in enumerate(seqidx)}
    days=json.loads((a.bank/'complete.json').read_text()).get('initialization_BJT_days',{}).get('val',{})
    # Resolve time strings from full source table if the bank contract only has month-stratified days.
    lead_min=np.arange(15,241,15)
    for case in selected:
        s=case['sequence']; assert s in lookup, f'sequence {s} missing from validation bank'
        bi=lookup[s]
        # Model inputs are normalized predicted AGRI + geometry; show a clearly labeled
        # sequence-wise contrast-stretched false-colour composite (C02,C03,C13).
        rgb_idx=(1,2,10)
        ims=np.asarray(image[bi,1,:,:13],dtype=np.float32) # station=Zhujia, lead, channel,y,x
        cotf=np.expm1(np.maximum(np.asarray(cf[bi,1,:,0],dtype=np.float32),0))
        coto=np.expm1(np.maximum(np.asarray(co[bi,1,:,0],dtype=np.float32),0))
        hcot=np.expm1(np.maximum(np.asarray(history[bi,1,:,0],dtype=np.float32),0))
        # Shared scaling across all 16 forecast frames for each channel.
        rgb=[]
        for t in range(16):
            ch=[]
            for c in rgb_idx:
                q=ims[:,c]
                lo,hi=np.nanpercentile(q,[2,98]); ch.append(np.clip((ims[t,c]-lo)/max(hi-lo,1e-6),0,1))
            rgb.append(np.stack(ch,-1))
        # GHI time series for this exact sequence, retaining actual validation targets.
        m=rowseq==s
        order=np.argsort(lead[m]); truth=y[m][order]; z=p0[m][order]; f=pc[m][order]; li=lead[m][order]
        # Match available targets to canonical leads.
        target_min=li*15+15
        # AGRI forecast false-colour panel.
        fig,axes=plt.subplots(4,4,figsize=(12,12),constrained_layout=True)
        fig.suptitle(f"Zhujia validation sequence {s}: predicted AGRI false-colour proxy (C02/C03/C13)\n"
                     f"{case['label']}; direct COT-vs-no-COT ΔSE={case['delta_se']:.0f} W²m⁻⁴",fontsize=13)
        for t,ax in enumerate(axes.flat):
            ax.imshow(rgb[t],interpolation='nearest');ax.set_title(f'+{lead_min[t]} min',fontsize=9);ax.axis('off')
        fig.savefig(a.out/f'zhujia_event_{s}_agri_false_colour.png',dpi=170);plt.close(fig)
        # Predicted versus oracle COT maps on a shared physical scale.
        vmax=max(1.,float(np.nanpercentile(np.concatenate([cotf.ravel(),coto.ravel()]),99)))
        fig,axes=plt.subplots(8,4,figsize=(12,20),constrained_layout=True)
        fig.suptitle(f"Zhujia sequence {s}: forecast COT versus real-future-AGRI retrieval\n"
                     "Top: deployable forecast-AGRI→R COT. Bottom: oracle real-future-AGRI→R COT; not CPP truth.",fontsize=13)
        for t in range(16):
            ax=axes[t//4,t%4];im=ax.imshow(cotf[t],cmap='magma',vmin=0,vmax=vmax,interpolation='nearest')
            ax.set_title(f'Forecast +{lead_min[t]} min',fontsize=8);ax.axis('off')
            ax=axes[4+t//4,t%4];ax.imshow(coto[t],cmap='magma',vmin=0,vmax=vmax,interpolation='nearest')
            ax.set_title(f'Oracle +{lead_min[t]} min',fontsize=8);ax.axis('off')
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.45,label='COT (dimensionless)')
        fig.savefig(a.out/f'zhujia_event_{s}_cot_forecast_vs_oracle.png',dpi=160);plt.close(fig)
        # GHI validation series for the same 16 target leads.
        fig,ax=plt.subplots(figsize=(10,4.5),constrained_layout=True)
        ax.plot(target_min,truth,'k-o',label='Observed 15-min mean')
        ax.plot(target_min,z,'o-',label='No-COT head')
        ax.plot(target_min,f,'s-',label='Direct-COT head')
        ax.axvline((case['lead_idx']+1)*15,color='crimson',ls='--',alpha=.7,label='Selected error lead')
        ax.set(xlabel='Forecast lead (min)',ylabel='GHI (W m⁻²)',xlim=(15,240),title=f'Zhujia sequence {s}: observed and predicted GHI')
        ax.grid(alpha=.25);ax.legend(ncol=2)
        fig.savefig(a.out/f'zhujia_event_{s}_ghi.png',dpi=170);plt.close(fig)
    (a.out/'cases.json').write_text(json.dumps({'scope':'fixed validation only; no test','selected_cases':selected,'rgb':'sequence-wise contrast stretch of normalized predicted AGRI C02/C03/C13; illustrative false-colour proxy, not calibrated RGB','oracle':'real future AGRI passed through frozen R; not CPP and not deployable'},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'state':'FIGURES_COMPLETE','out':str(a.out),'cases':selected},indent=2))
if __name__=='__main__': main()
