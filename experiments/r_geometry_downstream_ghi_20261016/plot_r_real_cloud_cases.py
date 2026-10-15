#!/usr/bin/env python3
"""Plot paired R outputs on real AGRI validation scenes with fixed scales."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--payload',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    z=np.load(a.payload,allow_pickle=False);meta=json.loads(a.payload.with_suffix('.json').read_text(encoding='utf-8'))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.unicode_minus':False})
    raw=np.asarray(z['rgb_c01_c02_c03'],np.float64)
    rgb=np.moveaxis(raw[:,[2,1,0]],1,-1) # display RGB = C03,C02,C01
    # One shared per-band stretch across all selected scenes.
    lo=np.nanpercentile(rgb.reshape(-1,3),2,axis=0);hi=np.nanpercentile(rgb.reshape(-1,3),98,axis=0)
    hi=np.maximum(hi,lo+1e-6)
    rgb=np.clip((rgb-lo[None,None,None,:])/(hi-lo)[None,None,None,:],0,1)**(1/2.2)
    n=len(meta['cases']);fig,axs=plt.subplots(n,5,figsize=(14,2.65*n),constrained_layout=True,squeeze=False)
    titles=['Real AGRI false-colour','CPP retrieval reference','R: old geometry retrain','R: corrected geometry retrain','Corrected − old R']
    for i,c in enumerate(meta['cases']):
        arrays=[None,z['cpp_reference'][i],z['old_r'][i],z['corrected_r'][i],z['corrected_r'][i]-z['old_r'][i]]
        axs[i,0].imshow(rgb[i],interpolation='nearest');axs[i,0].plot(8,8,'w+',mec='k',mew=1.2,ms=9)
        for j,v in enumerate(arrays[1:],1):
            if j<4:im=axs[i,j].imshow(v,vmin=0,vmax=60,cmap='magma',interpolation='nearest')
            else:im=axs[i,j].imshow(v,vmin=-20,vmax=20,cmap='coolwarm',interpolation='nearest')
            axs[i,j].plot(8,8,'w+',mec='k',mew=1.2,ms=9)
        for ax in axs[i]:
            ax.set_xticks([0,8,15]);ax.set_yticks([0,8,15]);ax.set_xlabel('column');ax.set_ylabel('row')
        axs[i,0].set_title(f"{c['station'].title()} · {c['target_time_utc']} UTC\n{c['selection_rule']}")
    for j,t in enumerate(titles):axs[0,j].set_title(t if j else titles[0]+'\n'+axs[0,j].get_title())
    # Colorbars communicate the fixed physical ranges for each column family.
    fig.colorbar(im,ax=list(axs[:,4]),shrink=.72,label='ΔCOT (corrected − old)')
    sm=plt.cm.ScalarMappable(norm=plt.Normalize(0,60),cmap='magma')
    fig.colorbar(sm,ax=list(axs[:,1:4].ravel()),shrink=.72,label='COT',extend='max')
    fig.suptitle('Paired R validation on the same observed AGRI scenes · validation diagnostics only',fontsize=13)
    fig.text(.5,-.005,'RGB=[C03,C02,C01], shared 2–98% per-band stretch over shown scenes; COT panels share 0–60 and Δ uses −20…+20. Star marks station pixel (8,8). CPP is retrieval reference, not independent truth. Cases follow fixed selection rules.',ha='center',fontsize=8)
    fig.savefig(a.output,dpi=180,bbox_inches='tight')
    fig.savefig(a.output.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'state':'FIGURE_COMPLETE','output':str(a.output),'rgb_shared_limits_lo':lo.tolist(),'rgb_shared_limits_hi':hi.tolist(),'cot_range':[0,60],'difference_range':[-20,20]},indent=2))
if __name__=='__main__':main()
