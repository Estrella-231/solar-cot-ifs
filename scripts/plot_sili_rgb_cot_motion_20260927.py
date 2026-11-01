"""Same-channel and same-stretch real/forecast RGB and COT comparison."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
exp=root/'experiments/regional_station_cot_20260925';out=exp/'sili_thick_motion_20260927_v1'
sc=json.loads((root/'configs/s_frozen_hunan_seed42.json').read_text());norm=json.loads(Path(sc['normalization']).read_text())
images=np.load(exp/'regional_ghi_pilot_bank_20260927_v1/val/image_local.npy',mmap_mode='r')
for d in json.loads((out/'selected_cases.json').read_text()):
    label=d['label'];z=np.load(out/(label+'_source.npz'));j=d['bank_index']
    channels=[2,1,0];mean=np.array(norm['mean'])[channels][None,:,None,None];std=np.array(norm['std'])[channels][None,:,None,None]
    forecast_rgb=np.moveaxis(images[j,0][:,channels]*std+mean,1,-1)
    lo=np.array(d['rgb_percentile_low']);hi=np.array(d['rgb_percentile_high'])
    real=np.clip((z['rgb']-lo)/(hi-lo+1e-8),0,1);forecast_rgb=np.clip((forecast_rgb-lo)/(hi-lo+1e-8),0,1)
    leads=sorted(set([0,3,7,11,15,d['peak_lead_min']//15-1]));fig,axs=plt.subplots(5,len(leads),figsize=(3*len(leads),12),layout='constrained')
    for col,ld in enumerate(leads):
        panels=[real[ld],forecast_rgb[ld],np.where(z['valid'][ld],z['reference'][ld],np.nan),z['simvp'][ld],z['oracle'][ld]]
        for ri,p in enumerate(panels):
            ax=axs[ri,col]
            if np.isfinite(p).any():im=ax.imshow(p,interpolation='nearest',**({} if ri<2 else dict(cmap='magma',vmin=0,vmax=70)))
            else:ax.text(.5,.5,'Missing',transform=ax.transAxes,ha='center')
            ax.plot(32,32,'+',color='cyan',ms=9);ax.set_xticks([]);ax.set_yticks([])
            if col==0:ax.set_ylabel(['Real AGRI RGB proxy','SimVP AGRI RGB proxy','CPP reference','SimVP AGRI -> R','Real AGRI -> R'][ri])
            if ri==0:ax.set_title(f'+{(ld+1)*15} min')
    fig.colorbar(im,ax=axs[2:,:],label='COT',extend='max',shrink=.5)
    fig.suptitle(f'Sili {d["init"]} BJT | {label}\nSame C03/C02/C01 channels and shared real-image 2-98% stretch; cyan station cross')
    fig.savefig(out/(label+'_rgb_cot.png'),dpi=140);fig.savefig(out/(label+'_rgb_cot.pdf'));plt.close(fig)
print('COMPLETE_RGB_COT',flush=True)
