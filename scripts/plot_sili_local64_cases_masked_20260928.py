from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

base=Path(__file__).resolve().parents[1]/'results'/'regional_station_cot_20260927'/'sili_patch_case_audit_20260928_v3'
z=np.load(base/'case_maps.npz')
cases=json.loads((base/'case_manifest.json').read_text())
fig,axes=plt.subplots(len(cases),6,figsize=(15,2.8*len(cases)),constrained_layout=True)
cols=['CPP retrieval','SimVP→R','Persistence','Transport / step 0','Learned residual','Learned − CPP']
for j,title in enumerate(cols): axes[0,j].set_title(title,fontsize=10)
cmap=plt.get_cmap('magma').copy(); cmap.set_bad('#d8d8d8')
dcmap=plt.get_cmap('RdBu_r').copy(); dcmap.set_bad('#d8d8d8')
cot_im=diff_im=None
for r,c in enumerate(cases):
    valid=z['common_support'][r] & z['reference_valid'][r]
    fields=[z['truth'][r],z['simvp'][r],z['persistence'][r],z['transport'][r],z['learned'][r],z['learned'][r]-z['truth'][r]]
    for j,field in enumerate(fields):
        ax=axes[r,j]
        masked=np.ma.masked_where(~valid,field)
        if j<5:
            im=ax.imshow(masked,origin='upper',cmap=cmap,vmin=0,vmax=100,interpolation='nearest')
            if j==0: cot_im=im
        else:
            im=ax.imshow(masked,origin='upper',cmap=dcmap,norm=TwoSlopeNorm(vmin=-50,vcenter=0,vmax=50),interpolation='nearest')
            diff_im=im
        ax.scatter([32],[32],marker='+',color='cyan',s=45,linewidths=1.2)
        ax.set_xticks([]); ax.set_yticks([])
        if j==0: ax.set_ylabel(f"{c['kind']}\n{c['init_day_BJT']}\n+{c['lead_min']} min",fontsize=8)
fig.colorbar(cot_im,ax=axes[:,:5].ravel().tolist(),shrink=.75,label='COT',location='bottom',pad=.02)
fig.colorbar(diff_im,ax=axes[:,5].ravel().tolist(),shrink=.75,label='Learned − CPP COT',location='bottom',pad=.02)
fig.suptitle('Sili local 64×64 validation cases — only CPP-valid & transport-supported pixels shown; gray = excluded',fontsize=13)
for ext in ('png','pdf'): fig.savefig(base/f'case_maps_common_support.{ext}',dpi=180 if ext=='png' else None)
plt.close(fig)
print(base/'case_maps_common_support.png')
