from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
R=Path(__file__).parent;plt.rcParams.update({'font.size':11,'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'pdf.fonttype':42})
fig,axes=plt.subplots(2,3,figsize=(10.5,6.7))
for row,init in enumerate(('202508230500','202511060500')):
 z=np.load(R/'source_probe'/(init+'_with_cot_ghi.npz'))
 for col,(ag,cot,label,hour) in enumerate([('canonical_agri','canonical_cot','Correct-time real AGRI','09:00'),('saved_truth','saved_truth_cot','Time-misbound saved real AGRI','17:00'),('saved_pred','saved_pred_cot','Time-misbound SimVP output','17:00')]):
  ax=axes[row,col];a=z[ag][-1];rgb=np.clip(np.moveaxis(a[[2,1,0]],0,-1)/.8,0,1)**.6;ax.imshow(rgb);ax.scatter(6,8,marker='*',s=85,c='white',edgecolors='black');ax.add_patch(Rectangle((4.5,6.5),3,3,fill=False,edgecolor='#56b4e9'));ax.set_xticks([]);ax.set_yticks([]);date=init[:4]+'-'+init[4:6]+'-'+init[6:8];v=z[cot][-1,0,8,6];ax.set_xlabel(f'{label}\n{date} {hour} BJT | R-COT = {v:.2f}',fontsize=10)
fig.tight_layout();fig.savefig(R/'source_figures/source_clock_proof.png',dpi=220,bbox_inches='tight');fig.savefig(R/'source_figures/source_clock_proof.pdf',bbox_inches='tight');plt.close(fig)
