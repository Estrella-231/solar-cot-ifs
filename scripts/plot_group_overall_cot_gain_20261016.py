from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
out=Path(r'F:\CODE_Classify\辐照度预报\solar-cot-ifs\docs\组会原始材料_20261016\09_整体增益图')
font=FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':14,'pdf.fonttype':42,'svg.fonttype':'none'})
regions=['湖南','湖北']; a=[123.980,74.361]; ac=[120.240,71.428]
x=np.arange(2); w=.34
fig,ax=plt.subplots(figsize=(8.5,5.3),layout='constrained')
ax.bar(x-w/2,a,w,label='无 COT',color='#647b8a')
ax.bar(x+w/2,ac,w,label='加 COT',color='#008b80',hatch='//',edgecolor='white',linewidth=.7)
ax.set_xticks(x,regions)
ax.set_ylabel('站点等权验证 RMSE（W/m²）')
ax.set_xlabel('区域')
ax.set_ylim(0,140)
ax.legend(frameon=False,ncols=2,loc='upper right')
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
for ext in ['png','pdf','svg']:
 fig.savefig(out/f'整体同期GHI增益_湖南湖北.{ext}',dpi=240,facecolor='white')
plt.close(fig)
