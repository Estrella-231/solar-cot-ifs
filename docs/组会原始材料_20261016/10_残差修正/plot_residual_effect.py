import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

out = Path(r'F:\CODE_Classify\辐照度预报\solar-cot-ifs\docs\组会原始材料_20261016\10_残差修正')
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':['Microsoft YaHei','SimHei','DejaVu Sans'],'font.size':13,'axes.titlesize':16,'axes.labelsize':13,'figure.facecolor':'white','axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
fig, (a,b)=plt.subplots(1,2,figsize=(12.5,5.4),layout='constrained',gridspec_kw={'width_ratios':[1.15,1]})
blue='#0072B2'; orange='#D55E00'; gray='#8A8A8A'
# Whole validation: honest zero-based bars
names=['无残差修正','AGRI 残差','AGRI + COT 残差']
vals=[123.979,111.277,113.419]
colors=[gray,blue,orange]
bars=a.bar(np.arange(3),vals,color=colors,width=.62,zorder=3)
for i,(bar,v) in enumerate(zip(bars,vals)):
    a.text(bar.get_x()+bar.get_width()/2,v+2.2,f'{v:.1f}',ha='center',va='bottom',fontsize=13,fontweight='bold')
a.text(.98,105.2,'↓ 10.2%',ha='center',color=blue,fontweight='bold',fontsize=12)
a.text(1.98,105.2,'↓ 8.5%',ha='center',color=orange,fontweight='bold',fontsize=12)
a.set_ylim(0,140); a.set_xticks(range(3),names); a.set_ylabel('验证集 GHI RMSE（W/m²）'); a.set_title('整体预报误差'); a.grid(axis='y',color='#DDDDDD',lw=.8); a.set_axisbelow(True)
# Direction correctness on baseline-underprediction samples
names2=['AGRI 残差','AGRI + COT 残差']; vals2=[54.9,55.9]
bars=b.bar([0,1],vals2,color=[blue,orange],width=.58,zorder=3)
for bar,v in zip(bars,vals2): b.text(bar.get_x()+bar.get_width()/2,v+2.2,f'{v:.1f}%',ha='center',fontweight='bold',fontsize=14)
b.axhline(50,color='#333333',lw=1.5,ls='--'); b.text(1.42,51.2,'50%',ha='right',fontsize=11,color='#333333')
b.set_ylim(0,100); b.set_xticks([0,1],names2); b.set_ylabel('修正方向正确率'); b.set_title('基线低估时，是否成功上调？'); b.grid(axis='y',color='#DDDDDD',lw=.8); b.set_axisbelow(True)
fig.suptitle('残差修正：总体误差下降；低估样本的上调仍接近五五开',fontsize=17,fontweight='bold')
fig.savefig(out/'残差修正效果.png',dpi=220,bbox_inches='tight')
fig.savefig(out/'残差修正效果.svg',bbox_inches='tight')
plt.close(fig)
print(out/'残差修正效果.png')
