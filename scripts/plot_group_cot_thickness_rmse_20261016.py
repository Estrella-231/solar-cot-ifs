from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
import numpy as np
base=Path(r'F:\CODE_Classify\辐照度预报\solar-cot-ifs\docs\组会原始材料_20261016\02_厚度分层')
df=pd.read_csv(base/'thickness_summary.csv')
order=['nearzero_[0,0.1)','thin_[0.1,5)','moderate_[5,30)','thick_[30,100]']
df=df[df.source.eq('CPP_reference_centre')].set_index('bin').loc[order].reset_index()
labels=['近零\n[0, 0.1)\nn=464','薄云\n[0.1, 5)\nn=181','中等\n[5, 30)\nn=466','厚云\n[30, 100]\nn=173']
font=FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':12,'axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12,'pdf.fonttype':42,'svg.fonttype':'none'})
x=np.arange(len(df));w=.34
fig,ax=plt.subplots(figsize=(9.2,5.8))
fig.subplots_adjust(left=.12,right=.98,bottom=.20,top=.97)
bars1=ax.bar(x-w/2,df.A_RMSE_Wm2,w,label='无 COT',color='#0072B2',edgecolor='white',linewidth=.8,zorder=3)
bars2=ax.bar(x+w/2,df.AC_RMSE_Wm2,w,label='加 COT',color='#009E73',edgecolor='white',linewidth=.8,zorder=3)
for bars in [bars1,bars2]:
 for bar in bars:
  ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2.5,f'{bar.get_height():.1f}',ha='center',va='bottom',fontsize=10,color='#263746')
ax.set_xticks(x,labels)
ax.set_xlabel('CPP参考中心 COT（无量纲）')
ax.set_ylabel('站点等权 GHI RMSE（W m$^{-2}$）')
ax.set_ylim(0,190);ax.set_yticks(np.arange(0,191,30))
ax.legend(frameon=False,ncols=2,loc='upper right',handlelength=1.6,columnspacing=1.4)
ax.grid(axis='y',color='#d8e0e6',linewidth=.7,zorder=0)
ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#55636f');ax.spines['bottom'].set_color('#55636f')
ax.tick_params(axis='both',length=4,width=.8,color='#55636f')
for ext in ['png','pdf','svg']:
 fig.savefig(base/f'CPP厚度分层_GHI_RMSE.{ext}',dpi=300,facecolor='white')
plt.close(fig)
