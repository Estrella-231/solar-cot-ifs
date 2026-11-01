from pathlib import Path
import csv,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
ROOT=Path(r'F:\CODE_Classify\辐照度预报\solar-cot-ifs')
OUT=ROOT/'docs/组会原始材料_20261016/09_整体增益图'
OUT.mkdir(exist_ok=True)
font=FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':13,'pdf.fonttype':42,'svg.fonttype':'none'})
src=[ROOT/'refine-logs/cot_value_first_20261013/results/T18_RAL_RESULTS_20261014.md',ROOT/'refine-logs/hubei_cot_independent_20261014/PROGRESS_20261016.md']
rows=[dict(region='湖南（2站）',A=123.980,AC=120.240,AP=123.822,reduction_pct=3.02,improvement=3.740,low=1.597,high=6.180,nval=4479),dict(region='湖北（3站）',A=74.361,AC=71.428,AP=73.894,reduction_pct=3.94,improvement=2.932,low=.939,high=5.070,nval=2456)]
with (OUT/'整体增益_绘图数据.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
fig,(a,b)=plt.subplots(1,2,figsize=(13.8,6.5),gridspec_kw={'width_ratios':[1.2,1]})
fig.subplots_adjust(left=.07,right=.975,bottom=.24,top=.78,wspace=.38)
x=np.arange(2); width=.29
for shift,key,label,col,hatch in [(-width/2,'A','AGRI + 太阳几何','#647b8a',''),(width/2,'AC','AGRI + 太阳几何 + COT','#008b80','//')]:
 vals=[r[key] for r in rows];bars=a.bar(x+shift,vals,width,label=label,color=col,hatch=hatch,edgecolor='white')
 for p,v in zip(bars,vals):a.text(p.get_x()+p.get_width()/2,v+2,f'{v:.2f}',ha='center',fontsize=13)
a.set_xticks(x,[r['region'] for r in rows]);a.set_ylim(0,148);a.set_ylabel('GHI RMSE（W/m²）');a.set_title('（a）整体同期估计误差',loc='left',fontsize=15,pad=14)
a.legend(loc='upper center',bbox_to_anchor=(.5,-.10),frameon=False,fontsize=10)
for i,r in enumerate(rows):a.text(i,138,f'RMSE降低 {r["reduction_pct"]:.2f}%',ha='center',color='#00786f',fontweight='bold',fontsize=14)
y=np.array([1,0]);v=np.array([r['improvement'] for r in rows]);lo=np.array([r['low'] for r in rows]);hi=np.array([r['high'] for r in rows]);b.errorbar(v,y,xerr=np.vstack([v-lo,hi-v]),fmt='o',color='#007f76',capsize=6,markersize=8,linewidth=2)
b.axvline(0,color='#6b7780',linestyle='--',linewidth=1);b.set_xlim(-.4,7);b.set_ylim(-.65,1.65);b.set_yticks(y,[r['region'] for r in rows]);b.set_xlabel('RMSE降低量：无COT − 加COT（W/m²）');b.set_title('（b）配对差值与95%日期块区间',loc='left',fontsize=15,pad=14)
for r,yy in zip(rows,y):b.text(r['improvement'],yy+.21,f'{r["improvement"]:.2f}  [{r["low"]:.2f}, {r["high"]:.2f}]',ha='center',fontsize=12)
for ax in (a,b):
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y' if ax is a else 'x',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('加入COT后，基础同期GHI头的验证RMSE降低约3%–4%',fontsize=20,fontweight='bold',y=.96)
fig.text(.5,.865,'湖南与湖北独立训练；各区域内同样本配对，站点等权、三种子等权',ha='center',fontsize=13,color='#4d6070')
fig.text(.07,.075,'口径：真实目标时刻AGRI → 冻结R反演COT → 同期15分钟GHI估计；各区域使用各自验证集。',fontsize=10,color='#4d6070')
fig.text(.07,.035,'验证选模后的阶段性结果；图中区间为配对差值的日期块bootstrap区间。3%–4%不代表未来预报增益。',fontsize=10,color='#4d6070')
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'整体同期GHI增益_湖南湖北.{ext}',dpi=240,facecolor='white')
plt.close(fig)
(OUT/'图注与口径.md').write_text('''# 整体同期GHI增益图

推荐图注：在湖南两站和湖北三站分别独立训练的基础同期GHI头中，加入冻结反演器输出的局地COT后，站点与三种子等权验证RMSE分别由123.980降至120.240 W/m²、由74.361降至71.428 W/m²，对应约3.02%和3.94%的下降。左图柱状轴从零开始；右图为配对RMSE降低量及95%日期块bootstrap区间，不是单个模型RMSE的误差条。

这是基础同期头的结果，不是未来0–4h预报，也不是所有头部合同共有的提升。更强AGRI残差头和三帧局地头的COT额外收益尚不稳定，详见材料04/05。两省各自独立评价，不将其合并为统一总体百分比。

湖南区间采用共享52日期union bootstrap [−6.180,−1.597] 的方向反转；湖北采用报告中的共享日期区间 [−5.070,−0.939]。区间有验证选模条件，不是独立测试结论。数据来自已有审计报告，本次未重新运行bootstrap或训练。

湖北A/AC公开数值相减为2.933，原报告使用未四舍五入数值得到2.932；右图保留原报告差值。相对降幅只标约3.94%。湖南4479条验证行；湖北三站共2456条验证行；各自三种子不是新增独立观测样本。

PNG直接插入PPT；PDF为矢量；SVG可编辑；CSV保留图中数值。来源与SHA见manifest.json。
''',encoding='utf-8')
(OUT/'manifest.json').write_text(json.dumps({'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in src],'matplotlib':matplotlib.__version__,'scope':'same-time base-head validation; within-region paired; no forecast gain claim','uncertainty':'reported paired shared-date percentile bootstrap 95% CI; sign reversed from AC-A','transform':'plot reported rounded endpoints and reported paired deltas; no raw-prediction recomputation','files':[p.name for p in OUT.iterdir()]},ensure_ascii=False,indent=2),encoding='utf-8')
print(OUT/'整体同期GHI增益_湖南湖北.png')

