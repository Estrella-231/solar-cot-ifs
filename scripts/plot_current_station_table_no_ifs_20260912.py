"""Render the current Hunan station table in the supplied blue reference style."""
import hashlib,json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'figures/current_no_ifs_20260912/current_station_table_no_ifs_20260912.json'
OUT=ROOT/'figures/current_no_ifs_20260912/current_station_metrics_no_ifs_20260912'
for suffix in ('.png','.pdf','.svg','.manifest.json'):
    assert not OUT.with_suffix(suffix).exists(),f'refuse overwrite: {OUT.with_suffix(suffix)}'
report=json.loads(DATA.read_text())
assert report['state']=='COMPLETE_CURRENT_NO_IFS_STATION_TABLE_METRICS' and report['test_used'] is False

mpl.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False})
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
bold=FontProperties(fname='C:/Windows/Fonts/msyhbd.ttc')
blue='#4778C4';light1='#D8E1F2';light2='#EBEFF9';white='#FFFFFF';black='#15191F';note='#374151'
columns=['站点','反演方法','R²','RMSE (W/m²)','nRMSE (%)','nMAE (%)']
widths=[.12,.29,.15,.19,.145,.145];scale=sum(widths);widths=[w/scale for w in widths]
x=[.02]
table_width=.96
for w in widths:x.append(x[-1]+table_width*w)
top=.94;header_h=.145;row_h=.155

fig=plt.figure(figsize=(12,5.0),facecolor='white')
ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')

def cell(x0,y0,w,h,color,text,fp=font,size=14,text_color=black,lw=2.5):
    ax.add_patch(Rectangle((x0,y0),w,h,facecolor=color,edgecolor=white,linewidth=lw))
    ax.text(x0+w/2,y0+h/2,text,ha='center',va='center',fontproperties=fp,fontsize=size,color=text_color,linespacing=1.25)

for j,label in enumerate(columns):cell(x[j],top-header_h,x[j+1]-x[j],header_h,blue,label,bold,15,white)

summary={(r['station'],r['method']):r for r in report['summary']}
metrics=('r2','rmse_wm2','nrmse_pct','nmae_pct')
formats={'r2':lambda m,s:f'{m:.3f} ± {s:.3f}',
         'rmse_wm2':lambda m,s:f'{m:.2f} ± {s:.2f}',
         'nrmse_pct':lambda m,s:f'{m:.2f} ± {s:.2f}',
         'nmae_pct':lambda m,s:f'{m:.2f} ± {s:.2f}'}
rows=[('sili','四里','A'),('sili','四里','AC'),('zhujia','竺家','A'),('zhujia','竺家','AC')]
best={}
for station in ('sili','zhujia'):
    for metric in metrics:
        vals={arm:summary[(station,arm)][metric]['mean'] for arm in ('A','AC')}
        best[(station,metric)]=max(vals,key=vals.get) if metric=='r2' else min(vals,key=vals.get)

for i,(station,station_cn,arm) in enumerate(rows):
    y=top-header_h-(i+1)*row_h;color=light1 if i%2==0 else light2
    if i%2==0:cell(x[0],y-row_h,x[1]-x[0],2*row_h,blue,station_cn,bold,16,white)
    item=summary[(station,arm)]
    method='FY4B外推（A）' if arm=='A' else 'FY4B外推 + COT云属性（AC）'
    cell(x[1],y,x[2]-x[1],row_h,blue,method,bold,13.5,white)
    for j,metric in enumerate(metrics,2):
        m=item[metric]['mean'];s=item[metric]['std_ddof1'];fp=bold if best[(station,metric)]==arm else font
        cell(x[j],y,x[j+1]-x[j],row_h,color,formats[metric](m,s),fp,12.8,black)

note_y=.075
ax.text(.02,note_y,
    '注：湖南验证集；数值为 mean ± SD（3个训练随机种子）。四里 n=49,806，竺家 n=50,043。\n'
    'nRMSE/nMAE 均除以同站同批实测 GHI 均值；粗体为均值最优，不代表显著差异。当前未加入 IFS。',
    ha='left',va='center',fontproperties=font,fontsize=10.5,color=note,linespacing=1.45)

fig.savefig(OUT.with_suffix('.png'),dpi=200,facecolor='white',transparent=False)
fig.savefig(OUT.with_suffix('.pdf'),facecolor='white',transparent=False)
fig.savefig(OUT.with_suffix('.svg'),facecolor='white',transparent=False)
plt.close(fig)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=dict(state='COMPLETE_NO_IFS_STATION_TABLE_EXPORT',source=str(DATA),source_sha256=sha(DATA),
    outputs={p.suffix:{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in (OUT.with_suffix('.png'),OUT.with_suffix('.pdf'),OUT.with_suffix('.svg'))},
    figure_inches=[12,5.0],png_dpi=200,background='opaque white',font='Microsoft YaHei',
    style='blue merged-cell table based on user-supplied reference image; no source pixels reused',
    aggregation='metric mean plus sample SD across seeds42/43/44; nominal mean winner bolded',
    missing_data='none in saved prediction rows; no IFS row because IFS experiment is absent',
    caveats=report['limitations'])
OUT.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False))
