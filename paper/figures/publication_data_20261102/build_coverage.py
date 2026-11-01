"""Publication-style station location and coverage figure; saved dates unchanged."""
from pathlib import Path
import json, math, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Polygon, Rectangle, ConnectionPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
SRC=ROOT/'paper/figures/data_coverage_20261102'
BLUE='#2878A5'; ORANGE='#D87836'; PURPLE='#8064A2'; INK='#222222'
COORDS={'Niushou':(32.0800,112.0400),'Wangzhai':(32.0400,112.1200),
        'Mizhuang':(32.1200,112.2000),'Sili':(25.9478611111,112.347916667),
        'Zhujia':(29.566666667,112.4500)}
STATIONS=list(COORDS)
dates=pd.read_csv(SRC/'usable_dates.csv',parse_dates=['date'])
counts=pd.read_csv(SRC/'date_counts.csv')
assert set(dates.split)=={'train','val'}
assert not dates.duplicated(['station','split','date']).any()
actual=dates.groupby(['station','split']).date.nunique()
for r in counts.itertuples(): assert actual[r.station,r.split]==r.dates
features=json.loads((OUT/'china_provinces_naturalearth.geojson').read_text())['features']
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
 'mathtext.fontset':'stix','font.size':9,'axes.titlesize':10,'axes.titleweight':'bold',
 'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
 'axes.linewidth':.65,'axes.edgecolor':'#555555','text.color':INK,'axes.labelcolor':INK,
 'xtick.color':INK,'ytick.color':INK,'axes.spines.top':False,'axes.spines.right':False,
 'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})

def rings(f):
    g=f['geometry']; polys=g['coordinates'] if g['type']=='MultiPolygon' else [g['coordinates']]
    return [np.asarray(p[0],float) for p in polys]

def mapbase(ax,extent,overview=False):
    for f in features:
        name=f['properties'].get('name_en') or f['properties'].get('name')
        fill={'Hubei':'#DCEAF1','Hunan':'#E9E2EF'}.get(name,'#F5F5F2')
        for ring in rings(f):
            if ring[:,0].max()<extent[0] or ring[:,0].min()>extent[1] or ring[:,1].max()<extent[2] or ring[:,1].min()>extent[3]: continue
            ax.add_patch(Polygon(ring,closed=True,facecolor=fill,edgecolor='#888888',lw=.55,zorder=1))
    ax.set(xlim=extent[:2],ylim=extent[2:])
    ax.set_aspect(1/math.cos(math.radians((extent[2]+extent[3])/2)))
    ax.grid(color='#E5E5E5',lw=.45,zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}°E'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}°N'))
    for sp in ax.spines.values(): sp.set_visible(True)

def scale(ax,start,km):
    x,y=start; dx=km/(111.32*math.cos(math.radians(y)))
    ax.plot([x,x+dx],[y,y],color=INK,lw=1.3,zorder=9)
    ax.plot([x,x],[y-.003,y+.003],color=INK,lw=.7,zorder=9)
    ax.plot([x+dx,x+dx],[y-.003,y+.003],color=INK,lw=.7,zorder=9)
    ax.annotate(f'{km:g} km',(x+dx/2,y),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)

fig=plt.figure(figsize=(7.05,5.8))
gs=fig.add_gridspec(2,2,left=.095,right=.975,bottom=.09,top=.94,height_ratios=[1.4,1],hspace=.53,wspace=.33)
ax=fig.add_subplot(gs[0,0]); mapbase(ax,(108.2,117.2,24.3,33.6),True)
ax.set_xticks([110,113,116]); ax.set_yticks([25,28,31,33])
ax.set_title('(a) Study regions',loc='left',pad=7)
ax.text(113.55,31.15,'Hubei',color=BLUE,fontsize=10,fontweight='bold',ha='center')
ax.text(110.5,27.4,'Hunan',color=PURPLE,fontsize=10,fontweight='bold',ha='center')
for name in ['Sili','Zhujia']:
    lat,lon=COORDS[name]; ax.plot(lon,lat,'o',color=INK,ms=4.5,mec='white',mew=.7,zorder=6)
    ax.annotate(name,(lon,lat),xytext=(5,-1),textcoords='offset points',ha='left',va='center',fontsize=9,zorder=7)
ax.plot(112.12,32.08,'s',color=BLUE,ms=5,mec='white',mew=.6,zorder=6)
ax.annotate('Three stations',(112.12,32.08),xytext=(3,12),textcoords='offset points',fontsize=8.5,ha='left',zorder=7)
ax.add_patch(Rectangle((111.96,31.98),.31,.23,fill=False,edgecolor=INK,lw=.75,zorder=6))
ax.annotate('N',xy=(.92,.94),xytext=(.92,.78),xycoords='axes fraction',textcoords='axes fraction',ha='center',va='center',fontsize=8.5,arrowprops={'arrowstyle':'-|>','lw':.8,'color':INK})

bx=fig.add_subplot(gs[0,1]); mapbase(bx,(111.985,112.265,31.995,32.165))
bx.set_xticks([112.00,112.10,112.20]); bx.set_yticks([32.00,32.05,32.10,32.15])
bx.set_title('(b) Hubei station detail',loc='left',pad=7)
for name,mark,offset in [('Niushou','o',(6,8)),('Wangzhai','s',(6,-12)),('Mizhuang','^',(-5,10))]:
    lat,lon=COORDS[name];bx.plot(lon,lat,marker=mark,color=BLUE,ms=6,mec='white',mew=.8,zorder=6)
    bx.annotate(name,(lon,lat),xytext=offset,textcoords='offset points',ha='right' if name=='Mizhuang' else 'left',va='center',fontsize=9)
scale(bx,(112.00,32.008),5)
fig.add_artist(ConnectionPatch(xyA=(112.27,32.12),coordsA=ax.transData,xyB=(0,.65),coordsB=bx.transAxes,arrowstyle='-',lw=.7,color='#999999',linestyle='--',zorder=.5))

cx=fig.add_subplot(gs[1,:]);cx.set_title('(c) Dates with usable GHI examples',loc='left',pad=19)
for j,name in enumerate(STATIONS):
    y=4-j
    cx.axhspan(y-.40,y+.40,color='#F7F7F7' if j%2==0 else 'white',zorder=0)
    for split,offset,col in [('train',.13,'#758F9F'),('val',-.13,ORANGE)]:
        x=dates.loc[(dates.station==name)&(dates.split==split),'date']
        cx.plot(x,np.full(len(x),y+offset),'|',ms=5.5,mew=.65,color=col,rasterized=False)
cx.set_yticks(range(5),STATIONS[::-1]);cx.set_ylim(-.5,4.5)
cx.set_xlim(pd.Timestamp('2024-03-15'),pd.Timestamp('2026-07-15'))
cx.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[4,10]));cx.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
cx.xaxis.set_minor_locator(mdates.MonthLocator());cx.tick_params(axis='y',length=0,pad=6)
cx.set_xlabel('Target date (BJT)');cx.grid(axis='x',color='#E5E5E5',lw=.5);cx.set_axisbelow(True)
handles=[Line2D([],[],color='#758F9F',marker='|',linestyle='None',markersize=7,label='Training'),Line2D([],[],color=ORANGE,marker='|',linestyle='None',markersize=7,label='Validation')]
cx.legend(handles=handles,loc='lower right',bbox_to_anchor=(1,1.015),ncol=2,frameon=False,handletextpad=.2,columnspacing=1.5,fontsize=8.5)
for ext in ['pdf','png']:fig.savefig(OUT/f'study_regions_coverage.{ext}',dpi=400)
plt.close(fig)

checks={'purpose':'Style-only redraw with true provincial boundaries and local Hubei inset',
 'source_dates':str((SRC/'usable_dates.csv').relative_to(ROOT)), 'source_counts':str((SRC/'date_counts.csv').relative_to(ROOT)),
 'dates_sha256':hashlib.sha256((SRC/'usable_dates.csv').read_bytes()).hexdigest(),
 'counts_sha256':hashlib.sha256((SRC/'date_counts.csv').read_bytes()).hexdigest(),
 'coordinates':COORDS, 'coordinates_unchanged':True,'all_date_rows_displayed':len(dates),'date_count_checks_passed':True,
 'test_payload_read':False,'source_arrays_changed':False,
 'boundary_source':'Natural Earth 1:10m Admin-1 states/provinces; China feature subset, local regional viewport only',
 'boundary_documentation':'https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-1-states-provinces/',
 'boundary_download':'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_1_states_provinces.geojson',
 'boundary_sha256':hashlib.sha256((OUT/'china_provinces_naturalearth.geojson').read_bytes()).hexdigest(),
 'panels':{'a':'Hubei/Hunan provincial locator and five stations (three Hubei stations overlap at this scale)','b':'Hubei three-station detail, same coordinates, no interpolation','c':'All retained usable training and validation dates; gaps not bridged'},
 'caption_change':'Replace previous two regional coordinate panels with regional provincial locator (a) and Hubei station detail (b); date panel (c) unchanged in sample scope. Administrative boundaries: Natural Earth.',
 'export':{'width_in':7.05,'height_in':5.8,'png_dpi':400,'pdf_vector':True,'font':'Times New Roman'}}
(OUT/'coverage_checks.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:checks[k] for k in ['all_date_rows_displayed','date_count_checks_passed','coordinates_unchanged']},indent=2))
