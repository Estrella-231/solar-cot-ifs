"""Station coordinates and usable observed-image GHI dates; no test payloads."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
HB = ROOT / 'refine-logs/hubei_same_time_geometry_20261102/recovered/pack/rows.csv'
HN = ROOT / 'paper/revision_records/20261102_completion/hunan_ghi_trainval_keys.csv'
COORDS = {'Niushou': (32.0800,112.0400), 'Wangzhai': (32.0400,112.1200),
          'Mizhuang': (32.1200,112.2000), 'Sili': (25.9478611111,112.347916667),
          'Zhujia': (29.566666667,112.4500)}
STATIONS = list(COORDS)


def main():
    hb = pd.read_csv(HB)
    hb['date'] = pd.to_datetime(hb.time_bjt.astype(str), format='%Y%m%d%H%M').dt.normalize()
    hn = pd.read_csv(HN)
    hn = hn.loc[~hn.excluded_station_day].copy()
    hn['station'] = hn.station.str.capitalize()
    hn['date'] = pd.to_datetime(hn.target_time_bjt).dt.normalize()
    df = pd.concat([hb[['station','split','date']], hn[['station','split','date']]], ignore_index=True)
    assert set(df.split) <= {'train','val','validation'}
    df['split'] = df['split'].replace({'validation':'val'})
    dates = df.drop_duplicates().sort_values(['station','split','date'])
    dates.to_csv(OUT/'usable_dates.csv', index=False)
    stats = df.groupby(['station','split']).agg(rows=('date','size'), dates=('date','nunique'), start=('date','min'), end=('date','max')).reset_index()
    stats.to_csv(OUT/'date_counts.csv',index=False)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
                         'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
                         'pdf.fonttype':42, 'ps.fonttype':42, 'axes.spines.top':False,
                         'axes.spines.right':False}):
        fig = plt.figure(figsize=(7.15,5.0))
        gs = fig.add_gridspec(2,2,height_ratios=[1.15,1],left=.12,right=.975,bottom=.12,top=.94,hspace=.57,wspace=.38)
        for col,(region,names,lims) in enumerate([
            ('Hubei',STATIONS[:3],(111.96,112.28,31.96,32.20)),
            ('Hunan',STATIONS[3:],(109.8,115.0,25.05,30.45))]):
            ax = fig.add_subplot(gs[0,col])
            for i,name in enumerate(names):
                lat,lon=COORDS[name]
                ax.scatter(lon,lat,s=42,marker=['o','s','^'][i],c='#1B6489',edgecolor='white',linewidth=.7,zorder=4)
                dy = -12 if name=='Wangzhai' else 8
                ax.annotate(name,(lon,lat),xytext=(5,dy),textcoords='offset points',fontsize=8,ha='left',va='center')
            ax.set(xlim=lims[:2],ylim=lims[2:],xlabel='Longitude (°E)',ylabel='Latitude (°N)')
            ax.set_aspect(1/math.cos(math.radians(np.mean([COORDS[n][0] for n in names]))))
            ax.set_title(f'({chr(97+col)}) {region} station locations',loc='left',fontweight='bold')
            ax.grid(color='#e5e8eb',linewidth=.55)
            ax.ticklabel_format(useOffset=False)
        ax=fig.add_subplot(gs[1,:])
        for j,name in enumerate(STATIONS):
            for split,offset,color in [('train',.13,'#7D8B95'),('val',-.13,'#D55E00')]:
                x=dates.loc[(dates.station==name)&(dates.split==split),'date']
                ax.plot(x,np.full(len(x),4-j+offset),'|',ms=5.5,mew=.7,color=color,label={'train':'Training dates','val':'Validation dates'}[split] if j==0 else None)
        ax.set_yticks(range(5),STATIONS[::-1])
        ax.set_ylim(-.45,4.65)
        ax.set_xlim(pd.Timestamp('2024-03-15'),pd.Timestamp('2026-07-15'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[4,10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.set_xlabel('Target date (BJT)')
        ax.set_title('(c) Dates with usable observed-image GHI samples',loc='left',fontweight='bold',pad=21)
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.00),ncol=2,frameon=False,handletextpad=.4,columnspacing=2)
        ax.grid(axis='x',color='#e5e8eb',linewidth=.6)
        ax.tick_params(axis='y',length=0)
        fig.savefig(OUT/'study_regions_coverage.pdf',facecolor='white')
        fig.savefig(OUT/'study_regions_coverage.png',dpi=350,facecolor='white')
        plt.close(fig)
    (OUT/'SOURCE.json').write_text(json.dumps({'source_paths':[str(HB.relative_to(ROOT)),str(HN.relative_to(ROOT))],
        'coordinates':COORDS,'coordinate_source':'current main.tex station-registration values; precision from paper/figures/plot_fig1_study_regions_coverage.py',
        'display':'Longitude/latitude locator panels without administrative boundaries; each lower tick is one actual usable station date, not interpolated coverage.',
        'exclusion':'Existing Sili 2024-09-05 station-day policy applied; 37 validation rows excluded.',
        'test_payloads_read':0},indent=2),encoding='utf-8')
    print(stats.to_string(index=False))


if __name__=='__main__':
    main()
