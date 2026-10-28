"""Visualize fixed actual CPP case, conditional geometry scenarios, no scoring."""
import hashlib,json
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

P=Path(__file__).resolve().parent
data=json.loads((P/'ACTUAL_SHADOW_SCENARIOS.json').read_text())
centers={'Niushou':(111,124),'Wangzhai':(112,126),'Mizhuang':(110,128)}
times=sorted({r['time_bjt'] for r in data['records'] if r['status']=='geometric_nearest_cloud_pixel'})
assert len(times)==7
styles={'nominal_geolocation':('#0072B2','o','Nominal coordinates'),
        'already_corrected_geolocation':('#D55E00','s','Already corrected')}
fig,axs=plt.subplots(2,3,figsize=(11,7),layout='constrained')
for col,(station,(cy,cx)) in enumerate(centers.items()):
    ax=axs[0,col];ax.add_patch(Rectangle((-1.5,-1.5),3,3,fill=False,lw=1.1,ls='--',ec='.4'))
    ax.plot(0,0,'k*',ms=10,label='Station')
    for scenario,(color,marker,label) in styles.items():
        rows=sorted([r for r in data['records'] if r.get('station')==station and r.get('scenario')==scenario],key=lambda x:x['time_bjt'])
        assert [r['time_bjt'] for r in rows]==times
        for r in rows:
            z=np.load(P/'height_extract_attempt1'/(r['time_bjt']+'.npz'))
            assert np.isclose(z['COT_raw'][r['source_row'],r['source_col']],r['source_COT'],rtol=0,atol=0)
            assert np.isclose(z['COT_raw'][cy,cx],r['station_COT'],rtol=0,atol=0)
        ax.plot([r['col_offset'] for r in rows],[r['row_offset'] for r in rows],marker=marker,color=color,alpha=.8,lw=1,label=label)
        xx=[datetime.strptime(r['time_bjt'],'%Y%m%d%H%M') for r in rows]
        axs[1,col].plot(xx,[r['source_COT']-r['station_COT'] for r in rows],marker=marker,color=color,lw=1.4,label=label)
    ax.set(title=station,xlim=(-3.5,3.5),ylim=(3.5,-3.5),xlabel='Column offset (pixels)',ylabel='Row offset (pixels)');ax.set_aspect('equal');ax.grid(alpha=.2)
    axs[1,col].axhline(0,c='.3',lw=.8);axs[1,col].grid(alpha=.2);axs[1,col].set(ylabel='Source − station COT',xlabel='BJT, 2025-08-06',ylim=(-2,2))
    axs[1,col].set_xticks([datetime.strptime(times[0],'%Y%m%d%H%M'),datetime.strptime(times[3],'%Y%m%d%H%M'),datetime.strptime(times[-1],'%Y%m%d%H%M')],['10:30','11:15','12:00'])
axs[0,0].legend(loc='lower left',fontsize=7)
fig.suptitle('One observed cloud event: conditional nearest-shadow-source geometry',fontsize=13)
fig.supxlabel('7 valid history frames; 10:15 missing. CPP quality and geolocation provenance unresolved. No GHI scoring.',fontsize=9)
for ext in ['png','pdf']:fig.savefig(P/f'actual_shadow_scenarios.{ext}',dpi=180)
plt.close(fig)
(P/'SHADOW_FIGURE_MANIFEST.json').write_text(json.dumps({'source_sha256':hashlib.sha256((P/'ACTUAL_SHADOW_SCENARIOS.json').read_bytes()).hexdigest(),
    'figure':['actual_shadow_scenarios.png','actual_shadow_scenarios.pdf'],'matplotlib':matplotlib.__version__,'n_independent_days':1,
    'interpretation':'conditional geometric candidates; not prediction accuracy or physical shadow footprint'},indent=2))
print('PLOT_COMPLETE')
