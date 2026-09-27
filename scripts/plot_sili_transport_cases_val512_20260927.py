from pathlib import Path
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(r'solar-cot-ifs/results/regional_station_cot_20260927/sili_transport_variants_val512')
cases=json.loads((p/'sequence_events.json').read_text())
byinit={x['init']:x for x in cases}
chosen=[
 ('Cloud growth','2025-07-14T10:30:00','CPP rises from near zero to thick cloud; translation cannot create new cloud'),
 ('Thick-cloud passage and decay','2025-07-09T14:00:00','Large estimated local shift, followed by a rapid decrease in reference COT'),
 ('Moving thick-cloud peak','2025-08-05T14:30:00','Large local motion estimate and very thick reference peak'),
 ('Relatively accurate forecast','2025-08-19T07:45:00','Lower SimVP station RMSE and no severe thick-cloud miss')]
colors={'cpp_reference':'#222222','simvp':'#d1495b','local32_med7':'#00798c','persistence':'#777777'}
labels={'cpp_reference':'CPP COT reference','simvp':'SimVP -> frozen R','local32_med7':'32x32 multi-frame transport','persistence':'Persistence'}
fig,axs=plt.subplots(2,2,figsize=(13,8),sharex=True,layout='constrained')
meta=[]
for ax,(kind,init,reason) in zip(axs.flat,chosen):
 x=byinit[init];tracks=x['station_tracks'];xx=np.arange(1,17)*15
 for n in ('cpp_reference','simvp','local32_med7','persistence'):
  yy=np.array([np.nan if v is None else v for v in tracks[n]],float)
  ax.plot(xx,yy,marker='o' if n=='cpp_reference' else None,lw=2 if n=='cpp_reference' else 1.6,ls='--' if n=='persistence' else '-',color=colors[n],label=labels[n])
 ax.axhline(30,color='#999999',lw=1,ls=':',label='Thick diagnostic threshold = 30' if ax is axs.flat[0] else None)
 ax.axhline(10,color='#bd7b00',lw=1,ls=':',label='Severe-miss threshold = 10' if ax is axs.flat[0] else None)
 ax.set_title(f'{kind} | {init.replace("T"," ")} BJT\n{reason}',fontsize=10)
 ax.set_xlim(15,240);ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.set_ylabel('Station-pixel COT');ax.set_xlabel('Forecast lead (min)')
 m=x['motion']['local32'];mag=math.hypot(*m)
 ax.text(.98,.95,f'CPP peak {x["cpp_peak"]:.1f} at +{x["cpp_peak_lead_min"]} min\nSimVP RMSE {x["site_rmse_simvp"]:.1f} | thick misses {x["thick_miss_leads"]}\nLocal shift ({m[0]:+.2f}, {m[1]:+.2f}) px/15 min | norm {mag:.2f}',transform=ax.transAxes,ha='right',va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.88,edgecolor='#cccccc'))
 meta.append({'category':kind,'init_BJT':init,'sequence_index':x['sequence'],'selection_reason':reason,'cpp_peak':x['cpp_peak'],'cpp_peak_lead_min':x['cpp_peak_lead_min'],'simvp_site_rmse':x['site_rmse_simvp'],'simvp_thick_miss_leads':x['thick_miss_leads'],'local32_motion_row_col_per_15min':m})
axs.flat[0].legend(loc='upper left',fontsize=8)
fig.suptitle('Sili station: real validation COT tracks from the fixed 512-sequence cohort\nCPP is a retrieval reference, not independent truth. Translation moves historical COT only; it cannot create, thicken, or dissipate clouds.',fontsize=13)
fig.savefig(p.parent/'sili_transport_case_tracks_val512_20260927.png',dpi=180)
fig.savefig(p.parent/'sili_transport_case_tracks_val512_20260927.pdf')
(p.parent/'sili_transport_case_tracks_val512_20260927.json').write_text(json.dumps(meta,indent=2))
print('figure written',p.parent/'sili_transport_case_tracks_val512_20260927.png')
