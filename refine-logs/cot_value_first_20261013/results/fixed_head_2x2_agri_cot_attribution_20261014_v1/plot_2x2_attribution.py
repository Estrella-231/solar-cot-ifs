from pathlib import Path
import json
import matplotlib.pyplot as plt
p=Path(__file__).parent;s=json.loads((p/'fp64_independent_audit.json').read_text())
items=[('AGRI forecast effect\n(COT held at same-time R)', 'AGRI_effect_at_same_time_COT'),('COT forecast effect\n(AGRI held real)', 'COT_effect_at_true_AGRI'),('COT forecast effect\n(AGRI predicted)', 'COT_effect_at_forecast_AGRI'),('Joint forecast inputs\n(FF vs TT)', 'joint_forecast_input_effect')]
colors=['#0072B2','#D55E00','#009E73','#CC79A7']
fig,ax=plt.subplots(figsize=(8.4,4.8),layout='constrained')
for y,(label,key),color in zip(range(len(items)-1,-1,-1),items,colors):
 v=s['station_equal_delta_RMSE_by_contrast'][key];lo,hi=s['date_block_paired_bootstrap_95CI'][key]
 ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',color=color,ecolor=color,capsize=4,markersize=6,linewidth=1.6)
 ax.text(hi+.6,y,f'{v:+.1f} [{lo:+.1f}, {hi:+.1f}]',va='center',fontsize=8.5,color='#333333')
ax.axvline(0,color='#333333',linewidth=1,linestyle='--')
ax.set_yticks(range(len(items)),[x[0] for x in reversed(items)])
ax.set_xlabel('Change in station-equal mean GHI RMSE (W m⁻²)')
ax.set_title('Fixed-head 2×2 attribution: AGRI and COT forecast inputs')
ax.grid(axis='x',color='#dddddd',linewidth=.6)
ax.set_xlim(min(-6,min(s['date_block_paired_bootstrap_95CI'][k][0] for _,k in items)-2),max(s['date_block_paired_bootstrap_95CI'][k][1] for _,k in items)+12)
ax.text(.99,.80,'95% paired percentile CI · 5,000 BJT date-block resamples · 8 dates\nOverlapping forecast windows retained; fixed head, not a trained forecast model',transform=ax.transAxes,ha='right',va='top',fontsize=8.2,color='#444444')
fig.savefig(p/'fixed_head_2x2_attribution.png',dpi=220,facecolor='white')
fig.savefig(p/'fixed_head_2x2_attribution.svg',facecolor='white')
print('saved',p/'fixed_head_2x2_attribution.png')


