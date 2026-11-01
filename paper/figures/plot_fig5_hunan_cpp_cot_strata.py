from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
METRICS = ROOT / 'refine-logs/cot_value_first_20261013/results/cot_thickness_story_20261014/thickness_metrics.json'
OUTDIR = Path(__file__).resolve().parent
D = json.loads(METRICS.read_text(encoding='utf-8'))
GROUPS = [g for g in D['groups'] if g['source'] == 'CPP_reference_centre' and g['bin'] not in ('supported',)]
ORDER = ['nearzero_[0,0.1)', 'thin_[0.1,5)', 'moderate_[5,30)', 'thick_[30,100]']
GROUPS.sort(key=lambda g: ORDER.index(g['bin']))
LABELS = ['Near-zero\n[0, 0.1)', 'Thin\n[0.1, 5)', 'Moderate\n[5, 30)', 'Thick\n[30, 100]']
rows=[]
for g in GROUPS:
    rows.append({
        'bin': g['bin'], 'n': g['n'], 'BJT_dates': g['BJT_dates'],
        'A_RMSE_Wm2': g['RMSE_Wm2']['A'],
        'AC_RMSE_Wm2': g['RMSE_Wm2']['AC'],
        'AC_minus_A_Wm2': g['AC_minus_A_RMSE_Wm2'],
        'CI95_low_Wm2': g['AC_minus_A_95CI_Wm2'][0],
        'CI95_high_Wm2': g['AC_minus_A_95CI_Wm2'][1],
    })
CSV=OUTDIR/'fig5_hunan_cpp_cot_strata.csv'
pd.DataFrame(rows).to_csv(CSV,index=False,float_format='%.6f')

plt.rcParams.update({
    'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':9,'axes.labelsize':9,'axes.titlesize':10,
    'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
})
fig,(ax0,ax1)=plt.subplots(1,2,figsize=(7.2,3.5),gridspec_kw={'width_ratios':[1.2,1]},layout='constrained')
colors={'A':'#586f7c','AC':'#008b79'}
y=list(range(4))[::-1]
for yi,g,label in zip(y,GROUPS,LABELS):
    a=float(g['RMSE_Wm2']['A']); ac=float(g['RMSE_Wm2']['AC'])
    ax0.plot([a,ac],[yi,yi],color='#aab2b5',lw=1.2,zorder=1)
    ax0.scatter(a,yi,s=34,c=colors['A'],marker='o',edgecolors='white',linewidths=.5,zorder=3,label='AGRI baseline' if yi==y[0] else None)
    ax0.scatter(ac,yi,s=36,c=colors['AC'],marker='D',edgecolors='white',linewidths=.5,zorder=3,label='AGRI + COT' if yi==y[0] else None)
    ax0.text(184,yi,f"n={g['n']}; {g['BJT_dates']} dates",va='center',ha='left',fontsize=7,color='#41494d')
    delta=float(g['AC_minus_A_RMSE_Wm2']); lo,hi=map(float,g['AC_minus_A_95CI_Wm2'])
    ax1.errorbar(delta,yi,xerr=[[delta-lo],[hi-delta]],fmt='o',ms=4.5,color='#007f6e',ecolor='#007f6e',capsize=2.5,lw=1.3)
    ax1.text(hi+1.0,yi,f'{delta:.1f} [{lo:.1f}, {hi:.1f}]',va='center',ha='left',fontsize=7)
ax0.set_yticks(y,LABELS); ax1.set_yticks(y,LABELS)
ax0.set_xlim(45,235); ax0.set_xlabel('Station-equal RMSE (W m$^{-2}$)'); ax0.set_title('(a) GHI error within CPP-COT strata',loc='left')
ax0.grid(axis='x',color='#d9dddf',lw=.6); ax0.set_axisbelow(True); ax0.spines[['top','right']].set_visible(False)
ax0.legend(loc='upper left',frameon=False)
ax1.axvline(0,color='#363d40',lw=.9); ax1.set_xlim(-26,13); ax1.set_xlabel('AGRI + COT minus AGRI baseline (W m$^{-2}$)'); ax1.set_title('(b) Paired date-block difference',loc='left')
ax1.grid(axis='x',color='#d9dddf',lw=.6); ax1.set_axisbelow(True); ax1.spines[['top','right']].set_visible(False)
fig.savefig(OUTDIR/'fig5_hunan_cpp_cot_strata.pdf',facecolor='white')
fig.savefig(OUTDIR/'fig5_hunan_cpp_cot_strata.png',dpi=400,facecolor='white')
fig.savefig(OUTDIR/'fig5_hunan_cpp_cot_strata.svg',facecolor='white')
plt.close(fig)
print(f'Wrote {CSV.name} and three figure formats from {METRICS}')


