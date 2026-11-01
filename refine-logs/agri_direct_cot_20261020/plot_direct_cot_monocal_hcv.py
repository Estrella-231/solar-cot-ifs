"""Make a compact, auditable summary figure from fixed monocal H-CV scores."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

COL={'none':'#6C7A89','pxc_raw':'#0072B2','pxc_cal':'#D55E00'}
LABEL={'none':'No COT','pxc_raw':'Raw predicted COT','pxc_cal':'Monotone-calibrated COT'}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--metrics',type=Path,required=True); p.add_argument('--outdir',type=Path,required=True); a=p.parse_args()
    m=json.loads(a.metrics.read_text(encoding='utf-8'))
    if m.get('state')!='COMPLETE_COT_MONOCAL_H_CV_SCORE': raise RuntimeError('complete score required')
    a.outdir.mkdir(parents=True,exist_ok=True); r=m['results']
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    ax=axs[0,0]; labels=['All leads','+15–60 min']; comps=[('cal_minus_raw','Cal − raw'),('cal_minus_none','Cal − no COT')]
    x=np.arange(2); width=.32
    for j,(key,lab) in enumerate(comps):
        vals=[r[key]['all_leads' if i==0 else 'plus15_60'] for i in range(2)]; y=np.array([v['delta_rmse_wm2'] for v in vals]); lo=np.array([v['ci95_low'] for v in vals]); hi=np.array([v['ci95_high'] for v in vals])
        xpos=x+(j-.5)*width; ax.errorbar(xpos,y,yerr=[y-lo,hi-y],fmt='o',capsize=3,lw=1.4,color=('#D55E00' if j==0 else '#009E73'),label=lab)
    ax.axhline(0,color='black',lw=1); ax.set_xticks(x,labels); ax.set_ylabel('Paired ΔRMSE (W m$^{-2}$)'); ax.set_title('Date-block paired uncertainty'); ax.legend(frameon=False,fontsize=8)
    ax=axs[0,1]; xpos=np.arange(2); offsets=np.array([-.22,0,.22])
    for j,arm in enumerate(('none','pxc_raw','pxc_cal')):
        vals=[r['absolute_metrics'][arm]['stations'][st]['all_leads']['rmse_wm2'] for st in ('sili','zhujia')]
        ax.plot(xpos+offsets[j],vals,marker='o',lw=1.5,color=COL[arm],label=LABEL[arm])
    ax.set_xticks(xpos,['Sili','Zhujia']); ax.set_ylabel('RMSE (W m$^{-2}$)'); ax.set_title('Station-equal full-horizon RMSE'); ax.legend(frameon=False,fontsize=8)
    ax=axs[1,0]; leads=np.array([q['lead_minutes'] for q in r['leadwise_cal_minus_raw']]); vals=np.array([q['delta_rmse_wm2'] for q in r['leadwise_cal_minus_raw']])
    ax.axhline(0,color='black',lw=1); ax.plot(leads,vals,'o-',color='#D55E00',lw=1.4,ms=4); ax.set_xticks(leads[::2]); ax.set_xlabel('Forecast lead (min)'); ax.set_ylabel('Calibrated − raw ΔRMSE (W m$^{-2}$)'); ax.set_title('Lead-wise paired point estimates')
    ax=axs[1,1]; q=r['learned_slope_summary']; k=np.asarray(q['knots_log1p']); mean=np.asarray(q['mean']); sd=np.asarray(q['sd']); ax.errorbar(k,mean,yerr=sd,fmt='o-',capsize=3,color='#009E73',label='Mean ± SD across 30 fitted maps'); ax.axhline(1,color='black',lw=1,ls='--'); ax.set_xlabel('log(1 + COT) knot'); ax.set_ylabel('Local monotone slope'); ax.set_title('Learned calibration vs identity'); ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Forecast COT monotone calibration: H-only date-block CV',fontsize=12)
    for ext in ('png','pdf'):
        fig.savefig(a.outdir/f'monocal_hcv_summary.{ext}',dpi=300,facecolor='white')
    (a.outdir/'ALT_TEXT.txt').write_text('Four-panel summary: paired full and short-lead GHI RMSE differences with 95% two-day date-block bootstrap intervals; station-wise absolute RMSE for no-COT, raw predicted COT, and monotone-calibrated COT; per-lead calibrated-minus-raw RMSE; and mean fitted spline slopes with standard deviation across 30 models. Negative paired differences indicate lower error. This is H-internal screening, not independent test evidence.\n',encoding='utf-8')
    print(json.dumps({'figure_dir':str(a.outdir),'png_bytes':(a.outdir/'monocal_hcv_summary.png').stat().st_size,'pdf_bytes':(a.outdir/'monocal_hcv_summary.pdf').stat().st_size}))
if __name__=='__main__': main()
