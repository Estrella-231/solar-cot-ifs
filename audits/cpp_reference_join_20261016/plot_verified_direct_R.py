"""Plot the original correctly paired full R validation predictions."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
import numpy as np

HERE=Path(__file__).resolve().parent

def main():
    with (HERE/'sources/rows.csv').open(encoding='utf-8') as f:sup=[r for r in csv.DictReader(f) if r['state']=='SUPERVISED']
    with np.load(HERE/'sources/validation_predictions.npz') as z:
        ids=z['sample_indices'];ref=z['reference_cot'].astype(float);pred=np.expm1(z['pred_log1p_cot'].astype(float));mask=z['mask'].astype(bool)
    panels=[];edges=np.logspace(-1,2,71);delta=3/70
    for name,label in [('sili','四里'),('zhujia','竺家')]:
        select=np.array([sup[int(i)]['station']==name for i in ids]);q=mask[select]
        x=ref[select][q];y=pred[select][q];e=y-x
        shown=(x>=.1)&(y>=.1)&(x<=100)&(y<=100)
        h,_,_=np.histogram2d(x[shown],y[shown],bins=(edges,edges));h/=shown.sum()*delta**2
        s=dict(station=name,n=int(len(x)),rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),
               bias=float(np.mean(e)),r=float(np.corrcoef(x,y)[0,1]),omitted_display_pct=float(100*np.mean(~shown)))
        panels.append((label,h,s))
    cmap=LinearSegmentedColormap.from_list('verified_density',['#f7fcff','#b5e3f5','#47afe4','#147cce','#009e83','#08764c']);cmap.set_bad('white')
    norm=Normalize(0,max(h.max() for _,h,_ in panels))
    with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'font.size':11,'pdf.fonttype':42}):
        fig,axes=plt.subplots(1,2,figsize=(10.6,5.6))
        fig.subplots_adjust(left=.09,right=.86,bottom=.23,top=.83,wspace=.20)
        for ax,(label,h,s) in zip(axes,panels):
            im=ax.pcolormesh(edges,edges,np.ma.masked_equal(h.T,0),cmap=cmap,norm=norm,rasterized=True)
            ax.plot([.1,100],[.1,100],ls='--',lw=.8,color='#666666')
            ax.set(xscale='log',yscale='log',xlim=(.1,100),ylim=(.1,100),aspect='equal',
                   title=label,xlabel='CPP 参考 COT',ylabel='真实 AGRI 反演 COT')
            ax.text(.96,.04,f"R: {s['r']:.3f}\nMBE: {s['bias']:.2f}\nMAE: {s['mae']:.2f}\nRMSE: {s['rmse']:.2f}",
                    transform=ax.transAxes,ha='right',va='bottom',fontsize=10,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
            ax.text(.03,.97,f"N={s['n']:,}\n近零未显示 {s['omitted_display_pct']:.1f}%",transform=ax.transAxes,va='top',fontsize=8)
        cax=fig.add_axes([.89,.29,.016,.45]);fig.colorbar(im,cax=cax,label='log10 坐标空间概率密度')
        fig.suptitle('真实 AGRI 反演 COT：原始正确配对的完整 R 验证集',fontsize=14,y=.96)
        fig.text(.5,.075,'原始冻结 R 验证输出：1,398个站点时刻、356,169个有效像元；按保存的全表索引绑定站点。\n对数显示0.1～100；指标包含所有有效点（含0值）。本图为同期反演，不是未来预报。',ha='center',fontsize=9)
        for ext in ('png','pdf'):fig.savefig(HERE/f'verified_direct_R_density.{ext}',dpi=280)
        plt.close(fig)
    (HERE/'verified_direct_R_metrics.json').write_text(json.dumps([p[2] for p in panels],ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([p[2] for p in panels],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
