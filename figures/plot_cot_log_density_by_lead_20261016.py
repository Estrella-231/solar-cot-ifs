"""Lead-wise log-axis COT density, observed and forecast AGRI paired to CPP."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'refine-logs/cot_value_first_20261013/results/forecast_cot_val_bank_20261014_v1'
OUT=ROOT/'figures/cot_log_density_by_lead_20261016'

def main():
    if (SRC/'INVALID_CPP_JOIN.md').exists():
        raise RuntimeError('CPP reference join invalid; see audits/cpp_reference_join_20261016')
    path=SRC/'forecast_cot_val_bank.npz'
    receipt=json.loads((SRC/'forecast_cot_metrics_fp64.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['bank_npz_sha256']
    with np.load(path,allow_pickle=False) as z:
        mask=z['cpp_valid_mask'].astype(bool);ref=z['cpp_reference_cot'].astype(np.float64)
        arrays={key:np.expm1(z[key].astype(np.float64)) for key in ('target_agri_r_log1p_cot','forecast_log1p_cot')}
    edges=np.logspace(-1,2,71);delta=3/70
    panels={};stats=[]
    for si in range(2):
        for ri,(key,mode) in enumerate((('target_agri_r_log1p_cot','observed'),('forecast_log1p_cot','forecast'))):
            for ci,hour in enumerate((1,2,3,4)):
                li=4*hour-1;q=mask[:,si,li]
                x=ref[:,si,li][q];y=arrays[key][:,si,li][q]
                assert np.isfinite(x).all() and np.isfinite(y).all()
                err=y-x
                shown=(x>=.1)&(y>=.1)&(x<=100)&(y<=100)
                hist,_,_=np.histogram2d(x[shown],y[shown],bins=(edges,edges))
                # Density per unit area in log10 coordinates, conditional on shown support.
                density=hist/(shown.sum()*delta**2)
                st=dict(station=('sili','zhujia')[si],mode=mode,lead_hours=hour,
                        n_all=int(x.size),n_displayed=int(shown.sum()),
                        omitted_from_log_display_pct=float(100*np.mean(~shown)),
                        reference_below_0p1=int((x<.1).sum()),prediction_below_0p1=int((y<.1).sum()),
                        rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.mean(np.abs(err))),
                        mbe=float(np.mean(err)),pearson_r=float(np.corrcoef(x,y)[0,1]))
                stats.append(st);panels[(si,ri,ci)]=(density,st)
    vmax=max(a[0].max() for a in panels.values())
    cmap=LinearSegmentedColormap.from_list('paper_density',['#f7fcff','#d6effb','#73c8eb','#2585d4','#087bb4','#00a582','#08764c'])
    cmap.set_bad('white')
    OUT.mkdir(exist_ok=True)
    with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
                         'font.size':10,'pdf.fonttype':42}):
        for si,station in enumerate(('四里','竺家')):
            fig,axes=plt.subplots(2,4,figsize=(13.2,7.9))
            fig.subplots_adjust(left=.095,right=.89,top=.88,bottom=.19,wspace=.23,hspace=.33)
            for ri in range(2):
                for ci,hour in enumerate((1,2,3,4)):
                    ax=axes[ri,ci];density,st=panels[(si,ri,ci)]
                    im=ax.pcolormesh(edges,edges,np.ma.masked_equal(density.T,0),cmap=cmap,
                                     norm=Normalize(0,vmax),rasterized=True)
                    ax.plot([.1,100],[.1,100],color='#555555',ls='--',lw=.7)
                    ax.set(xscale='log',yscale='log',xlim=(.1,100),ylim=(.1,100),aspect='equal')
                    ax.tick_params(which='minor',bottom=False,left=False)
                    if ri==0:ax.set_title(f'Lead Time: {hour}h',fontsize=12,fontweight='bold',pad=10)
                    if ci==0:ax.set_ylabel('真实 AGRI→R 的 COT' if ri==0 else 'SimVP 预测 AGRI→R 的 COT',fontsize=10)
                    if ri==1:ax.set_xlabel('CPP 参考 COT')
                    ax.text(.97,.035,f"R: {st['pearson_r']:.2f}\nMBE: {st['mbe']:.2f}\nMAE: {st['mae']:.2f}\nRMSE: {st['rmse']:.2f}",
                            ha='right',va='bottom',transform=ax.transAxes,fontsize=9,
                            bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1))
                    ax.text(.03,.97,f"N={st['n_all']:,}\n近零未显示 {st['omitted_from_log_display_pct']:.1f}%",
                            ha='left',va='top',transform=ax.transAxes,fontsize=7.5,
                            bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1))
            cax=fig.add_axes([.915,.275,.015,.51])
            fig.colorbar(im,cax=cax,label='log10 坐标空间概率密度（显示范围内归一化）')
            fig.suptitle(f'{station}：真实图像与 SimVP 预测图像的 COT 反演',fontsize=14,y=.97)
            fig.text(.5,.065,'同一新月度验证集、CPP参考和有效掩膜；16×16场逐像元配对，各列为准确的+1/+2/+3/+4h。\n对数显示范围0.1～100；任一轴COT<0.1的点仅在图中省略，比例已标；统计量包含全部有效点。\n271个重叠窗口，CPP支持8个目标日期；概率密度、坐标及色标在两站/两种输入/四时效间统一。',
                     ha='center',fontsize=9)
            for ext in ('png','pdf'):fig.savefig(OUT/f'{("sili","zhujia")[si]}_cot_log_density.{ext}',dpi=280)
            plt.close(fig)
    result=dict(source=str(path),source_sha256=receipt['bank_npz_sha256'],bin_edges=edges.tolist(),
                density='histogram count / displayed N / squared log10 bin width; no smoothing or KDE',
                color_limits=[0,float(vmax)],metrics_support='all CPP-valid pairs including zero',
                display_support='both physical COT in [0.1,100]',test_read=0,statistics=stats)
    (OUT/'metrics_and_display_contract.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(OUT)

if __name__=='__main__':main()
