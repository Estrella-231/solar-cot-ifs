"""Matched COT density: R(forecast AGRI) or offline R(observed AGRI)."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'refine-logs/cot_value_first_20261013/results/forecast_cot_val_bank_20261014_v1'
OUT=ROOT/'figures/simvp_cot_density_20261016'

def main(mode='forecast'):
    if (SRC/'INVALID_CPP_JOIN.md').exists():
        raise RuntimeError('CPP reference join invalid; use audits/cpp_reference_join_20261016 repaired data and verified R validation')
    observed=mode=='observed'
    out=ROOT/'figures/observed_agri_cot_density_20261016' if observed else OUT
    prefix='observed_agri_cot_density' if observed else 'simvp_cot_density'
    key='target_agri_r_log1p_cot' if observed else 'forecast_log1p_cot'
    metric_key='offline_true_future_AGRI_to_R_rmse_cot' if observed else 'forecast_simvp_to_R_rmse_cot'
    axis_label='真实目标时刻 AGRI→R 的 COT' if observed else 'SimVP 预测 AGRI→R 的 COT'
    title='真实 AGRI 反演 COT：与 CPP 参考的散点密度' if observed else 'SimVP 预测云图反演 COT：与 CPP 参考的散点密度'
    source=SRC/'forecast_cot_val_bank.npz'
    receipt=json.loads((SRC/'forecast_cot_metrics_fp64.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==receipt['bank_npz_sha256']
    with np.load(source,allow_pickle=False) as z:
        mask=z['cpp_valid_mask'].astype(bool)
        ref=z['cpp_reference_cot'].astype(np.float64)
        pred=np.expm1(z[key].astype(np.float64))
        times=z['target_time_utc'].copy()
    assert ref.shape==pred.shape==mask.shape==(271,2,16,16,16)
    assert mask.sum()==550400
    limit=int(np.ceil(max(ref[mask].max(),pred[mask].max())/10)*10)
    edges=np.arange(0,limit+1,1,dtype=float)
    panels=[]
    for si,name in enumerate(('四里','竺家')):
        q=mask[:,si];x=ref[:,si][q];y=pred[:,si][q]
        assert np.isfinite(x).all() and np.isfinite(y).all()
        e=y-x
        stat=dict(station=receipt['station_metrics'][si]['station'],n_pixel_lead_records=int(x.size),
                  rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(np.abs(e))),bias=float(np.mean(e)),
                  pearson_r=float(np.corrcoef(x,y)[0,1]),r2_sse_sst=float(1-np.sum(e**2)/np.sum((x-x.mean())**2)))
        assert abs(stat['rmse']-receipt['station_metrics'][si][metric_key])<1e-10
        hist,_,_=np.histogram2d(x,y,bins=(edges,edges))
        assert int(hist.sum())==x.size
        panels.append((name,hist,stat))
    # Match the forecast-image figure exactly; explicit overflow guard.
    norm=LogNorm(vmin=1,vmax=8522)
    assert max(p[1].max() for p in panels)<=8522
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('white')
    with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
                         'font.size':11,'pdf.fonttype':42}):
        fig,axes=plt.subplots(1,2,figsize=(11,5.8))
        fig.subplots_adjust(left=.08,right=.86,bottom=.24,top=.85,wspace=.18)
        for ax,(name,hist,s) in zip(axes,panels):
            im=ax.pcolormesh(edges,edges,np.ma.masked_equal(hist.T,0),cmap=cmap,norm=norm,rasterized=True)
            ax.plot([0,limit],[0,limit],ls='--',lw=1,color='#666666',label='1:1')
            ax.set(xlim=(0,limit),ylim=(0,limit),aspect='equal',title=name,
                   xlabel='CPP 参考 COT',ylabel=axis_label)
            ax.text(.04,.96,f"RMSE = {s['rmse']:.2f}\nMAE = {s['mae']:.2f}\nBias = {s['bias']:.2f}\nr = {s['pearson_r']:.3f}\nN = {s['n_pixel_lead_records']:,}",
                    transform=ax.transAxes,ha='left',va='top',fontsize=9,
                    bbox=dict(facecolor='white',alpha=.90,edgecolor='none'))
        cax=fig.add_axes([.89,.29,.018,.50])
        fig.colorbar(im,cax=cax,label='每个 1×1 COT 区间内的配对记录数（对数色标）')
        fig.suptitle(title,fontsize=14,y=.96)
        note='真实目标时刻 AGRI→冻结 R 的离线反演；与预测图相同样本、CPP掩膜、坐标和密度色标。' if observed else '新月度验证集；+15～+240 min 全时效；16×16 场逐像元配对，未做空间平均或随机抽点。'
        fig.text(.5,.09,note+'\n271个重叠预报窗口，CPP参考覆盖8个目标日期；N为像元—时效记录数，不是独立天气样本数。\n虚线为1:1；空白表示该区间无配对点；CPP为卫星检索参考。',ha='center',fontsize=9)
        out.mkdir(exist_ok=True)
        for ext in ('png','pdf'):fig.savefig(out/f'{prefix}.{ext}',dpi=280)
        plt.close(fig)
    np.savez_compressed(out/'density_counts.npz',edges=edges,sili_counts=panels[0][1],zhujia_counts=panels[1][1])
    meta=dict(source=str(source),source_sha256=receipt['bank_npz_sha256'],
              model_chain='observed target AGRI -> frozen R -> physical COT; offline only' if observed else 'historical AGRI -> SimVP forecast AGRI -> frozen R -> physical COT',
              target='CPP reference at matching station/target time/pixel; existing CPP-valid mask',
              physical_transform=f'one FP64 expm1 of {key}',bin_width_cot=1,
              axis_limits=[0,limit],density_scale='LogNorm count; shared across panels; zero-count bins white',
              samples='all saved common-support pixel-lead pairs; no downsampling or clipping',
              test_read=0,window_count=271,cpp_supported_target_dates=8,
              density_count_limits=[1,8522],statistics=[p[2] for p in panels])
    (out/'metrics_and_source.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta['statistics'],ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['forecast','observed'],default='forecast')
    main(parser.parse_args().mode)
