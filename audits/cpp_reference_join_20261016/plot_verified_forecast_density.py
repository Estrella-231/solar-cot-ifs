"""Forecast-only COT densities using the repaired, key-verified CPP reference."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'verified_forecast_figures'

def panel(x,y,edges):
    assert np.isfinite(x).all() and np.isfinite(y).all()
    err=y-x;shown=(x>=.1)&(y>=.1)&(x<=100)&(y<=100)
    counts,_,_=np.histogram2d(x[shown],y[shown],bins=(edges,edges))
    assert counts.sum()==shown.sum()
    density=counts/(shown.sum()*(3/70)**2)
    stats=dict(n=int(x.size),n_displayed=int(shown.sum()),omitted_pct=float(100*np.mean(~shown)),
               rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.mean(abs(err))),
               mbe=float(np.mean(err)),pearson_r=float(np.corrcoef(x,y)[0,1]))
    return density,stats

def main():
    audit=json.loads((HERE/'AUDIT.json').read_text(encoding='utf-8'))
    path=HERE/'repaired_existing_271_windows/forecast_cot_val_bank.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['new_bank_sha256']
    with np.load(path,allow_pickle=False) as z:
        ref=z['cpp_reference_cot'].astype(float);mask=z['cpp_valid_mask'].astype(bool)
        pred=np.expm1(z['forecast_log1p_cot'].astype(float));times=z['target_time_utc'].astype(str)
    assert ref.shape==mask.shape==pred.shape==(271,2,16,16,16)
    assert int(mask.sum())==audit['new_valid_pixel_records']==668672
    valid_wl=mask.any(axis=(1,3,4))
    dates=sorted({(dt.datetime.strptime(t,'%Y-%m-%d %H:%M:%S')+dt.timedelta(hours=8)).date().isoformat() for t in times[valid_wl]})
    edges=np.logspace(-1,2,71);overall={};hourly={};all_leads=[]
    for si,sta in enumerate(('sili','zhujia')):
        q=mask[:,si];overall[si]=panel(ref[:,si][q],pred[:,si][q],edges)
        expected=next(r['corrected'] for r in audit['paired_metrics'] if r['station']==sta and r['mode']=='SimVP forecast AGRI -> R')
        assert abs(overall[si][1]['rmse']-expected['rmse'])<1e-10
        for li in range(16):
            q=mask[:,si,li];p=panel(ref[:,si,li][q],pred[:,si,li][q],edges)
            all_leads.append(dict(station=sta,lead_minutes=15*(li+1),**p[1]))
            if (li+1)%4==0:hourly[si,(li+1)//4]=p
    vmax=max(p[0].max() for p in [*overall.values(),*hourly.values()])
    cmap=LinearSegmentedColormap.from_list('verified_forecast_density',['#f7fcff','#b5e3f5','#47afe4','#147cce','#009e83','#08764c']);cmap.set_bad('white')
    norm=Normalize(0,vmax)
    def draw(ax,p,title):
        h,s=p
        im=ax.pcolormesh(edges,edges,np.ma.masked_equal(h.T,0),cmap=cmap,norm=norm,rasterized=True)
        ax.plot([.1,100],[.1,100],ls='--',lw=.8,color='#666666')
        ax.set(xscale='log',yscale='log',xlim=(.1,100),ylim=(.1,100),aspect='equal',title=title)
        ax.tick_params(which='minor',bottom=False,left=False)
        ax.text(.96,.04,f"R: {s['pearson_r']:.3f}\nMBE: {s['mbe']:.2f}\nMAE: {s['mae']:.2f}\nRMSE: {s['rmse']:.2f}",
                transform=ax.transAxes,ha='right',va='bottom',fontsize=9,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
        ax.text(.03,.97,f"N={s['n']:,}\n近零未显示 {s['omitted_pct']:.1f}%",transform=ax.transAxes,va='top',fontsize=7.5)
        return im
    foot=f'已按站点、目标时刻及像元修正CPP配对；仅已有271个预报窗口子集，CPP支持{len(dates)}个目标日期。\n16×16场逐像元评分；对数显示0.1～100，近零点未显示比例已标，全部有效点均参与物理单位指标计算。'
    OUT.mkdir(exist_ok=True)
    with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'font.size':10,'pdf.fonttype':42}):
        fig,axes=plt.subplots(1,2,figsize=(10.6,5.7))
        fig.subplots_adjust(left=.09,right=.86,bottom=.23,top=.83,wspace=.20)
        for si,(ax,name) in enumerate(zip(axes,('四里','竺家'))):
            im=draw(ax,overall[si],name);ax.set_xlabel('CPP 参考 COT');ax.set_ylabel('SimVP 预测 AGRI→R 的 COT')
        cax=fig.add_axes([.89,.29,.016,.45]);fig.colorbar(im,cax=cax,label='log10 坐标空间概率密度')
        fig.suptitle('未来预报 COT：SimVP 预测 AGRI → 冻结 R',fontsize=14,y=.96)
        fig.text(.5,.07,'+15～+240 min 全部时效汇总。'+foot,ha='center',fontsize=9)
        for ext in ('png','pdf'):fig.savefig(OUT/f'forecast_density_overall.{ext}',dpi=280)
        plt.close(fig)
        fig,axes=plt.subplots(2,4,figsize=(13.2,7.8))
        fig.subplots_adjust(left=.095,right=.89,bottom=.19,top=.88,wspace=.23,hspace=.28)
        for si,name in enumerate(('四里','竺家')):
            for ci,hour in enumerate((1,2,3,4)):
                ax=axes[si,ci];im=draw(ax,hourly[si,hour],f'Lead Time: {hour}h' if si==0 else '')
                if ci==0:ax.set_ylabel(f'{name}：预测 COT')
                if si==1:ax.set_xlabel('CPP 参考 COT')
        cax=fig.add_axes([.915,.27,.015,.51]);fig.colorbar(im,cax=cax,label='log10 坐标空间概率密度')
        fig.suptitle('SimVP 预测云图反演 COT：逐时效散点密度',fontsize=14,y=.97)
        fig.text(.5,.06,foot+'\n两站、各时效及汇总图使用相同坐标与概率密度色标；CPP为卫星检索参考。',ha='center',fontsize=9)
        for ext in ('png','pdf'):fig.savefig(OUT/f'forecast_density_by_lead.{ext}',dpi=280)
        plt.close(fig)
    result=dict(source=str(path),source_sha256=audit['new_bank_sha256'],model_outputs_unchanged=True,
                scope='existing 271 windows only, not re-enumerated full validation coverage',cpp_target_dates=dates,
                density='log10-coordinate histogram density, no smoothing',color_limits=[0,float(vmax)],
                bin_edges=edges.tolist(),overall={('sili','zhujia')[si]:p[1] for si,p in overall.items()},
                lead_metrics=all_leads,test_read=0)
    (OUT/'metrics_and_source.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(target_dates=len(dates),overall=result['overall'],hourly=[r for r in all_leads if r['lead_minutes']%60==0]),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
