"""Actual observed AGRI/R-COT sequences and unchanged 15-minute GHI labels."""
import json
from datetime import datetime,timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/cot_interval_history_20261015';D=OUT/'diagnostics_attempt1'


def main():
    plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'axes.unicode_minus':False,'font.size':10,'pdf.fonttype':42})
    src=np.load(D/'actual_cloud_cases.npz');meta=json.loads((D/'case_source_manifest.json').read_text());cases=meta['cases'];frames=meta['frames'];figures=OUT/'figures';figures.mkdir(exist_ok=True)
    actual=np.load(OUT/'results_attempt1/validation_inputs.npz');alignment=[]
    with PdfPages(OUT/'actual_AGRI_COT_history_and_GHI.pdf') as pdf:
        for i,c in enumerate(cases):
            t=datetime.fromisoformat(c['utc'])+timedelta(hours=8);name='四里' if c['station']=='sili' else '竺家';kind='非零样本中最大改善案例' if 'improvement' in c['category'] else '非零样本中最大退化案例'
            fig=plt.figure(figsize=(13,11));gs=fig.add_gridspec(3,3,height_ratios=[1,1,1.1],left=.07,right=.89,bottom=.12,top=.90,hspace=.34,wspace=.22)
            for col,k in enumerate((30,15,0)):
                rec=next(r for r in frames if r['case']==i and r['offset_minutes']==-k);scan=t-timedelta(minutes=k)+timedelta(seconds=rec['scan_offset_seconds'])
                raw=src[f'c{i}_m{k}_rgb_channels'];rgb=np.clip(np.moveaxis(raw[[2,1,0]],0,-1)/.8,0,1)**(1/2.2)
                vi=c['validation_row_index'];ti=(30,15,0).index(k)
                np.testing.assert_array_equal(raw[:,7:10,7:10],actual['agri_raw'][vi,ti,:3])
                np.testing.assert_allclose(np.log1p(src[f'c{i}_m{k}_cot'][7:10,7:10]),actual['cot_log1p'][vi,ti,0],atol=1e-6,rtol=0)
                alignment.append(dict(case=i,offset_minutes=-k,AGRI_exact_equal=True,COT_log1p_max_abs_difference=float(np.max(np.abs(np.log1p(src[f'c{i}_m{k}_cot'][7:10,7:10])-actual['cot_log1p'][vi,ti,0])))))
                ax=fig.add_subplot(gs[0,col]);ax.imshow(rgb,interpolation='nearest');ax.set_title(f'真实AGRI：T−{k} min\n名义 {(t-timedelta(minutes=k)):%H:%M}；扫描 {scan:%H:%M:%S}')
                ax.plot(8,8,'*',mfc='white',mec='k',ms=10);ax.add_patch(Rectangle((6.5,6.5),3,3,fill=False,ec='#00FFFF',lw=1.5));ax.set_xticks([0,8,15]);ax.set_yticks([0,8,15])
                ax=fig.add_subplot(gs[1,col]);cot=src[f'c{i}_m{k}_cot'];im=ax.imshow(cot,vmin=0,vmax=60,cmap='magma',interpolation='nearest');ax.set_title(f'连续R-COT；图内最大值 {np.nanmax(cot):.1f}')
                ax.plot(8,8,'*',mfc='white',mec='k',ms=10);ax.add_patch(Rectangle((6.5,6.5),3,3,fill=False,ec='#00FFFF',lw=1.5));ax.set_xticks([0,8,15]);ax.set_yticks([0,8,15])
            cb=fig.add_axes([.91,.40,.015,.22]);fig.colorbar(im,cax=cb,label='R-COT（无量纲）',extend='max')
            ax=fig.add_subplot(gs[2,:]);native=[float(x) for x in c['raw_GHI_5min'].split('|')];nt=[t-timedelta(minutes=m) for m in (10,5,0)]
            ax.plot(nt,native,'ko-',label='原始5分钟区间均值（结束时刻）',ms=6);ax.axvspan(t-timedelta(minutes=15),t,color='#E69F00',alpha=.14)
            ax.hlines(c['truth'],t-timedelta(minutes=15),t,color='#000000',lw=2,linestyle=':',label=f'15分钟真值 {c["truth"]:.1f}')
            for key,label,color in [('A_seed_predictions','AGRI历史','#0072B2'),('AC_seed_predictions','AGRI+COT历史','#D55E00')]:
                vals=np.asarray(c[key]);ax.hlines(vals.mean(),t-timedelta(minutes=15),t,color=color,lw=2,label=f'{label}：三seed均值 {vals.mean():.1f}');ax.fill_between([t-timedelta(minutes=15),t],vals.min(),vals.max(),color=color,alpha=.15)
            ax.axvline(t+timedelta(seconds=next(r['scan_offset_seconds'] for r in frames if r['case']==i and r['offset_minutes']==0)),color='#009E73',ls='--',label='T帧实际扫描')
            ax.set_xlim(t-timedelta(minutes=17),t+timedelta(minutes=5));ax.set_xlabel('北京时间；标签区间为 (T−15 min,T]');ax.set_ylabel('GHI（W/m²）');ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'));ax.legend(fontsize=9,ncol=2);ax.grid(alpha=.2)
            fig.suptitle(f'{name} {t:%Y-%m-%d %H:%M}：{kind}',fontsize=15,y=.965)
            fig.text(.5,.035,'真实观测C03/C02/C01固定拉伸假彩色；COT为冻结R反演，并非CPP真值。16×16仅作云背景显示；青框3×3为GHI实际输入。\n模型读取三帧局地AGRI/几何，两组仅COT槽不同；下方色带为三seed范围，非置信区间。极端案例仅解释机制，不代表总体效果。\n这是同期区间反演；T图像扫描晚于T，不能称为T时可用的业务预报。',ha='center',fontsize=9)
            fig.savefig(figures/f'case{i}_{c["station"]}.png',dpi=150);pdf.savefig(fig);plt.close(fig)
    (OUT/'case_input_alignment.json').write_text(json.dumps(dict(state='PASS_FIGURE_CROPS_MATCH_ACTUAL_GHI_INPUTS',checks=alignment),indent=2))
    print('FOUR_ACTUAL_CLOUD_SEQUENCE_FIGURES_COMPLETE')


if __name__=='__main__':main()
