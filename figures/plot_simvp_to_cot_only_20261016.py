"""Only score SimVP-predicted AGRI -> frozen R COT against matched CPP."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'refine-logs/cot_value_first_20261013/results/forecast_cot_val_bank_20261014_v1'
OUT=ROOT/'figures/simvp_to_cot_only_20261016'

def main():
    if (SRC/'INVALID_CPP_JOIN.md').exists():
        raise RuntimeError('CPP reference join invalid; see audits/cpp_reference_join_20261016')
    with (SRC/'forecast_cot_lead_metrics_fp64.csv').open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    metrics=json.loads((SRC/'forecast_cot_metrics_fp64.json').read_text())
    OUT.mkdir(exist_ok=True)
    selected=[]
    for r in rows:
        selected.append(dict(station=r['station'],lead_minutes=int(r['lead_min']),
            valid_pixel_records=int(r['n_pixels']),
            rmse=float(r['forecast_simvp_to_R_rmse_cot']),
            mae=float(r['forecast_simvp_to_R_mae_cot']),
            bias=float(r['forecast_simvp_to_R_bias_cot'])))
    with (OUT/'metrics_by_lead.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
    with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
                         'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
        fig,ax=plt.subplots(figsize=(8.5,5.2))
        fig.subplots_adjust(left=.12,right=.97,top=.86,bottom=.24)
        for sta,label,color,marker in [('sili','四里','#0072B2','o'),('zhujia','竺家','#D55E00','s')]:
            rr=sorted([r for r in selected if r['station']==sta],key=lambda r:r['lead_minutes'])
            assert len(rr)==16
            ax.plot([r['lead_minutes']/60 for r in rr],[r['rmse'] for r in rr],lw=2,color=color,marker=marker,ms=4,label=label)
            for r in rr:
                if r['lead_minutes']%60==0:
                    ax.annotate(f"{r['rmse']:.2f}",(r['lead_minutes']/60,r['rmse']),xytext=(0,8 if sta=='sili' else -18),textcoords='offset points',ha='center',fontsize=9,color=color)
        ax.set(title='SimVP 预测 AGRI → COT 反演器 R → 预测 COT',xlabel='预测时效（小时）',
               ylabel='相对 CPP 参考的物理 COT RMSE',xlim=(.15,4.15),ylim=(0,36),xticks=[.25,1,2,3,4])
        ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
        fig.text(.5,.065,'新月度划分的验证集；16×16 COT 场逐像元评分，未先做空间平均。\n271个窗口，CPP参考支持8个目标日期；CPP为卫星检索参考。',ha='center',fontsize=9)
        for ext in ('png','pdf'):fig.savefig(OUT/f'simvp_to_cot_rmse.{ext}',dpi=260)
        plt.close(fig)
    report=dict(chain='historical AGRI -> frozen SimVP forecast AGRI -> frozen R -> physical COT',
                reference='matched future CPP, evaluation only',scale='16x16, errors computed per pixel',
                split='monthly validation',windows=metrics['sequence_count'],
                cpp_supported_target_dates=metrics['unique_target_bjt_dates'],test_read=0,
                simvp_checkpoint_sha256=metrics['simvp_checkpoint_sha256'],R_checkpoint_sha256=metrics['R_checkpoint_sha256'],
                source=str(SRC/'forecast_cot_metrics_fp64.json'),
                station_metrics=[{k:r[k] for k in ('station','n_pixels','forecast_simvp_to_R_rmse_cot','forecast_simvp_to_R_mae_cot','forecast_simvp_to_R_bias_cot')} for r in metrics['station_metrics']],
                extraction='existing independently FP64-recomputed metrics; no new inference')
    (OUT/'source.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(OUT)

if __name__=='__main__':main()
