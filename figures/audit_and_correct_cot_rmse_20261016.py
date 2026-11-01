"""Restore the requested 512-window comparison and audit all source numbers."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/regional_station_cot_20260927'
OUT = ROOT / 'figures/cot_rmse_same_cohort_corrected_20261016'


def main():
    tracks = BASE / 'sili_station_val_audit_20260928_v2/station_tracks.npz'
    complete = BASE / 'sili_full_vs_repeat_last_history_20260928_v5/full_history/training_complete.json'
    original = ROOT / 'results/history_cot_cloud_fno_20261015/hourly_metrics.csv'
    reference = ROOT / 'results/cot_lead_evaluation_20261003_v2/station_512_per_lead.csv'
    d = json.loads(complete.read_text())
    with np.load(tracks, allow_pickle=False) as z:
        t = z['cpp'].astype(np.float64)
        valid = z['cpp_valid'].astype(bool)
        leads = z['lead_minutes'].copy()
        pred = {m: z[m].astype(np.float64) for m in ('simvp', 'transport', 'persistence')}
        seq = z['sequence_index'].copy()
        days = z['init_day_BJT'].copy()
    assert t.shape == valid.shape == (512, 16)
    assert len(np.unique(seq)) == 512 and len(np.unique(days)) == 72
    assert np.array_equal(leads, np.arange(15, 241, 15))
    with original.open(encoding='utf-8-sig', newline='') as f:
        old = list(csv.DictReader(f))
    with reference.open(encoding='utf-8-sig', newline='') as f:
        corrected_reference = list(csv.DictReader(f))

    raw_rows, axis_witness = [], []
    for j, lead in enumerate(leads):
        q = valid[:, j]
        assert np.isfinite(t[q, j]).all()
        for m, a in pred.items():
            assert np.isfinite(a[q, j]).all()
            e = a[q, j] - t[q, j]
            r = dict(lead_minutes=int(lead), method=m, n=int(q.sum()),
                     rmse=float(np.sqrt(np.mean(e**2))), mae=float(np.mean(np.abs(e))))
            saved = next(r0 for r0 in corrected_reference if int(r0['lead_minutes']) == lead and r0['method'] == m)
            assert r['n'] == int(saved['n'])
            assert abs(r['rmse'] - float(saved['rmse'])) < 1e-10
            raw_rows.append(r)
        repeated = float(np.sqrt(np.mean((pred['simvp'][q, 0] - t[q, j])**2)))
        v5 = d['selected_lead_metrics']['station'][str(j+1)]['simvp']['rmse']
        assert abs(repeated-v5) < 2e-6
        axis_witness.append(dict(lead_minutes=int(lead), repeated_first_rmse=repeated,
                                 v5_saved_simvp_rmse=v5,
                                 corrected_simvp_rmse=next(r['rmse'] for r in raw_rows if r['method']=='simvp' and r['lead_minutes']==lead)))

    hourly = []
    for h in (1, 2, 3, 4):
        n = int(valid[:, 4*h-1].sum())
        for name, key, stored_key in (
            ('末帧持续性', 'persistence', 'selected_lead_metrics'),
            ('历史 COT 输运（已选中）', 'transport', 'selected_lead_metrics'),
            ('8帧历史混合修正（候选）', 'learned', 'best_learned_lead_metrics'),
        ):
            metric = d[stored_key]['station'][str(4*h)][key]
            assert int(metric['n']) == n
            old_method = {'persistence':'Persistence', 'transport':'Transport (selected)',
                          'learned':'8-frame hybrid (candidate)'}[key]
            previous = next(r for r in old if r['scope']=='station' and r['method']==old_method and int(r['hour'])==h)
            assert abs(float(previous['rmse'])-metric['rmse']) < 1e-12
            if key in pred:
                raw = next(r for r in raw_rows if r['method']==key and r['lead_minutes']==60*h)
                assert abs(raw['rmse']-metric['rmse']) < 2e-6
            hourly.append(dict(hour=h, method=key, label=name, n=n, rmse=metric['rmse'],
                               source='saved v5 matched checkpoint summary' if key=='learned' else 'FP64 station_tracks.npz recomputation; v5 agreement checked'))
        r = next(r for r in raw_rows if r['method']=='simvp' and r['lead_minutes']==60*h)
        hourly.append(dict(hour=h, method='simvp', label='预测 AGRI → R 的 COT', n=n, rmse=r['rmse'],
                           source='FP64 station_tracks.npz recomputation; corrected lead-axis receipt verified'))

    OUT.mkdir(exist_ok=True)
    with (OUT/'hourly_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(hourly[0]));w.writeheader();w.writerows(hourly)
    with (OUT/'recomputed_all_leads.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(raw_rows[0]));w.writeheader();w.writerows(raw_rows)

    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
                         'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False,
                         'pdf.fonttype':42})
    fig, axes=plt.subplots(1,2,figsize=(12.0,5.4),sharey=True)
    fig.subplots_adjust(left=.07,right=.985,top=.92,bottom=.27,wspace=.12)
    hours=[1,2,3,4]
    lit=[next(float(r['rmse']) for r in old if r['method']=='Cloud-FNO' and int(r['hour'])==h) for h in hours]
    axes[0].plot(hours,lit,'D--',lw=2,color='#9467bd',label='Cloud-FNO（论文 Figure 2）')
    for h,v in zip(hours,lit): axes[0].annotate(f'{v:.2f}',(h,v),xytext=(0,8),textcoords='offset points',ha='center')
    for key,color,marker,style in [('persistence','#777777','^',':'),('transport','#0072B2','o','-'),
                                   ('learned','#D55E00','s','-'),('simvp','#A60064','D','--')]:
        rr=[r for r in hourly if r['method']==key]
        axes[1].plot(hours,[r['rmse'] for r in rr],lw=2,marker=marker,ls=style,color=color,label=rr[0]['label'])
    for ax,title in zip(axes,['文献：2017年测试 / 全天 CPP 参考','本项目：同一批512个旧验证窗口 / 四里站点像元']):
        ax.set(xlim=(.8,4.2),ylim=(0,14),xticks=hours,xlabel='预测时效（小时）',title=title)
        ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,loc='upper left',fontsize=9)
    axes[0].set_ylabel('物理 COT RMSE（无量纲，越低越好）')
    fig.text(.5,.085,'右图保留原图三条线，并补充同一批样本的预测 AGRI→R；各时效共同有效样本数：360 / 372 / 343 / 322。\n持续性、输运和预测 AGRI→R 已由保存的逐样本预测以 FP64 复算；混合修正沿用原候选指标。\nCPP为检索参考；文献数据与本项目不同，分面仅作量级参照；本图不是新月度划分结果，也不是GHI精度。',ha='center',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(OUT/f'hourly_comparison_corrected.{ext}',dpi=240)
    plt.close(fig)
    audit={'status':'CORRECTED_ORIGINAL_512_COHORT_COMPARISON', 'test_read':False,
           'previous_figure_status':'WITHDRAWN_AS_ANSWER_TO_THIS_REQUEST; 128-cohort point values match saved summary but change the requested experiment',
           'original_figure_semantics':'historical retrieved-COT forecasting, not target-time observed-image retrieval RMSE',
           'sequences':512,'initialization_days':72,'per_hour_counts':[360,372,343,322],
           'raw_recomputation_verified':['simvp','transport','persistence'],
           'hybrid':'original v5 update96 saved metric retained; no local per-sample hybrid rerun',
           'v5_simvp_axis_bug':'f[:,0] repeats first forecast frame over all target leads; independently reproduced against saved station targets; use corrected per-lead NPZ series',
           'axis_witness':axis_witness,
           'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (tracks,complete,original,reference)}}
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# COT RMSE 图核查与纠正（2026-10-16）','','1. 原图的三种方法都是未来COT预报，不能称真实未来图像的同期反演曲线。',
           '2. 上一版错误更换成128序列的预测/同期oracle对照，替掉原图512序列的三条线；数值有旧表来源，但不回答原请求。',
           '3. 正确补线沿用512窗口、72初始化日、同一CPP及共同掩膜，原三线保留。',
           '4. 新线由station_tracks.npz逐样本FP64复算；并核对已修正时效轴的512结果。',
           '5. v5汇总中的simvp基线错误重复第一预报帧；本次用逐时效预测替代。该错误不改变原图未包含simvp的三条线，也不能直接推断混合候选训练有误。',
           '6. 混合修正沿用同一v5候选保存指标；本地无该候选完整逐样本预测，未声称重新推理或逐样本复算候选。',
           '7. 右图是旧区域R256探索结果，不等于前述竺家16×16个例，也不是当前新月度R验证结果。',
           '', '|方法|+1h|+2h|+3h|+4h|','|---|---:|---:|---:|---:|']
    for key in ('persistence','transport','learned','simvp'):
        rr=[r for r in hourly if r['method']==key]
        lines.append('|'+rr[0]['label']+'|'+'|'.join(f"{r['rmse']:.3f}" for r in rr)+'|')
    (OUT/'AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for tag in ('predicted_image_cot_rmse_20261016','predicted_image_cot_rmse_20261016_v2','predicted_image_cot_rmse_20261016_v3'):
        old_dir=ROOT/'figures'/tag
        if old_dir.exists():
            (old_dir/'WITHDRAWN_FOR_THIS_REQUEST.md').write_text('该图换用了128序列，并非用户原图512序列的补线对照。旧点值有来源，但不得用于回答原图同样本对比。纠正版及核查见 ../cot_rmse_same_cohort_corrected_20261016/AUDIT.md\n',encoding='utf-8')
    print(OUT)
    for r in hourly:print(r['hour'],r['method'],r['n'],round(r['rmse'],6))


if __name__=='__main__':main()
