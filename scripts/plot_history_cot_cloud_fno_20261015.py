"""Reproduce a descriptive cross-study comparison from existing validation metrics."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'results/regional_station_cot_20260927/sili_full_vs_repeat_last_history_20260928_v5'
OUT = ROOT / 'results/history_cot_cloud_fno_20261015'
OUT.mkdir(exist_ok=True)
d = json.loads((SRC/'full_history/training_complete.json').read_text())
snapshots = [json.loads(s) for s in (SRC/'full_history/day_metrics.jsonl').read_text().splitlines()]
by_update = {s['update']: s for s in snapshots}
assert d['best_learned_update_by_patch_thick_rmse'] == 96
assert d['selected_update'] == 0
assert d['selected_lead_metrics'] == by_update[0]['lead_metrics']
assert d['best_learned_lead_metrics'] == by_update[96]['lead_metrics']
assert len(by_update[0]['day_sums']) == 72
audit = []
for update in (0, 96):
    for scope in ('patch', 'station'):
        for method in ('transport', 'persistence', 'learned'):
            bs = [v[scope][method] for v in by_update[update]['day_sums'].values()]
            rmse = float(np.sqrt(np.sum([b['sq'] for b in bs], dtype=np.float64) / np.sum([b['n'] for b in bs], dtype=np.float64)))
            expected = by_update[update]['metrics'][scope][method]['rmse']
            assert np.isclose(rmse, expected, rtol=1e-10)
            audit.append(dict(update=update, scope=scope, method=method, rmse=rmse))

# Manual transcription of the numerical annotations in supplied PDF, p.7, Fig.2(s-v).
fno_rmse = [6.75, 8.09, 8.93, 9.61]
fno_mae = [4.23, 5.00, 5.50, 5.92]
hours = [1, 2, 3, 4]
rows = []
for h, r, a in zip(hours, fno_rmse, fno_mae):
    rows.append(dict(study='Cloud-FNO Fig.2', scope='paper_image', method='Cloud-FNO', hour=h, lead_minutes=60*h, rmse=r, mae=a, n='', thick_rmse='', thick_n='', update=''))
for scope in ('patch', 'station'):
    for method, key, source, update in (
        ('Persistence', 'persistence', 'selected_lead_metrics', 0),
        ('Transport (selected)', 'transport', 'selected_lead_metrics', 0),
        ('8-frame hybrid (candidate)', 'learned', 'best_learned_lead_metrics', 96),
    ):
        for h in hours:
            m = d[source][scope][str(4*h)][key]
            rows.append(dict(study='Sili old validation', scope=scope, method=method, hour=h, lead_minutes=60*h, rmse=m['rmse'], mae=m['mae'], n=int(m['n']), thick_rmse=m['thick_rmse'], thick_n=int(m['thick_n']), update=update))
with (OUT/'hourly_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','DejaVu Sans'], 'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
fig, axes = plt.subplots(1,2,figsize=(10.2,4.9),sharey=True)
axes[0].plot(hours,fno_rmse,'D--',color='#9467bd',lw=2,label='Cloud-FNO（论文 Figure 2）')
for h,v in zip(hours,fno_rmse): axes[0].annotate(f'{v:.2f}',(h,v),xytext=(0,9),textcoords='offset points',ha='center')
labels=['文献：2017年测试 / 全天 CPP 参考','本项目：四里站点像元 / 旧验证集']
styles=[('Persistence','末帧持续性','#999999','^',':'),('Transport (selected)','历史 COT 输运（已选中）','#0072B2','o','-'),('8-frame hybrid (candidate)','8帧历史混合修正（候选）','#D55E00','s','-')]
for ax,scope in zip(axes[1:],('station',)):
    for method,label,color,marker,ls in styles:
        selected=[r for r in rows if r['scope']==scope and r['method']==method]
        ax.plot(hours,[r['rmse'] for r in selected],marker=marker,ls=ls,color=color,lw=2,label=label)
for ax,label in zip(axes,labels):
    ax.set_xticks(hours);ax.set_xlim(.8,4.2);ax.set_ylim(0,14);ax.set_xlabel('预测时效（小时）\n'+label);ax.grid(axis='y',alpha=.2);ax.legend(loc='upper left',fontsize=9,frameon=False)
axes[0].set_ylabel('物理 COT RMSE（无量纲，越低越好）')
fig.text(.5,.035,'跨研究描述性对照，不能据此排名：参考产品、地区、分辨率和样本不同。\n本项目：四里站点像元；旧验证集512个窗口 / 72个初始化日；候选含预测COT辅助输入，非纯COT网络。',ha='center',fontsize=10)
fig.subplots_adjust(bottom=.29,wspace=.12,left=.065,right=.99,top=.97)
for suffix in ('png','pdf'): fig.savefig(OUT/f'hourly_comparison.{suffix}',dpi=210)
plt.close(fig)
table=['|范围/方法|+1 h|+2 h|+3 h|+4 h|','|---|---:|---:|---:|---:|']
for scope,method in [('paper_image','Cloud-FNO')]+[(s,m[0]) for s in ('patch','station') for m in styles]:
    rr=[r for r in rows if r['scope']==scope and r['method']==method]
    table.append('|'+scope+' / '+method+'|'+'|'.join(f"{r['rmse']:.3f}" for r in rr)+'|')
report='''# 历史 COT 外推与 Cloud-FNO：1–4小时描述性对照

本次只提取已保存的旧验证指标，没有训练、没有读取新 test 载荷。
Cloud-FNO 数字由用户提供论文第7页 Figure 2(s–v) 标注逐项核对；对应各小时，而非0到该时效累计平均，也非MODIS对照表。
本项目采用20260928 v5 full_history运行：512个验证窗口、72个BJT初始化日；原分析同时包含64×64逐像元和站点(32,32)评价，本图只展示站点像元。评价为物理COT误差，非log域误差，非先对图取空间平均；CPP有效与输运可达范围取交集。
历史COT输运：历史C13估运动，输运历史末帧R-COT；8帧混合修正还使用未来SimVP→R COT等输入，不能称纯历史COT输入模型。
冻结厚云准则最终选中step0输运。update96是按原记录保留的最佳学习候选（未选中）；四个时效统一用同一候选，不逐时效挑模型。

## 全云况物理 COT RMSE

'''+ '\n'.join(table)+'''

## 解读与边界

在本项目相同验证样本上，混合修正的1小时空间误差高于输运，4小时空间/站点误差低于输运。全天候平均误差改善不等同厚云改善，故不能忽略原选模结果。
Cloud-FNO为另一地区、另一产品与分辨率的2017年测试评价；本项目为四里旧划分验证。图分面且共享纵轴，仅展示数值量级及随时效变化，不能计算成相对Cloud-FNO的模型提升或将旧结果冒充新月度结果。
本项目CPP是检索参考，不是独立云物理真值；这也不是GHI精度。

## 复核与复现

逐时效数值从training_complete.json提取，与day_metrics.jsonl相应update的lead_metrics逐项一致；总时段RMSE另从日期级平方误差和/样本数用FP64复算通过。当前没有重新读取逐样本预测，不能把这一提取称为逐样本重新评估；逐时效CI未重算。
hourly_metrics.csv包含MAE、样本数、厚云RMSE及厚云数量。图为hourly_comparison.png / .pdf。
运行：python scripts/plot_history_cot_cloud_fno_20261015.py
论文 DOI：https://doi.org/10.1029/2025GL119553
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
(OUT/'extraction_audit.json').write_text(json.dumps({'source':str(SRC),'sha256':hashlib.sha256((SRC/'full_history/training_complete.json').read_bytes()).hexdigest(),'lead_snapshot_match':True,'aggregate_fp64_recomputations':audit,'new_test_payload_read':False},indent=2),encoding='utf-8')
print('\n'.join(table))
