"""Summarize saved validation aggregates; this does not read test or predictions."""
from pathlib import Path
import json
import math

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'monitor_20261030_1947_final'
rows=[]
for path in sorted((SOURCE/'forecast_evaluated').glob('validation_epoch_*.json')):
    m=json.loads(path.read_text(encoding='utf-8-sig'))
    epoch=int(path.stem.rsplit('_',1)[1])
    errors=m['forecast_rmse_lead_channel']
    counts=m['valid_pixels_lead_channel']
    reconstructed=math.sqrt(sum(e*e*n for er,nr in zip(errors,counts) for e,n in zip(er,nr))/sum(sum(nr) for nr in counts))
    assert abs(reconstructed-m['forecast_normalized_rmse'])<1e-12
    assert m['sequence_count']==3140 and m['test_payloads_read']==0
    rows.append(dict(epoch=epoch, **{k:m[k] for k in ('forecast_normalized_rmse','forecast_normalized_mae','matched_model_rmse','matched_persistence_rmse','mse_skill_vs_persistence','rmse_0_1h','rmse_1_2h','rmse_2_4h')}))
best=min(rows,key=lambda m:m['forecast_normalized_rmse'])
last=next(r for r in rows if r['epoch']==150)
gain=100*(1-last['matched_model_rmse']/last['matched_persistence_rmse'])
assert abs(last['mse_skill_vs_persistence']-(1-(last['matched_model_rmse']/last['matched_persistence_rmse'])**2))<1e-12
out={'full_forecast_evaluations':len(rows),'validation_sequences':3140,'best':best,'final':last,
     'final_matched_rmse_reduction_percent':gain,
     'final_vs_best_relative_rmse_increase_percent':100*(last['forecast_normalized_rmse']/best['forecast_normalized_rmse']-1),
     'aggregate_reconstruction_check':'PASS from saved lead/channel RMSE and counts; not raw-prediction rescoring',
     'metrics_scope':'13 standardized AGRI channels, 16 leads, valid pixels, full validation; not COT/GHI metrics',
     'checkpoint_directory':'/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanDyDiff_20261028/runs/seed42_150ep_hunan_v1',
     'test_payloads_read':0}
(SOURCE/'RESULT_SUMMARY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 湖南独立DyDiff训练结果','',
       '150轮、66300步已完成。PBS216594正常退出，8张node21 A100训练墙钟49小时2分13秒。',
       '训练35320序列，验证3140序列；8帧历史预测未来16帧，+15至+240分钟。测试数据未读取。','',
       '## 全验证结果','',
       '| 模型 | 归一化RMSE | 归一化MAE |','|---|---:|---:|',
       f"| 最佳验证权重，第{best['epoch']}轮 | {best['forecast_normalized_rmse']:.6f} | {best['forecast_normalized_mae']:.6f} |",
       f"| 预定最终权重，第150轮 | {last['forecast_normalized_rmse']:.6f} | {last['forecast_normalized_mae']:.6f} |",'',
       f"同一有效像元上，第150轮RMSE为{last['matched_model_rmse']:.6f}，末帧不变基线为{last['matched_persistence_rmse']:.6f}；RMSE相对降低{gain:.2f}%，MSE skill为{last['mse_skill_vs_persistence']:.6f}。",'',
       '## 收敛与时效','',
       '| 轮次 | 完整验证RMSE |','|---|---:|']
for r in rows:
    if r['epoch'] in (1,20,50,95,140,145,150):lines.append(f"| {r['epoch']} | {r['forecast_normalized_rmse']:.6f} |")
lines+=['',f"最终轮较最佳轮RMSE高{out['final_vs_best_relative_rmse_increase_percent']:.3f}%，后期变化已经很小。",'',
        '| 未来区间（第150轮） | 归一化RMSE |','|---|---:|',
        f"| +15至+60分钟 | {last['rmse_0_1h']:.6f} |",f"| +75至+120分钟 | {last['rmse_1_2h']:.6f} |",f"| +135至+240分钟 | {last['rmse_2_4h']:.6f} |",'',
        '这些分数是13个标准化AGRI通道的云图误差，不能作为COT误差或GHI的W/m²误差。31次完整预报验证均保留；从逐时效逐通道RMSE及像元数重建总体RMSE通过，但本次没有另做逐预测像元复算。','',
        '## 权重与后续','',f"权重目录：`{out['checkpoint_directory']}`；`last.pt`为第150轮，`best.pt`为第145轮。两份均保留。",'',
        '下一步按既定计划，在相同湖南验证窗口上比较底图，经同一湖南COT反演器后评价COT，再用相同GHI模型评价有无预测COT的收益。此前湖北模型直接用于湖南的评价与本次湖南独立训练模型分开报告。']
(SOURCE/'RESULT_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
