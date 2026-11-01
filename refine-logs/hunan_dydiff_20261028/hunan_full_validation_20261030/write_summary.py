"""Readable summary from the saved paired scores, without result selection."""
from pathlib import Path
import json
import sys

root=Path(sys.argv[1])
done=json.loads((root/'FULL_EVALUATION_COMPLETE.json').read_text())
assert done['status']=='PASS'
g=json.loads((root/'ghi_scores_fp64.json').read_text())
comp=g['comparisons']['dydiff_add_COT']
lines=['# 湖南独立DyDiff第150轮：预测COT对GHI的作用','',
       '本次为完整冻结验证集上的固定模型回放，使用湖南独立训练的第150轮EMA、同一湖南修正COT反演器、同一批三帧3×3 GHI模型。','',
       f"共{done['windows']}个预报窗口，{done['common_rows']}条两站GHI记录，{len(g['unique_dates'])}个目标日期，三个种子42/43/44。",'',
       '| 范围 | 无COT RMSE (W/m²) | 加预测COT RMSE (W/m²) | RMSE降幅 |','|---|---:|---:|---:|']
for station,label in [('sili','四里'),('zhujia','竺家')]:
    m=comp['by_station'][station]
    lines.append(f"| {label} | {m['before_rmse']:.3f} | {m['after_rmse']:.3f} | {m['relative_reduction_pct']:.3f}% |")
lines.append(f"| 两站、三个种子等权 | {comp['before_macro_rmse_Wm2']:.3f} | {comp['after_macro_rmse_Wm2']:.3f} | {comp['relative_rmse_reduction_pct']:.3f}% |")
ci=comp['date_block_95CI']
lines+=['',f"加COT减去无COT的RMSE差为{comp['delta_rmse_after_minus_before']:.3f} W/m²，日期配对95%区间为[{ci[0]:.3f}, {ci[1]:.3f}]。负值表示加COT误差较低；区间跨零时保留该不确定性。",'',
        '各站点先逐种子计算RMSE，再等权平均；没有把种子预测平均后算误差，也没有将两站所有记录直接混池。', '']
sub=json.loads((root/'lead_subgroups_fp64.json').read_text())['all_forecast_slots_45_240']
lines+=[f"+45至+240分钟（三帧全部为预测）的子组：{sub['before_macro_rmse_Wm2']:.3f} → {sub['after_macro_rmse_Wm2']:.3f} W/m²，RMSE降低{sub['relative_rmse_reduction_pct']:.3f}%。",'',
        '真实未来图像/COT替换只用于离线诊断，不计作预报主结果。CPP为卫星检索参考，不能将其误差与GHI误差混用。', '',
        '数据文件：ghi_predictions.npz及ghi_rows.csv为逐记录GHI预测与标签；cot_predictions.npz及cot_rows.csv为站点邻域COT与参考；ghi_lead_metrics_fp64.csv为分站点/种子/时效结果；COMMON_KEYS_CHECK.json确认与既有完整验证的键和标签一致。']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('READABLE_RESULTS_WRITTEN',root/'RESULTS.md')
