"""Record the proven bad CPP join without overwriting historical arrays."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
text='''# CPP 配对错误：旧指标撤回（2026-10-16）

已由源rows.csv、原始validation_predictions.npz和预报bank逐项复现：构建器先筛选val后重新enumerate，却用该val-local索引查全SUPERVISED表索引sample_indices，导致CPP参考错配站点和日期。

本目录旧CPP误差/相关性/厚云/SSIM等依赖CPP配对的结论不可继续使用。重复复算同一错误bank或核对hash不能验证配对正确性。原始输出数组保留供审计，不视为已修复。

权威核查：solar-cot-ifs/audits/cpp_reference_join_20261016/AUDIT.json。
已有271窗的CPP参考已在独立目录修复；模型预测张量逐项验证未变，但这不是全验证窗口重新枚举。
完整正确原始R验证成绩：RMSE=9.074，MAE=4.646，r=0.821，356169像元。
'''

def main():
    r=ROOT/'refine-logs/cot_value_first_20261013/results'
    targets=[r/name for name in ('forecast_cot_val_bank_20261014_v1','cot_forecast_eval_20261014_v1',
        'cot_forecast_eval_20261014_v2','cot_forecast_eval_20261014_v3','cot_forecast_eval_20261014_v5',
        'cot_forecast_case_20261014_v1','forecast_agri3x3_bank_20261014_v1','forecast_agri3x3_allval_20261014_v1')]
    targets += [ROOT/'figures'/name for name in ('simvp_to_cot_only_20261016','simvp_cot_density_20261016',
               'observed_agri_cot_density_20261016','cot_log_density_by_lead_20261016')]
    for p in targets:
        if p.exists():(p/'INVALID_CPP_JOIN.md').write_text(text,encoding='utf-8')
    for p in (r/'forecast_cot_val_bank_20261014_v1/FORECAST_COT_VAL_BANK_RESULTS_20261014.md',
              r/'cot_forecast_eval_20261014_v3/COT_FORECAST_VALIDATION_RESULTS_20261014.md'):
        s=p.read_text(encoding='utf-8')
        notice='> **2026-10-16 撤回CPP评价：已确认参考CPP的全表索引/val局部索引错配。下文历史指标不可作为模型性能结论。详见本目录 INVALID_CPP_JOIN.md 与 audits/cpp_reference_join_20261016/AUDIT.json。**\n\n'
        if not s.startswith('> **2026-10-16 撤回CPP评价'):p.write_text(notice+s,encoding='utf-8')
    report=json.loads((HERE/'AUDIT.json').read_text(encoding='utf-8'))
    lines=['# CPP参考错配核查与纠正','',
           '已确认根因是CPP参考绑定错误，而不是通过改色标或删除异常点修饰模型成绩。',
           '训练代码的sample_indices索引全SUPERVISED表；profile构建器先筛val再enumerate，误把验证局部编号当全表编号。',
           '逐项重建错误映射后，2150个旧bank支持记录的参考与掩膜全部和错误映射一致。',
           '例：竺家2024-12-24 06:00 UTC被赋予四里2024-04-18 23:00 UTC的CPP。',
           '', '## 正确原始R验证',
           '直接从原始validation_predictions.npz及其全表索引复算：356169像元，RMSE=9.074，MAE=4.646，Bias=+0.279，r=0.821。',
           '四里RMSE=9.674、r=0.788；竺家RMSE=8.433、r=0.850。与训练报告一致。',
           '', '## 已有271个预报窗的局部修复',
           '仅重新按(station,target_time_utc)绑定正确CPP参考和有效掩膜，模型输出、权重、时刻完全保留。',
           '参考有效记录从550400改为668672；它是参考错误纠正，不是模型改善。',
           '|站点|输入模式|修复前错误RMSE|修复后RMSE|修复后r|','|---|---|---:|---:|---:|']
    for e in report['paired_metrics']:
        lines.append(f"|{e['station']}|{e['mode']}|{e['old_invalid']['rmse']:.3f}|{e['corrected']['rmse']:.3f}|{e['corrected']['pearson_r']:.3f}|")
    lines+=['','这些窗口当初通过错误参考映射筛选，因此仍需按修复代码重新枚举全val覆盖后生成完整预报评价；没有自动提交GPU任务。',
            '原始R完整验证与这271窗回放的日期、重复权重、输入装载/几何不同，不能混为同一支持。',
            '', '## 已完成代码修正',
            'profile_forecast_cot_val_100.py、profile_forecast_agri3x3_bank_20261014.py、profile_forecast_agri3x3_allval_20261014.py均改为先保留全SUPERVISED索引，再筛val，并核验保存的全表索引集合。',
            '仅修改本地工作代码；历史服务器代码和历史结果保留，未声称已重新运行全部推理。',
            '旧CPP相关指标和本次错误绘图已添加INVALID_CPP_JOIN.md；两个权威历史报告加撤回标记。',
            '所有绘图不得再把旧bank中27～30的RMSE和负相关解释为模型本身的性能。']
    (HERE/'AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('retractions and audit written')

if __name__=='__main__':main()
