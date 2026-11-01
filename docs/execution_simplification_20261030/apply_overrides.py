# -*- coding: utf-8 -*-
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
marker='## 2026-10-30 执行简化：不默认追加内部隔离与审查层'
policy='docs/execution_simplification_20261030/CURRENT_POLICY.md'
files={
 'AGENTS.md':f'''{marker}

用户已纠正默认U/H与过度防御。当前按[{policy}]({policy})执行：普通级联训练默认全部train和可复用现有资产；U/H、五折OOF、额外U-only R与多层内部日期划分是可选方法设计，不能仅因上游见过train样本而强制追加。预测COT可直接做有地面标签的GHI配对评价，不以CPP稀疏、COT指标未过门或新观测尚未到位为普遍阻断条件。validation可开发和选模，真实图训练后冻结的GHI头也可评价真实预报输入；如实报告验证与独立测试的区别。

主指标、站点/时效分析与继续开发分开，不默认要求每站每时效全部改善。新接口检查一次，来源/数据/代码未变则复用，取消额外审批、scope receipt和无变化重复全量审计。原时间/标签/索引、共同评价样本、未来推理因果性、train拟合归一化、test/buffer关闭、PBS≤3等规则保留；资源按现行多卡规则。检查报告见`docs/execution_simplification_20261030/AUDIT.md`。本次不改已完成结果、不重启暂停作业或定时、不修改论文正文；下方旧方案中的更严额外流程不再是普通新实验的默认要求。
''',
 'refine-logs/EXPERIMENT_PLAN.md':f'''{marker}

当前执行以[简化说明](../{policy})和[核查报告](../docs/execution_simplification_20261030/AUDIT.md)为准。默认复用完整train资产；不把U/H/OOF、嵌套日期、预报头重训或新观测数据作为普通GHI试验的通行证。最新湖南SimVP冻结头验证127.421→125.340 W/m²（1.633%）已可按实际范围报告。以下日期段保留为历史计划，不自动恢复旧任务、重训资产或追加门槛。数据完整性、测试关闭与现行资源约束不变。
''',
 'docs/COT_VALUE_FIRST_DIRECTION_20261013.md':f'''{marker}

后续普通开发按[简化执行方式](execution_simplification_20261030/CURRENT_POLICY.md)。本方向保留同期COT价值主线和单帧主候选，但F1不再强制先做预报域重训：真实图训练的冻结头预报回放与预测输入训练头均是有效、需分开说明的实验路径。完整train上游输出可用于train内下游训练；OOF/U/H与全新观测不是常规前提。已完成分支的旧规则和结果只作历史，不因本说明追认旧门槛通过。
''',
 'refine-logs/cot_value_first_20261013/EXPERIMENT_PLAN.md':f'''{marker}

以下为已完成/历史分阶段方案；普通新任务采用[当前简化方式](../../{policy})。不再因CPP覆盖少或COT辅助指标未过门而普遍阻止有效地面GHI上的廉价配对评价；冻结头预报也是可报告结果，不强制先做OOF和预测域重训。最新湖南冻结验证1.633%正结果与历史负结果均保留。旧T23–T29关闭记录和当时判定不改，不据旧表自动恢复任务。
''',
 'refine-logs/agri_direct_cot_20261020/EXPERIMENT_PLAN.md':f'''{marker}

本文件U/H、U-only R/P和单帧预测域头是已完成实验的特定设计，不能继续当全项目唯一可用路线。按[当前简化方式](../../{policy})，后续普通实验默认全train资产，不因R曾见过train内H而一律禁止复用；无需复刻U-only或五折。旧隔离、预算及评分不回写，本次不启动新训练。
''',
 'refine-logs/agri_direct_cot_20261020/EXPERIMENT_TRACKER.md':f'''{marker}

本tracker的U/H筛选及“下一步必须先核新观测”属于当时停止方案；不能作为全工程禁止继续开发的条件。当前[简化方式](../../{policy})允许在原train/validation继续有界开发、比较新假设，test仍关闭；不再要求每个候选都取得新日期或重做U/H。已有失败结果与停止记录不改，不自动恢复暂停任务。
''',
 'refine-logs/cot_aware_simvp_20261017/FUTURE_GHI_OOF_EXPERIMENT_PLAN.md':f'''{marker}

此文件为历史五折OOF方案，保留复现用途。五折不再是普通级联模型/GHI正结果的默认前提。新工作优先复用全train来源明确的资产，并按[简化方式](../../{policy})执行；不得因读到旧计划就重启五折或旧条件分支。既有任务、预算、分数和数据隔离事实均不修改。
''',
 'refine-logs/cot_aware_simvp_20261017/FUTURE_GHI_GATE.md':f'''{marker}

下方为旧候选的评价与调度历史，不再作为所有新实验通用准入门。采用[简化方式](../../{policy})：可直接评价全train级联/冻结头的因果GHI预报；不以缺少OOF、CPP指标未过门、某分站CI跨零或无新日期为通用停止理由。旧AC−AH多重判据回答的是特定历史/未来COT分解，不能覆盖其他有效问题，也不把旧未通过记录改成通过。
''',
 'refine-logs/hubei_forecast_cot_increment_20261022/EXPERIMENT_PLAN.md':f'''{marker}

本F/C及F_fit/F_select/C_fit/C_select分层只描述已完成的两阶段残差实验。新普通GHI训练不默认复制这些拆分或内部buffer，按[简化方式](../../{policy})使用原train/val并说明确需校准的部分。旧设置、对照和103.783→103.808负结果保留，不因简化重标为正结果，也不自动启动新任务。
''',
 'refine-logs/hubei_solar_geometry_repair_20261021/EXPERIMENT_PLAN.md':f'''{marker}

本修复方案已经完成，下文全train＋五折S/R和先单卡步骤为历史记录。后续不默认重做六套上游或重复已通过的几何审计；普通工作按[简化方式](../../{policy})复用合规资产，长任务直接遵循用户现行多卡规则。时间/几何真错误仍须修正；本说明不允许复用已确认错误版本冒充修复版，不触碰现有作业。
''',
}
changes=[]
for name,prefix in files.items():
 path=ROOT/name
 raw=path.read_bytes()
 backup=OUT/'before'/name
 backup.parent.mkdir(parents=True,exist_ok=True)
 if marker in raw.decode('utf-8-sig')[:220]:
  if not backup.exists():raise RuntimeError('Missing original snapshot for '+name)
  original=backup.read_bytes()
 else:
  original=raw
  if not backup.exists():backup.write_bytes(original)
 desired=(prefix.strip()+'\n\n---\n\n').encode('utf-8')+original
 if raw==desired:continue
 path.write_bytes(desired)
 changes.append({'path':name,'before_sha256':hashlib.sha256(original).hexdigest(),'after_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'change':'Current execution preface only; historical body preserved byte for byte','snapshot':str(backup.relative_to(ROOT)).replace('\\','/')})
receipt={'scope':'Second COT/GHI project execution defaults; no running code, weights, results, manuscript or memory changed','modified_files':changes,'qsub_submitted':0,'jobs_cancelled':0,'tests_opened':False,'timers_resumed':False}
if changes:(OUT/'CHANGES.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'prefaces_added':len(changes),'already_applied':len(files)-len(changes)},ensure_ascii=False))
