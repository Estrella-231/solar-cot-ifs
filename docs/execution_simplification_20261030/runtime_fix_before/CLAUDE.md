# ARIS 第二篇工作台

## Pipeline Status
- language: zh
- project: solar-cot-ifs
- stage: method_refinement_and_experiment_planning
- status: planning_only_no_training_launched
- skills: research-refine-pipeline -> research-refine -> experiment-plan
- executor: Codex
- reviewer: GPT-5.6-Sol xhigh, same-family provisional

先读 AGENTS.md，再读 refine-logs/PIPELINE_SUMMARY.md。全部输出限制在本工程。ARIS 是技能工作流名称，不代表已安装后台守护、已启动通宵任务或使用了 Claude 模型。

## Latest execution
- stage: validation-only COT diagnostic; no test and no causal IFS experiment
- status: four-statistic AC representation is not seed-stable; CPP producer binding and IFS causality remain unresolved
- latest_report: docs/PROJECT_STATUS_20260912.md
- deployed_root: /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907
- training: no active job asserted; latest completed validation work is PBS 208402 and 208492

## CPP provenance update
- latest_report: docs/CPP_QUALITY_PROVENANCE_20260905.md
- contract: configs/cot_contract.json
- status: source_traced_legacy_grid_failed_full_rebuild_pending
- independent_product_qa: absent_in_18_audited_cases
- corrected_samples: 18_coordinate_checks_passed
- upstream_batch_checkpoint_binding: unverified
- training: COT retrieval and head experiments completed on validation only; no new production run is authorized by this status record

CPP修复任务已启动（18例门禁后全量），最新执行说明：docs/CPP_REPAIR_RUNBOOK_20260905.md。全量完成情况以服务器status.json为准。
