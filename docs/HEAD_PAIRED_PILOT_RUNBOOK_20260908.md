# QC1 与 original 成对 pilot

前置科学决定及边界见QC_PILOT_DECISION_20260908.md，原模型定义见HEAD_PILOT_RUNBOOK_20260908.md。本阶段使用同一单卡依次运行QC1四次、original四次，不根据前者结果跳过后者。所有模型/学习率/损失/batch在启动前固定。源数据和base pack不修改；QCview只保存keep mask、准确排除键和保留train上的小头特征仿射归一化参数。保存预测的pack_row沿用原pack行号，val99849键完全相同。view的统计不是S/R模型归一化，不改变冻结上游。

Server project: `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907`，swc沿用，无环境重建。

CPU文档复跑：

```bash
cd /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907
python_bin=/public/home/slfu/miniconda3/envs/swc/bin/python
"$python_bin" -m py_compile scripts/prepare_head_qc_view.py scripts/train_head_qc_pilot.py
bash -n scripts/run_head_paired_pilot.pbs
"$python_bin" scripts/test_head_contract.py
"$python_bin" scripts/test_head_qc_view.py
```

QCview已由CPU prepare_head_qc_view.py创建，路径data/head_qc_view_20260908_v1；view.json必须COMPLETE_QC_SENSITIVITY_VIEW、448train排除/0val排除，保留537605/99849；源pack与view SHA核验通过。原始207615已profile/过拟合；QC分支需执行固定batch2048的GPU见证及8序列门禁后，才自动进入两组各两档LR的完整pilot。

独立审查与CPU检查通过后只提交一次：

```bash
/opt/gridview/pbs/dispatcher/bin/qsub scripts/run_head_paired_pilot.pbs
```

QC输出runs/head_A_AC_qc1_seed42_20260908_v1，日志head_pair_qc.log；original输出沿用runs/head_A_AC_seed42_20260908_v1的未训练pilot子目录，日志head_pair_original.log。两套complete.json及PBS日志PAIRED_PILOTS_COMPLETE才表示全部8次结束。任何失败保留现场并定位，不默认放大网络或换损失。最后统一保存预测重算、独立审查和go/no-go，test仍不使用。
