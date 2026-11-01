# CPP-only COT auxiliary supervision ablation

## Question

Does the mixed CPP / AGRI-to-R target in the current COT-aware SimVP trial dilute useful cloud-physical supervision because roughly two thirds of valid auxiliary pixels use R-derived fallback values?

## Single changed factor

Compare the current mixed-target arm with a new CPP-only target arm. Keep the Hunan monthly split, exact 32,000 fixed training rows, SimVP initialization, seed 42, batch 8, 4,000 optimizer updates, optimizer/schedule, frozen R checkpoint and differentiable input path, AGRI loss, validation rows, same CPP∩R-valid pixel support, and all evaluation code fixed. The only scientific change is the auxiliary target on that support: CPP physical COT replaces the mixed CPP/R target, and R fallback pixels outside CPP support are excluded.

Recompute `lambda_COT` with the identical train-only gradient-ratio calibration on the same calibration windows. Do not tune from validation results. The comparison is not a CPP-truth claim: CPP is a retrieval reference, not independent cloud truth.

## Evaluation and gates

1. Wait for PBS 214871 to finish and independently audit its saved prediction banks. No duplicate candidate is launched while it is running.
2. Run this arm only if the mixed-target arm fails the predefined COT skill gate or its gain is uncertain, and after a fresh account-wide `qstat -u slfu` shows fewer than three unfinished jobs. Use one A800 GPU; preserve operational headroom.
3. Compare CPP-only, mixed target, equal-budget AGRI-MSE, and the original SimVP on the exact same validation keys and CPP∩R-valid mask. Report station and lead macro RMSE, center-3×3 RMSE, thick-cloud CSI at COT≥30, near-zero false alarms, and paired BJT target-date bootstrap intervals.
4. Retain the AGRI seasonal validation guardrail. Do not select checkpoints by test data. Do not open the test payload.
5. If CPP-only improves COT but not GHI, proceed to a separately frozen-head, same-row downstream GHI comparison before claiming project value. If it also fails COT skill, investigate CPP validity/coverage and target construction as the next isolated factor; do not silently alter weights, masks, or sample rows.

## Reproducibility

Candidate code is copied into this directory and changes only the target-construction function/call sites and arm metadata. The independent validation auditor is copied unchanged except its module name in its synthetic test. The PBS script must use a distinct result path and lock, retain all frozen source hashes, and be created only after the active run's completed audit defines whether this follow-up is triggered.

## Staging status (2026-10-19)

Candidate is staged locally under `solar-cot-ifs/refine-logs/cot_aux_cpp_only_20261019` and remotely under `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/cot_aux_cpp_only_20261019`. Remote hashes, Python compilation, CPP-only mask unit witness, copied mixed-target gradient witness, paired-audit synthetic test, and real frozen-R CPU gradient witness pass. The PBS script uses a distinct experiment/output/lock and temporary directory. It reuses the identical node22 raw-frame cache populated by M1. A new train/validation-only prewarm step checks the unique frame set and fills any missing frames before GPU updates, using the same `hunan_data.load_frame` and atomic per-frame `.npz` format. Its synthetic witness checks duplicate-frame de-duplication and confirms that test/buffer paths are never opened. This performance-only change follows live M1 evidence of shared-filesystem waits and low GPU occupancy; it does not change the model/data contract. The updated script and witness pass remote hash, compile, and synthetic checks in `swc`. The candidate is not submitted.

Submission remains gated on the completed independent audit of active PBS `214871.tc6000`; if M1 passes all COT and AGRI gates, follow its registered seed-stability route instead. Before any qsub, recheck all `slfu` PBS jobs and enforce the three-unended-job limit. Test payloads remain closed.

## 2026-10-19 验证集来源差异诊断补充

在M1运行期间完成单核只读诊断PBS `214874.tc6000`，对验证集固定随机抽样1,024窗口（seed 20261019）作同像元CPP/R对照；未打开测试载荷。JSON：`cot_aware_simvp_20261017/results/label_source_audit_val_1024.json`，本地与服务器SHA256一致：`5a99178618392bc60b72d67f2199a62edc83047bce73e30b2acab3e3384c5aa6`。共同有效像元1,643,712；R−CPP合并MAE/RMSE/bias为4.534/8.960/+0.264 COT，四里RMSE 9.371、竺家8.527。R目标在CPP无效而R有效的回退区域更偏厚云：四里R-COT中位数13.94、≥30比例22.42%，对照CPP覆盖区R中位数9.24、≥30比例11.90%；竺家分别11.50/25.07%对照2.27/13.09%。CPP覆盖率约30.6%–34.7%。这是来源支持差异的描述性证据，不把CPP当独立真值，不证明该差异导致下游GHI变化。

该观察提高“混合来源权重/支持不均衡”作为候选机制的优先级，但不改变实验门禁：先完成M1 step-4000和同掩膜独立审计；若未通过或结果不确定，按本文CPP-only方案做唯一目标来源变化，保持样本、初始化、seed、更新数、AGRI损失和共同验证口径不变。不要用这份验证诊断调lambda或选checkpoint。CPP-only结果仍须进一步进入固定匹配的未来GHI对照才回答项目收益。

同抽样的COT分层R−CPP诊断已在`cot_aware_simvp_20261017/results/label_source_stratified_val_1024.json`完成：CPP<1时R均值3.524、18.13%预测达到COT≥1；CPP在1–10区间R RMSE为2.905；CPP≥30时R均值33.617对CPP均值42.636（偏差−9.020），R仍≥30的比例63.42%（四里56.38%、竺家70.28%）。这显示R伪标签有条件性虚警/厚云压低，不是统一偏差；这些差异不等于相对真值误差，CPP仍是检索参考，重叠窗口也不作像元独立推断。它增强了先隔离R回退监督的理由，但不改变门禁：M1先完成；再依CPP-only单因素结果与未来GHI配对收益决定后续。
