# 湖北COT辅助监督范围核查及隔离修正

2026-10-21。**已确认辅助监督的太阳几何有效范围未随冻结R更新；修正版数据已生成并验证，但没有替换运行中的训练，也没有新GHI收益。**

## 实际发现

当前`hubei_cot_auxiliary_retrain_attempt13_retry4/prepared_inputs.json`把辅助CPP源绑定到`hubei_cloud_probability_20261020/R_pack_correctutc_attempt2`。冻结R却使用`hubei_solar_geometry_repair_20261021/R_runs_attempt12/full/best.pt`和`prepared_attempt12/R_packs/full/norm.json`，后者的有效掩膜已经按修正后的太阳几何重建。

两个源pack都按正确UTC连接CPP，3,479个源时刻逐时间、split匹配；共同有效像元的CPP物理COT最大差为0。辅助数据实际19,481个可连接目标帧的24帧AGRI UTC+8h与aux BJT逐帧一致。故本次问题是**太阳几何筛选范围**，不是新的8小时时间错配、通道单位错误或CPP共同有效位置的数值变化。

源pack中76个train和10个val时刻有掩膜差异。每个时刻首个差异像元的修正太阳天顶角约77.07°至83.10°、day_mask均为1；这些样例位于既定80°筛选边界附近，不应称为“夜间标签混入”。样例角度范围不代替全部差异像元的范围统计。

## 当前辅助数据受到多大影响

按原预测窗口重复出现的目标像元记录计数，非独立样本数：

| 划分 | 原有效记录 | 修正后有效记录 | 原来有效、修正后排除 | 原来漏掉、修正后纳入 |
|---|---:|---:|---:|---:|
| train | 4,386,024 | 4,382,924 | 15,078（原有效的约0.344%） | 11,978 |
| val | 467,154 | 467,588 | 490（约0.105%） | 924 |

共同有效分别为4,370,946和466,664。原pack单次源像元统计另存，不与重复窗口统计混用。原15,078/490个被排除记录全部明显越过几何筛选边界（离边界大于预先声明的1e−6余量）；不是反标准化浮点误差造成。修正源有效像元越界数为0。

变化占比较小，不能据此断言之前全部GHI退化来自这处问题。它可能影响低太阳高度条件、辅助损失定权和验证选模，影响大小必须通过适当对照判断。

## 已完成的修正与验证

远端隔离输出：

`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/refine-logs/agri_direct_cot_20261020/hubei_attempt13_followup_20261021/corrected_cpp_shadow_v1/`

使用原构建器的固定副本，唯一更换为`prepared_attempt12/R_packs/full`标签源。实际构建4,086 train、407 val窗口；目标尺寸仍为16未来帧×1×16×16，CPP归一化目标乘100恢复物理COT。所有新数组逐帧与修正源目标/掩膜完全一致，所有目标时刻、初始化、split和可连接帧数与旧数据相同，无跨split连接。所有新有效像元均满足原有修正SZA80和day_mask规则。

完成状态为`PASS_CORRECTED_GEOMETRY_SHADOW_SIDECAR_NOT_ACTIVATED`。构建前后prepared_inputs、索引、活动sidecar manifest/目标/掩膜/键表SHA相等。没有操作215356/215359、控制器、checkpoint或评分目录，没有读取GHI、test/buffer数值，也没有运行模型训练或GHI评分。

本地保存核查JSON、32行逐lead差异表和manifest；约92 MB的修正目标/掩膜数组在上述服务器隔离目录，**没有声称本地已镜像完整数组**。原CPU审计的统计和86条代表像元记录已下载，本地脚本及计划SHA与远端产物绑定。

## 对当前实验的处理

1. 现场只读快照为2026-10-21 06:35:28 UTC：215356=R，215359=H/afterok，账户两个未结束作业。MSE-control Epoch14、171/282合并训练/验证batch；COT-aux尚未开始，无PAIR_COMPLETE或最终评价。该快照不是最终结果或全程ETA。
2. 不热替换活动标签。旧掩膜同时参与train梯度范数定λ和两臂验证选模，仅在后续臂开始前改目录也会破坏原两臂共同选模规则。原作业按旧监督范围解释，不能追认已经使用修正版。
3. 若需要完整修正版对照，应使用独立名称，按原固定train算法重算λ、相同初始化/预算/样本、共同修正掩膜选模。不能拿旧best与新best拼成唯一因素相同的结果，也不能因中途单臂指标而决定扩大搜索。本轮仅完成可直接复核的数据修正，不重复提交正在运行的训练。
4. 现有最终评分脚本仍需补齐原计划AGRI护栏与来源复核。源码中已新增局地AGRI审计，但它明确没有完成全场、分通道、分季节的物理误差；其13通道混合聚合字段还使用`wm2`名称，不能把多光谱通道混合误差报告为GHI单位。后续须按各通道实际单位分别计算，归一化全场摘要单独标注，不偷换成物理误差。只有全部预定检查通过才考虑OOF/GHI接续。

## 科学含义

本次减少了一个具体的监督来源不一致，使下一次COT辅助比较可以使用一致的几何有效范围；它本身不是新的模型方法或性能提升。目标仍是相同样本上预测COT带来的GHI改善，不能用数据修正完成、COT训练损失下降或中途验证值替代该证据。

计划与实现：`INPUT_SUPPORT_CHECK_PLAN.md`、`SHADOW_SUPPORT_REPAIR_PLAN.md`、`audit_cpp_geometry_support.py`、`build_and_verify_corrected_shadow.py`。产物：`results_v1/support_comparison.json`、`corrected_cpp_shadow_receipts_v1/shadow_verification.json`与`support_changes_by_lead.csv`。
