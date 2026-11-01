# 直接COT预报执行跟踪

2026-10-08：FD-107 全区域 train/val CPP 联合监督 mask 审计已完成，4 个分片共 15,676 个目标时刻，无重复；test 载荷未读取。train 14,346 个文件时刻中 7,833 个至少含一个有效像元，累计 70.34 亿监督像元；val 1,330 个时刻中 726 个有有效像元，累计 6.63 亿。有效像元中厚云 COT≥30 占 train 15.99%、val 14.77%。但 val 2026-02/03 全为近零COT，2026-04至06没有CPP；全量有131个HDF/已知问题项按缺测排除。全区域统计完成不等于可直接启动D42：target bank与逐帧manifest仍未冻结，验证季节/厚云事件代表性仍需按方案门槛判定。远端4分片CSV/JSON已归档至 `experiments/direct_cot_monthly_20261005_v1/m0_cpp_support/region_scan_archive/`；详见 `M0_AUDIT_20261005.md`。当前PBS队列无未结束作业；本轮没有提交训练。

2026-10-06：逐帧站点监督 mask 审计通过 M0 最低每站每季不同日期数要求（两站四季均≥10日），状态改为 `PASS_WITH_SCOPE_LIMIT`，仅放行 200-step M1 engineering smoke。CPP 文件时间覆盖仍约 train 31.23%、val 32.56%，且 val 2026-04至06无CPP；全区域质量审计未完成，新合同 R_region 未找到。故 12 轮正式 D_CPP、B_SR、输运对照仍关闭。test 载荷未读。详见 `experiments/direct_cot_monthly_20261005_v1/m0_cpp_support/M0_AUDIT_20261005.md`。

2026-10-06：M1 在 `212189.tc6000`（node22，A800 80GB）完成200更新工程烟测，8个train窗口，batch2；因果输入、mask不变性、checkpoint重载均通过，峰值分配约2.36GB。单卡含文件读取profile在 `212192.tc6000` 完成；最终校正profile选择FP32 batch72（峰值分配66.1GB，约100.6样本/秒，仍在最高实测吞吐5%以内）；I/O计时基于8个窗口重复，需用全量训练数据再核实。第一次启动 `212187.tc6000` 因模块路径参数问题在训练前退出，未产生训练工件。全量训练仍关闭，待训练/验证CPP全区域有效覆盖和厚云支持统计、target bank及训练manifest冻结。报告：`experiments/direct_cot_monthly_20261005_v1/m1_smoke/M1_SMOKE_20261006.md`。

2026-10-08：全区域逐像元审计完成（见上），替代“进行中”状态；test未读取。脚本：`scripts/audit_direct_cot_region_supervision_20261006.py`；远端临时目录 `/tmp/direct_cot_m0_20261006/`。

| ID | 任务 | 优先级 | 状态 | 放行条件 |
|---|---|---|---|---|
| M0 | 当前S/R来源、CPP新train/val覆盖及质量审计 | 必须 | PASS_WITH_SCOPE_LIMIT | 全区域支持已量化；CPP约三分之一时刻覆盖，val近零/厚云月份缺口、产品QA及131个排除项仍有限 |
| M1 | 直接8→16模型、mask/因果/重载检查、单卡profile | 必须 | PASS_ENGINEERING_ONLY | 200更新smoke及单卡profile完成；8窗重复只用于系统profile，不构成模型筛选 |
| B_SR | 新S→R逐样本CPP对照 | 必须 | TODO | 合规新R存在；绑定来源 |
| B_ADV | 历史COT纯输运与持久性 | 必须 | TODO | 同R及历史可用支持 |
| D42 | 直接CPP监督，seed42，12轮 | 必须 | GATED | 依据全区域支持审核季节/厚云代表性；冻结可用支持域target bank和逐帧训练manifest后再复核队列 |
| EVAL | 分站/分季/逐lead/厚云及事件图 | 必须 | TODO | 同参考同mask，保存逐样本预测 |
| D43_44 | 直接模型增加2个seed | 条件必须 | GATED | seed42通过第7节门槛 |
| A_MATCH | 匹配结构AGRI输出→R，3seed | 条件必须 | GATED | D筛选通过；归因所需 |
| D_phys | 单一物理COT损失辅助项 | 可选 | GATED | 明确发现log损失幅值瓶颈并单独冻结权重 |
| D_teacher | 真实未来AGRI→新R教师标签 | 可选 | GATED | 独立方案，不混CPP标签 |
| H_COMPARE | 三组相同GHI头公平重训 | 条件必须 | GATED | COT确认通过、新H合同完整 |
| TEST | 新合同独立测试 | 最终 | CLOSED | 所有资产及分析协议冻结 |

下一项是用全区域审计结果按方案门槛审核并冻结target bank与逐帧manifest；若验证季节/云类代表性不满足，就先缩小D42可支持的结论或补齐CPP，不直接qsub长训练。新R及GPU队列/资源条件需在后续实验前重新实时核验。
