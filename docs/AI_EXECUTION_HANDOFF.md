# 下一位AI的执行交接

## 当前范围

已授权建立第二篇工程和准备写作/消融方案。本轮未启动训练，未建立后台自动化。本文件是后续执行合同，不是已经存在可执行训练程序的声明。先读本工程AGENTS.md、docs/UPSTREAM_PROJECT_RULES.md和IRRADIANCE_ERROR_PREVENTION.md，再从EXPERIMENT_TRACKER的P200开始，勿继续第一篇R200编号。

## 路径

本地：`F:/CODE_Classify/辐照度预报/solar-cot-ifs`。
已登记M0服务器绝对路径：`/home/Data_Pool_3/wangyc/irradiance_paper/hunan_shortterm_m0_20260903_r101_r103_v1`，来自现有README，本轮未联网核验。
建议新服务器工程根：`/home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905`，目前未创建、未确认写权限；先在真实目标主机核对挂载与权限，不把该建议当作已部署事实。不得输出本地路径替代服务器执行路径。

## 执行步骤和失败规则

1. 使用已有SSH配置只读查证服务器M0、湖南SimVP、COT和IFS文件；按边界日期/明确manifest查，避免大目录全树find。核实训练区域/实际训练时间而非目录名。记录配置、norm、checkpoint和manifest的SHA256。只保存连接配置引用，别拷贝密钥。
2. 实现P200审计脚本与asset_manifest：用真实schema检查CPP质量位、填充值、COT范围和角度；每条IFS记录确定release≤init，严格同cycle相邻step差分。未知字段直接FAIL，不默认填0或“晴空”。
3. 核对M0 train/val/test支持范围、30个标签手算、2站grid真实HDF行列。新增cohort不得根据模型误差或未来CPP质量筛选。若第一篇已触碰test，记录暴露范围，披露限制。
4. 若无湖南合规SimVP权重，先查第一篇是否已有在训作业；已有则共享其完工资产，未有则只安排一次基线训练。COT缺合规资产时只训练候选小R，来源和合同独立。旧湖北资产不作为临时正式权重。
5. 用32个train/val初始化做端到端smoke；分别断言输入来源与时间、输出16lead、有限值、预测和标签单位、冻结参数无梯度。8个train样本过拟合仅用于检查学习链路。加载checkpoint时严格匹配通道和norm长度，禁止静默缺权重。
6. 使用单卡profile记录显存、batch、GPU利用率、样本/秒和I/O时间。先增加单卡吞吐，必要时梯度累积保持有效batch；饱和后可多卡。不得取消/抢占第一篇已有作业。
7. 外推缓存共用一次，建议站点patch级存储。每条cache保留station/init/target/lead、source_split、SimVP/R/norm哈希。正式保存全部逐样本预测，不能只存汇总metrics。
8. 按experiment-plan实现并运行A/AC单种子pilot；IFS审计通过才扩至AN/ACN。之后AL/ALN和容量对照判断表示归因，避免先跑大矩阵。明确错误修复须新run_id；无明确错误的负结果不反复搜索test。
9. 正式每组3种子，统一训练/选模预算。验证冻结配置后才统一评估test，按日期配对CI、多日块敏感性及各站指标出表。主指标重算、cohort主键/真值/掩膜一致性必须通过。
10. 使用result-to-claim判断C1/C2为supported/partial/rejected；再更新英文LaTeX和中文Markdown相同章节、公式和数字。无提升如实写，不调用自动审稿循环编造已完成结果。

## 原始产物最低字段

逐样本预测：run_id、seed、station、split、init_time、target_time、lead_minutes、ghi_true、ghi_pred、kt_pred、clear_sky_mean、day_mask、ifs_cycle、ifs_release_time、cohort_hash、upstream_hash。禁用仅字符串拼接的含混issue_time。
审计产物：版本化config、输入/资产manifest、norm、权重哈希、覆盖统计、训练日志、最佳验证轮次、退出码、profile。工程信息不进入论文正文。

## 完成标准

P200所有必要门通过；smoke冻结/因果/数值正确；核心组与归因对照按相同预算完成；共同样本原始预测可重算；负结果保留；双语一致；论文结论有逐项证据。若IFS因果数据不存在，则交付COT单线实验并将C2明确标为未完成，不能伪造完整融合结论。
