# 多种子验证执行记录

## 2026-09-10 按需核查：208030完成，种子稳定性未通过

服务器日志出现GHI_MULTISEED_TRAINING_AND_AUDIT_COMPLETE；audits/head_ghi_multiseed_20260910.json为PASS_ALL_12_FIXED_LR_PREDICTIONS_AND_CPU_RELOADS。8新增训练与12固定LR候选同99849验证行/真值/NumPy指标/CPU重载均验收通过，test_used=false。qstat记录已清除，Unknown Job Id不是失败，不重复提交；未取得PBS最终exit_status。

AC−A的GHI RMSE（W/m²，负值有利COT）：original seed42/43/44为−1.010415/−1.080577/+1.090858，均值−0.333378，样本标准差1.233923，2/3改善；QC1为−0.506901/+2.702890/+0.740733，均值+0.978907，样本标准差1.618096，1/3改善。仅新增43/44：original平均+0.005140，QC1平均+1.721812。原始三seed平均A149.959893、AC149.626515；QC1平均A149.132355、AC150.111262。

按冻结协议，新增种子方向不一致，当前四维COT统计拼接配置未通过种子稳定性判据。不能延续seed42的0.67%/0.34%作为稳定提升；不能由此推导COT普遍无效。分站亦不稳定，QC1四里三个种子均退化。保持所有候选与负结果，不自动加seed、调参、改变表示、打开test或恢复小时监控。若用户后续授权研究新表示，须另立单因素和参数匹配对照协议。

用户明确要求多种子验证，按HEAD_GHI_MULTISEED_PROTOCOL_20260910.md固定seed42/43/44。复用42，新增43/44两cohort×A/AC共8训练；original固定LR0.0003，QC1固定0.001，不再逐seed调参。

PBS208030.tc6000已提交；服务器提交UTC2026-09-10T01:39:28，提交前账户队列为空，提交后仅本任务Q。1 A100/8CPU，30min墙钟上限；不是已完成证明。服务器时钟与本地有已登记偏差，不混用mtime推算时长。

独立代理ghi_protocol_check确认新fit仅随机种子/目录/元数据改变，QC1处理、GHI损失、优化器和预算保持。远端按文档原命令CPU --cpu-check退出0：损失/梯度/kt等价式、原Head一致性、43/44同seed A/AC参数逐值一致及跨seed参数不同均通过。数学和网络检查不是泛化结果。

独立验收脚本覆盖12固定LR候选同99849验证行与真值、NumPy FP64主指标、原Head固定8行CPU重载；汇总3seed及新增43/44配对差值、ddof=1样本标准差、两站及各lead。被导入原指标脚本SHA已纳入部署manifest；原始code不改，CPP生产绑定和IFS因果缺口保留。

服务器根：/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907。

- 日志：head_ghi_multiseed_20260910_v1_pbs.log
- 提交回执：audits/head_ghi_multiseed_submission_20260910_v1.json
- 源码与协议manifest：audits/head_ghi_multiseed_launch_sha256_20260910.json
- 新运行目录：runs/head_ghi_multiseed_{original,qc1}_seed{43,44}_20260910_v1
- 最终汇总：audits/head_ghi_multiseed_20260910.json
- 完成标记：GHI_MULTISEED_TRAINING_AND_AUDIT_COMPLETE

只有全部训练与独立验收通过才报告本轮完成。若收益不稳定如实保留，不自动加种子、调参数、改COT表示或打开test；不恢复持续监控。

启动检查已确认PBS状态R，exec_gpus=node21-gpu/0，GPU入口为A100-SXM4-40GB且分配前0/40536 MiB占用；PBS内数学/配对初始化CPU见证通过。此时日志尚在源数据核验/加载阶段，不冒充已完成正式epoch。
