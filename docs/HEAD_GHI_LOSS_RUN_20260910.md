# GHI损失对照：执行记录

## 2026-09-10 按需检查：208020训练及验收完成

服务器日志已出现GHI_LOSS_TRAINING_AND_AUDIT_COMPLETE；head_ghi_paired_20260910.json为PASS_ALL_8_PREDICTION_AUDIT，head_ghi_reload_20260910.json为PASS_4_SELECTED_CPU_CHECKPOINT_WITNESSES，均无test。qstat记录已清除，Unknown Job Id不作为失败或重投依据；未取得PBS最终exit_status，不宣称已核验退出码。启动到最终见证约6分半，非已核验PBS计费墙钟。

相同99849验证行、两站等权GHI RMSE（W/m²）：original旧A149.817451/AC154.578853，新A150.016921/AC149.006507；QC1旧A149.905825/AC151.248653，新A149.795619/AC149.288718。新目标下COT相对A改善0.673534%/0.338395%，差中差−5.771817/−1.849729 W/m²。新损失两cohort两档同LR的AC−A均为负，但收益并非两站一致：original四里+1.833593、竺家−3.854423；QC1四里+1.477040、竺家−2.490842 W/m²。original新A较旧A略差0.199470，不能称所有模型均改善。

这是单seed且验证选模的初步正向证据，支持目标函数选择会影响当前COT增益，不证明唯一机理、普遍有效或正式显著性。下一步建议冻结本配置做成对种子稳定性与分站/分时效检查，不自动启动；test/IFS/CPP源绑定限制保留，小时监控仍关闭。

用户于2026-09-10明确要求按推荐推进。新实验协议HEAD_GHI_LOSS_PROTOCOL_20260910.md已冻结，独立复核与远端CPU原命令见证通过：损失值、解析梯度、kt等价式、Head同seed参数均通过。原train_head_pilot.py未改。

PBS208020.tc6000已提交并确认R，node21-gpu/0，1张A100-SXM4-40GB、8CPU、walltime30min。提交前账户队列为空，提交后只有本任务Q，后续启动检查R。Gold通过，GPU入口报告空闲显存0/40536 MiB。服务器提交UTC 2026-09-10T01:18:02；服务器存在已登记时钟偏差，该时刻不冒充本地时钟。

这是原始/QC1两cohort各A/AC×两档LR的8次新小头训练，batch2048；不改S/R、不重建cache、不读取test。PBS顺序执行数值/显存/过拟合门禁、训练、NumPy逐样本复算和4选择模型CPU重载；任何一步失败则停止。正式完成必须同时看训练及独立验收，不把R或qsub成功写成实验完成。

服务器根：/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907。

- 日志：head_ghi_loss_20260910_v1_pbs.log
- 提交凭据：audits/head_ghi_submission_20260910_v1.json
- 部署哈希：audits/head_ghi_launch_sha256_20260910.json
- 新输出：runs/head_ghi_original_seed42_20260910_v1、runs/head_ghi_qc1_seed42_20260910_v1
- 独立预测汇总：audits/head_ghi_paired_20260910.json
- CPU重载验收：audits/head_ghi_reload_20260910.json

汇总同时保留旧kt结果、原始与QC1、相同LR差值、各自选模A/AC差值与差中差。差中差改善不等于新AC优于A；不预先宣布COT增益。

这是一次性作业，未恢复小时监控；训练脚本内的验收不属于新的后台轮询。后续按用户要求查结果，不自动追加配置或打开test。

启动验收：已见PACK_READY original 538053/99849，随后原始A、LR0.0003进入正式训练并运行至epoch index6（第7轮）；可见每轮约1.4～1.8秒。按执行顺序，该cohort数学/显存/两组8序列过拟合门禁已通过。这仅验证启动，不报告中途分数为最终结果。
