# 独立CPU复跑

2026-09-05，代理 `/root/audit_reproduction`；只读runbook后执行其中原始命令，无修改、无GPU、无依赖安装。
退出码0，输入22484，train12630（6226/6404），validation6954（3449/3505），test2878（1435/1443），排除22。
status=CANDIDATE_TIME_SPLIT_ONLY。
source_sha256=c81de43432f0563e892f1e2c5cd54a2b0b28bd09452f070d2a96d5b20ad9d7f1。
output_sha256=bb17c60f593f2a619c8fb72efb642b2674e26f45ca622ff3abc0657bfc39f09d。
与runbook计数和源哈希预期一致；无文档差异。报告保留质量provenance、source availability、new-train normalization、fresh-COT-training四项未完成门。
服务器独立复跑产物：`/home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905/audits/p200_fresh_agent_split_v1`。
仅证明清单处理复跑成功，不能解释为模型/训练/CPP质量通过。
