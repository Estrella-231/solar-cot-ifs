# 湖南DyDiff推进记录

2026-10-28：用户授权湖南独立底图。source_hubei六核心SHA与同门epoch150来源相同。学院GPU/队列实时核对完成；node21空闲8 A10040GB，node23空闲7，node22无可确认空闲。CPU216589正常队列初请求node21未运行，核对后仅对自身Q任务qalter到node7四CPU，C/exit0约30秒。全元数据和6帧头/网格PASS，但1验证帧IR全部不在有效范围，完整数值QC未完成。source adapter、train norm、本地缓存与2/4/8卡真实吞吐仍待做；未开正式训练、未读test。见EXPERIMENT_PLAN.md、READINESS_CPU.json和RESOURCE_SNAPSHOT.json。
