> 2026-10-13：此窗口状态归档；分工、交接和独立定时要求撤销，scope receipt不再要求。当前状态见../../EXPERIMENT_TRACKER.md。

# 执行窗口3状态

状态：ACTIVE（分析准备；上游正式产物尚未就绪）。

- thread_id：待登记
- 自动化ID/状态：未创建
- 当前任务：准备 T11/T14/T19/T21/T22 的独立复算合同；预注册 T16 误差盲选规则。
- 已完成产物与hash：
  - `ANALYSIS_PREP.md`（独立复算、配对bootstrap、COT评价与因果审计合同；待总控/上游接口核验）
  - `T16_EVENT_SELECTION_RULE.md`（只依赖参考云场/COT的候选排序规则；未生成事件列表）
- GPU请求/提交收据：无
- 阻塞与下一步：窗口3所需上游任务仍为 PLANNED；等待 T09/T10 等逐样本产物及 T03/T06 冻结的样本/参考场合同。收到产物后先审计 provenance 和样本键，再运行FP64独立复算；T16只在GHI误差仍未查看时生成冻结manifest。未获T30许可前不打开test。
- 更新时间：2026-10-13 BJT

本文件由执行窗口3独占更新；主EXPERIMENT_TRACKER.md由总控汇总。
