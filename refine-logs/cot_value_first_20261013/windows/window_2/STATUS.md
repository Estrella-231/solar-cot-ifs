> 2026-10-13：此窗口状态归档；分工、交接和独立定时要求撤销，scope receipt不再要求。当前状态见../../EXPERIMENT_TRACKER.md。

# 执行窗口2状态

状态：ACTIVE_PREPARATION；用户于2026-10-13指定本聊天为执行窗口2。本窗口已完成启动规则阅读和T07 provisional CPU prototype；正式T07冻结及T08 GPU profile尚未开始。

- 当前责任：T07/T08及后续 GHI heads；严格遵守总控唯一远端GPU/PBS提交入口。
- T07：原型有输入/梯度/容量 CPU witness PASS，但T01资产来源审计、T03数据/标签/几何合同仍为PLANNED，所以不能把原型升格成冻结的 `head_contract.json`。
- T08：未启动。依赖T05节点缓存、T06真实AGRI→R COT bank、T07最终合同；不启动GPU profile。
- 本次 witness 环境：本地 `pytorch` Conda 环境，PyTorch 2.5.1；CPU执行，没有调用CUDA。
- 产物：`T07_DESIGN_NOTE.md`、`head_candidate_v0.py`、`witness_head_candidate_v0.py`、`head_witness_v0.json`。SHA-256登记于 `artifact_hashes.sha256`。
- GPU请求/提交收据：无；PBS/远程GPU未检查或使用；未申请资源。
- 自动化ID/状态：未创建；本窗口按需推进，不声称定时运行。
- 阻塞与下一步：等待总控发布T01、窗口1发布T03；随后按冻结输入刷新结构、梯度和参数匹配witness，产出正式 `head_contract.json`。若T03 geometry与当前两平面假设不同，先调整候选并重跑 witness。
- 更新时间：2026-10-13 BJT。

本文件由执行窗口2独占更新；主EXPERIMENT_TRACKER.md由总控汇总。
