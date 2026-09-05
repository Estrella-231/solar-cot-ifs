**第2轮审查（同家族 provisional）**

Problem Anchor 保留，无漂移。评分：fidelity 9.5、specificity 8.5、contribution 8.0、frontier appropriateness 9.5、feasibility 8.5、validation 9.0、venue readiness 8.0；**加权 8.68/10，CALIBRATION: none，Verdict: REVISE**。

**GAP：** 方法已可进入 P200 资产审计和 P201 smoke，审计通过后可做 P202 pilot；最近邻全文与服务器权重、CPP、IFS 资产尚未核验，因此当前不能宣称创新成立或实验链已放行。

必要修订：

1. `EXPERIMENT_PLAN.md` 仍写 `AN-aux`、投影经 train/val 选择，与末节权威定义冲突；统一为 AL/ALN：共享冻结 R、固定非训练正交投影、仅用 train 统计标准化、不得用 val 选投影。单一随机投影只能证明“接触相同 CPP 监督”，不足以排除任意隐表示差异；应收窄主张，或预注册少量固定投影种子作廉价敏感性检查。
2. IFS 的同周期差分、重叠加权和缺步断缺逻辑正确；再固定 cycle/release 并列时的唯一排序规则及产品标识即可。
3. R 架构和 masked Huber 已具体；补齐三项几何的名称、单位与归一化。CPP 未知质量位坚持不猜，在官方 schema 解析并写入 `cot_contract.json` 前不得训练 R。
