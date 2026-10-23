# 预测域COT校正：三seed完成，候选未通过继续条件

最新状态：215768已C/exit0完成三seed×600更新及独立评分。COT参考RMSE14.07309→13.55167，但相对原预测日期CI跨零、部分厚云FAR退化超2pp，按原规则封存。不打开C_select校正结果，不启动该候选GHI训练。完整分层与来源见RESULT_REPORT.md。下方为前一阶段参考准备记录，历史“未训练”不代表最新状态。

## 要检验的问题

未来AGRI经原R后厚云偏薄、漏报较多。检验能否只利用预报时可用信息学习COT残差校正，再在独立日期检验GHI增量。不是重复调当前C_select上的GHI小头，也不承诺校正有效。

## 本轮完成

- 原train中的F_fit106263条、F_select6148条，共112411个station/init/target记录；所需6959帧全部存在，未发现数据缺失。
- target严格等于init+lead，历史为−105至0分钟；全部原split=train。F_fit与F_select目标日期分离，且与原C_fit/C_select及扩展C_fit日期无交集。没有读取GHI数值作为校正输入或监督。
- 学院node22低优先级CPU提取6959帧真实16×16 AGRI/太阳几何，原文件及patch哈希逐项核验。与前一C域参考无相同帧，因此实际复用0帧，新增提取6959帧；不隐藏此事实。
- PBS **215767.tc6000**：浙师大学院node23，3张A100-SXM4-40GB（实际GPU0/3/4），C/exit0、30秒。仅冻结R推理，无任何新模型训练。提交前slfu队列为空，GPU1/2有他人任务未干预；提交后1个未结束作业，未超资源限制。
- 五fold R及归一化哈希与既有预测来源一致，forecast重放最大RMSE0.000269 COT、最大绝对差0.002528，通过事前0.05/0.25门槛。
- 独立audit_references.py核对全部reference时间键、raw索引、逐patch哈希、数组形状、有限非负值、太阳有效掩膜及无效零占位。REFERENCE_AUDIT.json为PASS。

## 当前仍未完成

还没有训练COT校正器，也没有新的COT精度或GHI收益。接下来按冻结方案构造F域预报输入：预测AGRI/几何、原预测COT、8帧因果历史同折R-COT与mask、lead；归一化只用F_fit。先检查step0、反向、缺效掩膜、断点恢复和吞吐，再按原600更新预算拟合三seed，F_select未过继续条件就封存，不打开C_select校正结果或新增GHI训练。

真实未来图经R只作监督代理，不是CPP真值，也不能进入部署输入。test、outer buffer、official validation和新时段载荷本轮均未新增读取。总体目标仍未完成。

## 复现位置

服务器：`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_forecast_domain_cot_20261023`。

方案：`PLAN.md`及`../hubei_matched_R_dynamics_20261023/NEXT_COT_DOMAIN_PLAN.md`；数据提取`prepare.py`、冻结推理`infer_roles.py/run.py`、提交`submit.py`、数组审查`audit_references.py`。raw元数据与哈希在raw_attempt1，R输出与权重来源在reference_attempt1，各阶段收据独立保存。实际raw16.npy留服务器，不上传Git。
