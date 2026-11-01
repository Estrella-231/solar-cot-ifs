# 湖北独立测试发布链复核 v4（2026-10-18）

## 最终判定

**真实 test：NO-GO。** A/AC/AP 与 forecast FA/FAC/FAP 的训练和 validation 已完成，且 27 个 forecast-head checkpoint 现已纳入完整文件哈希清单；但还不能产生可用于论文的独立 test 结果。test/buffer 数值载荷和模型权重张量未解码。本轮对已训练 forecast-head 文件流式计算 SHA-256，仅用于文件身份清点。

## 已闭合的有限范围

- 合成测试评估器现将输入、输出、回执和split/manifest路径限制在专用synthetic fixture根目录，并在读取回执前拒绝外部路径。固定实验身份账本不再随调用者复制split改路径；严格timestamp解析与完整连续两日bootstrap规则经过29项定向测试。新鲜独立复核报告见本目录 `REVIEW_TRACE.md`。
- 新 [`forecast_model_manifest_20261018_v2.json`](../forecast_model_manifest_20261018_v2.json) 登记3站×3臂FA/FAC/FAP×3随机种子，共27个checkpoint路径/hash。独立审阅确认27个组合唯一，既有18个FA/FAC路径与哈希完全一致，清单自身明确把full-train S、R、测试builder/inference和链级账本列为 `NOT_BOUND`。清单SHA-256=`b4873c60aa60a1e87216ecf00062747eae2935bc985a92e28c916f05007b3e78`。此清单仅解决head权重身份，不是release receipt。
- 本地py_compile、两个针对测试模块（29/29）和git diff --check通过；这些只证明合成契约实现，不证明真实数据来源或真实推理。

## 阻断真实test的项目

1. **真值来源/builder/访问裁决未闭合。** 虽有原始工作簿和文件/member哈希的metadata inventory，但仍需把test每条GHI真值回链至原始工作簿、字段、时间戳语义、右闭15分钟builder、单位操作口径和source-contact adjudication。当前不读取原工作簿test辐照度数值。
2. **前置S→R→GHI推理链未绑定。** forecast-head权重清单不包含用于test日期的full-train SimVP checkpoint/norm与冻结R checkpoint/norm、模型/预处理源码实际路径哈希、测试样本构造器、逐次inference receipt；现有forecast scorer不认证这些文件。
3. **一次性账本范围不足。** 当前账本在checkout内固定并由scorer领取，不能阻止复制整个checkout绕开，也没有在truth-builder或模型inference首次读取任何test payload之前领取。正式流程需要跨建包、推理、评分的同一个共享release identity和原子账本，并由只读/权限边界保护；本轮未创建/领取真实release槽。
4. **输入与既往接触风险。** 学院可读AGRI镜像没有2026-06-14/15 BJT历史输入日目录；这只证明该镜像缺日，不证明所有服务器/归档均无数据。2026-06-14存在一项旧回放任务（44起报时刻）的metadata证据，但其是否为湖北三站、是否使用当前标签或影响模型选择仍未裁决，不能称其为完全未触及。已有候选目标时刻上界不是冻结的forecast test keys。
5. **统计发布合同不全。** bootstrap次数/种子及评分依赖哈希仍由调用者决定/未绑定receipt；计划要求的月份、太阳高度和云况分层未落实或正式修订。不能在看过真实test结果后再选择分析口径。
6. **当前外部访问边界。** `zjnu-hpc` key-only只读会话可到tc6000登录节点；该节点无`nvidia-smi`命令，PBS账户快照仅见`214221.tc6000`排队。已配置的FD-201/FD-107 key-only连接均被服务器关闭；没有尝试或使用明文密码。故没有核验其它存储/计算节点的当前GPU和原始归档覆盖。

## 审计保护

- 没有读取test/buffer GHI值、AGRI像元、CPP/COT、forecast预测或test权重张量；没有真实test pack、推理或评分。
- 没有提交新训练任务，也没有覆盖历史资产。
- 湖北当前同步/forecast验证结果仍仅为validation证据；请维持论文中“无独立test结果”的边界。

下一步必须先拿到可审计的原始真值与缺失AGRI来源、解决6月14日既往接触裁决；再实现并审计共享链级release ledger、真值builder和S/R inference manifest/receipt。完成前不触碰test数值载荷。

