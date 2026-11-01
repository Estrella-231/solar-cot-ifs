# 湖北三站独立测试链复核 v3（2026-10-18）

## 结论

**真实独立 test：NO-GO。** 本复核仅审阅评估器、指标代码、测试和既有来源清单；没有运行真实评分，没有解码真实 test/buffer 的辐照度、AGRI、CPP/COT、预测或模型载荷。

## 已修复并由新鲜独立复核确认

| 门禁问题 | 当前证据 | 复核状态 |
|---|---|---|
| 一次性账本可因复制 split/output 另开槽 | 账本固定到冻结实验身份路径，不从调用者 split/output 推导；合成测试覆盖副本重放。 | 通过（限代码行为） |
| 合成评估器可能被指向外部真实路径 | 所有数据/输出/回执路径必须落在合成夹具根目录，且在读回执前拒绝外部路径；测试确认拒绝发生在 `np.load` 前。 | 通过（限合成隔离） |
| 两日 bootstrap 会静默漏掉孤立日期/不均匀截断 | 日期必须精确划分成完整连续、非重叠区块，且每站每区块均有样本；否则返回明确 unavailable。 | 通过（限当前敏感性实现） |
| 目标时间解析接受尾随或伪造时间戳 | 仅接受完整支持格式并验证日历时间； malformed/trailing 字符有拒绝测试。 | 通过（限解析器） |

新鲜独立复核方：`/root/hubei_test_chain_audit_fresh`，只读复查最新源码，确认上述4项修复；没有运行评估/测试或读取真实载荷。主会话于本轮实际运行：4个模块的 `py_compile` 通过；两个相关测试模块 `29/29` 通过；相关文件 `git diff --check` 通过。

## 仍未放行的关键项

1. **真实真值链未闭合。** 当前 truth manifest 仅验证路径/哈希/站点和数组键；没有经审计的原始工作簿到 15 分钟标签 builder、源单位与窗口语义、接触/访问记录的端到端绑定。
2. **真实模型与推理链未认证。** `model_manifest_sha256` 目前只是回执字段，不会实际打开并验证 checkpoint、训练配置/训练器、归一化、冻结 S/R 组件、推理入口及逐次 inference receipt。
3. **最终评分配置与依赖未冻结。** bootstrap 重复数/seed 仍可由调用者选择；指标依赖、账本依赖哈希及计划要求的月份/太阳高度/云况分层，还没有纳入一个可审计的正式 release receipt。
4. **没有真实 builder → inference → scorer 的独立干运行/放行链。** 评估器仍明确拒绝非合成 scope；不能把合成单测通过称为真实 test 通过。

这些是独立审计发现，不通过调整 test 样本或打开 test 载荷来“修复”。仍可推进的下一步是构造不读取 payload 的真实源文件/模型元数据 manifest 与 builder/inference 合同；完成后再次做新鲜审计。只有 release audit 明确 PASS 后才能触碰封存载荷。

## 当前验证源码 SHA-256

- `scripts/evaluate_hubei_forecast_heads_frozen_test.py`: `2e8bc42d5d22fc8b972a9d1f93d3112643c048cbeb6ba29d49aea71ccc660cf3`
- `scripts/hubei_forecast_metrics.py`: `6189f17d97a642a4a4c1a9c9210a5ae023bde5d7cf738b5169e5edaa9756004a`
- `scripts/test_hubei_forecast_heads_frozen_test.py`: `b76f01e8b5cfc3c781063e4d6ce4aa85cc3c6ee1ba17606b26158d49e164ab8e`
- `scripts/test_hubei_forecast_metrics.py`: `373ae3ecbcbaa588ce3f20ae8b92893831d9065ed86eb521a12f8d3659811688`

