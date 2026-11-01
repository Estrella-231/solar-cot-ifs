# 2026-11-02 用户确定：真实图训练GHI头

当前主路线使用真实同期AGRI三帧训练的既有GHI头，冻结后接预测AGRI及预测COT进行预报。不再用预测图训练或微调GHI头；不得把这类变化隐含在“全train重训”中。此决定覆盖此前“两条训练路径均可默认推进”的表述。train/validation/test外层日期清单与其他数据规则不变。

Details: docs/GHI_HEAD_ROUTE_20261102.md (repository root).

---

# 2026-11-01
用户改选湖南DyDiff第150轮EMA作为固定上游。未提交先前SimVP准备任务。四卡单作业内依次完成因果输入衔接小样本、全train/val预测COT数据准备、两站各三种子A/AC从头训练及保存预测评分。当前待PBS现场启动核验；队列号与进度随后补入。本轮不会训练上游或OOF，也不读取test/buffer载荷。

PBS 217013.tc6000已提交，node21/24CPU/4GPU，提交后状态Q；提交前账户无未结束任务，node21八卡物理0MiB/0%。当前为等待/输入准备阶段，不声称GHI头已开始训练。Luna负责只读现场启动核验。正式候选已由SimVP改为湖南DyDiff150轮EMA；两站各三seed A/AC、30epoch/global batch1792。

现场核验：217013已为R，exec_gpus=node21 GPU0–3。execution.log已出现节点初始GPU快照；脚本当前进入DyDiff→修正R小样本准备，尚不能称全量bank或GHI训练已开始。Luna继续核对首个smoke完成/进度日志。

本次Luna现场进度：PBS217013仍R，4个分片均642条，共2568/33253（7.72%），最近每卡0.565～0.578序列/秒。bank剩余约3小时41～46分，不含pack及GHI头训练；pack/heads未创建，两站GHI头尚未开始。GPU0～3约5410MiB/卡，利用率采样0/4/0/17%；未发现Traceback或OOM。

性能修正：实测batch8吞吐仅小幅高于2，16输出差异过大未采用。原217013停止并保留3912序列，新217015八卡batch8仅处理剩余29341；覆盖检查无重复缺失。短测外推输入约1.8小时，不含后续训练，以新作业实际速率校正。详见PROFILE_REPORT.md及SWITCH_COMPLETE.json。

八卡实测：217015 R，八尾片已base16；稳定batch8每卡17.40～17.72秒，采样时SM88～100%，显存约16220MiB。总吞吐约3.64序列/秒，剩余输入生成约2小时15分，不含打包与头训练；比4卡约1.65倍。旧进度保留，无错误。

最新核查：217015仍R；尾片24640/29341，加原前缀3912共28552/33253=85.86%。8卡实时采样均100%，约16220MiB/卡；未发现Traceback/OOM。预测COT尚未完成、merged/pack/heads尚未出现，两站GHI头未开始。按已记录稳定吞吐生成余21～22分钟；同结构旧seed42单臂训练17.8～28.3秒（5760/6240更新），新六对并行训练仍未实测。连同合并/打包/评分暂按总余30～45分钟规划，随实际阶段耗时更新。

2026-11-02 completion verified: no active jobs. All generation, merge, pack and 12 GHI models complete, 30 epochs each; Sili5760/Zhujia6240 updates per model. Independent saved-prediction point metrics recheck PASS. Train714914; val66086 (Sili31655, Zhujia34431). Macro112.446733 to111.766468 W/m2, reduction0.605%; target-date CI[-1.621964,+0.245070] crosses zero. Sili0.611%, Zhujia0.598%; all six seed pairs improve. Lead45-240 reduction0.880%, CI[-2.013036,-0.067296]. Development validation; no test read. See RESULTS.json and POINT_METRICS_RECHECK.json.
Metadata note: merged manifest forecast_instances retained first-tail count117376; correct aggregate is33253*2*16=1064096. This field is not used for pack indexing/scoring; retain original source manifest/hash and record corrected count in recheck receipt. Old rank progress state BUILDING does not override completed manifests and outputs.

2026-11-02 requested evaluation complete: newly generated DyDiff AGRI/COT replayed through 12 existing observed-trained M00/M10 heads with original normalization; no training. Full66086 rows:122.588303 to120.501835 W/m2, reduction1.702013%, date CI[-3.527266,-0.597792]. Sili134.723470 to131.395164 (2.470472%); Zhujia110.453136 to109.608505 (0.764696%, station CI crosses zero). Common63480 rows:124.939335 to122.773748 (1.733311%), matches prior1.737% within numeric/input variation. Saved predictions and independent local point check PASS. Results:observed_trained_heads_evaluation_v1/.
