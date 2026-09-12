# 冻结R数值后端见证：207626已完成，禁止重复提交

2026-09-08本轮自动检查已提交且完成207626.tc6000，单A10050秒，exit0。输出audits/r_cache_numeric_witness_20260908.json已存在，状态COMPLETE_WITH_NUMERICAL_DIFFERENCES。详见R_CACHE_NUMERIC_RESULTS_20260908.md；以下为保留的原执行协议，未提交文字仅属先前准备时点。

目的：解释六个固定forecast-cache序列CPU FP32重载与已保存GPU FP32 R输出的差异。现有差异最大log1p为0.008137；不得采用旧GPU FP16验证的0.01阈值宣布通过。本见证所有数值比较固定最大绝对误差≤0.001，超限保留并继续收齐证据，不放宽阈值、不训练、不改旧缓存。

服务器根：`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907`。

脚本 `scripts/verify_r_cache_numeric_backends.py`；PBS `scripts/run_r_cache_numeric_witness.pbs`。单张node21 A100、4 CPU、10min walltime。GPU使用原始完整分片的batch形状；CPU只复算原诊断固定的每序列32个patch。输入、R权重/source/norm与分片均以SHA绑定，不读取GHI/kt/test。

脚本分别比较NumPy/CPU torch/GPU torch归一化及两套GPU计算设置：benchmark=True且cuDNN TF32=True；benchmark=False且cuDNN TF32=False。两者同时改变算法选择与TF32，不能将其差值单独归因于TF32。GPU与CPU、旧缓存、四维统计及小头pack映射分别报告；任一数值超限不是伪装的PASS。来源/形状错误中止并保存错误报告。

准备验收：独立准备者已在远端完成py_compile、bash -n及--metadata-only（约3.8秒，无前向/GPU），六分片与R/规范化/特征索引SHA通过；主代理已读取代码复核。尚未提交GPU作业。

## 下一小时执行

先读实时PBS清单、`r_cache_numeric_witness_pbs.log`和`audits/r_cache_numeric_witness_20260908.json`。若已有该作业排队/运行或已有报告，先检查而不重复提交。当前本轮仅准备脚本和CPU检查，未qsub此见证。

```bash
cd /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907
/public/home/slfu/miniconda3/envs/swc/bin/python -m py_compile scripts/verify_r_cache_numeric_backends.py
bash -n scripts/run_r_cache_numeric_witness.pbs
/public/home/slfu/miniconda3/envs/swc/bin/python scripts/verify_r_cache_numeric_backends.py --project "$PWD" --metadata-only
/opt/gridview/pbs/dispatcher/bin/qstat -u slfu
```

以上源码/路径检查通过、确认无重复作业且A100资源可用后，只提交一次：

```bash
/opt/gridview/pbs/dispatcher/bin/qsub scripts/run_r_cache_numeric_witness.pbs
```

记录返回PBS号到最新状态和自动推进清单。完成后检查PBS退出状态、日志执行完成标记及JSON完整六样本；重新核对报告内source哈希，独立解释各比较。`COMPLETE_WITH_NUMERICAL_DIFFERENCES`表示收齐了差异证据，不等于研究失败或数值已解释。即使后端设置可以解释差异，也不能直接认定它导致AC无收益，不能因此自动改cache/重训；先记录影响及明确修复依据，再遵守已有独立新命名/同预算成对协议。

报告为服务器 `audits/r_cache_numeric_witness_20260908.json`。旧模型、旧cache和本轮8次负结果均保留。
