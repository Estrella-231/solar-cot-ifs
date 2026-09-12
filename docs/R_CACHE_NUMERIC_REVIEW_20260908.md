# 冻结 R 数值见证独立复核（2026-09-08）

结论：本次见证支持“后端设置相关的数值差异解释了此前 CPU 与旧 GPU cache 差异的主要部分”，但旧 cache 的最大绝对误差门 **0.001 仍未关闭**。不能单独归因于 TF32，也不能宣称这些差异造成 AC 的负收益。本复核未发现必须重跑 A/AC 的确定性实现错误；保留原始与 QC1 负结果，不据此扩大正式训练。

复核为独立代理进行的同模型家族审查，结论暂定（same-family/provisional），不是跨模型仲裁。已阅读上层与本仓库 AGENTS、两份防错规则及本仓库范围：湖南、冻结 SimVP、COT/IFS；本次未修改代码、模型、缓存或阈值，未申请 GPU、未重跑前向。

## 证据与复核范围

权威服务器根为 `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907`。

- 独立读取 `scripts/verify_r_cache_numeric_backends.py`、`scripts/cache_frozen_forecast.py` 的相关推理链和 `COTUNet` 实现；复核 FP32、eval、no_grad、冻结权重、通道拼接、原分片 batch、四维 log1p 统计及 pack C 映射。
- 独立以 Python 标准库核对 `audits/r_cache_numeric_witness_20260908.json`：固定样本、比较键、形状与点数、阈值与状态、分位数顺序、均值/RMSE/最大值关系；按逐样本 RMSE 和点数重建全部 pooled RMSE，重建汇总最大值、点数与状态，全部一致。
- 在服务器运行已审查脚本的 `--metadata-only`，得到 `PASS_CPU_METADATA_ONLY_NO_FORWARD`：重新验证 R config/checkpoint/source/norm、cache contract、六个分片 SHA 与 complete/sidecar、head norm/audit 及 C/index 数组 SHA。该检查不执行前向，不读 GHI/kt 或 test 数值。
- 独立核验远端报告与脚本 SHA；`qstat -f 207626.tc6000` 返回 C、exit_status=0、单 node21/GPU2、walltime 00:00:50。日志含 `COMPLETE_WITH_NUMERICAL_DIFFERENCES` 和 `R_NUMERIC_WITNESS_EXECUTION_COMPLETE`。

本轮 GPU 见证未保存新生成的逐像元输出数组，只保留比较统计。本次独立复核因此属于**统计一致性、代码及溯源复核**，不是从独立保存的原始新输出重新计算误差，也不是再次独立 CPU/GPU 前向。旧分片可以重新读取和验 SHA，不等同于已独立复算本轮全部新输出。

## 固定范围与汇总计算

三处样本列表（旧 diagnostic、本轮 samples、本轮 provenance）完全一致：

| split | shard | sequence | 原始序列 batch | 原始 R patch batch | 对应 pack C 行 |
|---|---|---:|---:|---:|---:|
| train | shard_00000.npz | 3 | 24 | 768 | 2 |
| train | shard_00577.npz | 13848 | 24 | 768 | 24 |
| train | shard_01152.npz | 27648 | 24 | 768 | 32 |
| val | shard_00000.npz | 0 | 24 | 768 | 31 |
| val | shard_00088.npz | 2112 | 24 | 768 | 32 |
| val | shard_00177.npz | 4248 | 10 | 320 | 32 |

每个样本均有同样 20 组比较，共 120 条；全部有限，13 组汇总 PASS、7 组 FAIL。GPU full-shard 输出比较覆盖 1,064,960 个 log1p 值；CPU fixed-sequence 比较覆盖 49,152 个值；pack 映射覆盖 153 行 × 4 维 = 612 个值。六样本不是六张单 patch 图，也不是全量 cache 数值重放。

独立汇总使用 `sqrt(sum(n_i * rmse_i**2) / sum(n_i))`，最大误差取逐样本最大值之最大值；每项状态严格按 `max_absolute_error <= 0.001` 判定。未将 RMSE 替代最大误差门，也未借用旧 FP16 的 0.01 容差。

| 比较 | 最大绝对误差 | pooled RMSE | 0.001 门 |
|---|---:|---:|---|
| NumPy / CPU torch / GPU torch 的三组 FP32 normalization | 0 | 0 | PASS |
| GPU benchmark=False、cuDNN TF32=False 对 CPU，固定序列 | 0.000004053116 | 0.000000343381 | PASS |
| GPU benchmark=True、cuDNN TF32=True 对旧保存输出，完整分片 | 0.001033306122 | 0.000091960775 | **FAIL** |
| 同上 GPU 设置的四维统计对旧四维统计，完整分片 | 0.000691413879 | 0.000063553421 | PASS |
| GPU benchmark=False、cuDNN TF32=False 对旧输出，完整分片 | 0.008136749268 | 0.000524575660 | **FAIL** |
| CPU torch 对旧输出，固定序列 | 0.008136749268 | 0.000513632284 | **FAIL** |
| GPU on 对 GPU off，完整分片 | 0.008163928986 | 0.000522198490 | **FAIL** |
| 保存 log1p 经 GPU 四维统计公式重构 | 0 | 0 | PASS |
| 保存四维统计经 float32 标准化到 pack C | 0 | 0 | PASS |

GPU on 对旧输出只有 train/shard_00000.npz 的完整分片超限；其余五个分片通过。该行不是声称超限像元必定位于预选 sequence=3；该比较覆盖整个原始分片。初始门超出量约 0.000033306122，即门值的 3.33%，仍为 FAIL。

## 可以与不可以作出的判断

输入归一化三路完全相同，GPU off 与 CPU 接近，而 GPU on/off 的差异与此前 CPU/旧 cache 差异处于同一量级，GPU on 也明显更接近旧 cache。这组内部对照支持后端相关数值差异是此前 warning 的主要解释；不支持“旧 cache 已精确复现”或“唯一原因就是 TF32”。本试验同时改变 cuDNN benchmark 和 cuDNN TF32，并且只观察每设置一次前向；旧 cache 未完整记录执行时的实际后端/算法选择。on 设置下剩余 0.001033306122 差异的具体原因尚未分离。

源代码中 R 的 GPU 路径与 `COTUNet.forward` 的数学调用链一致；R 始终冻结，使用原始完整分片形状，FP32 且在 S autocast 外。CPU 输入同值，保存输出到四维统计及 pack C 的公式关系也通过。未发现明确的通道、归一化、log1p 统计或 pack 映射错误。此结论局限于当前代码和已绑定样本，不能升级为全链无错误证明；四维 C 通过也不能替代 log1p 像元最大误差失败。

本次没有新的 GHI 预测或训练结果，不能把任何数值差异解释成 A/AC 排名原因。当前没有证据要求因实现错误重跑 A/AC；不得更改旧 cache 以“修好”负结果，不扩正式实验。

## 最小后续选择

若必须定位剩余数值来源，最小下一项是在相同六分片、相同权重/输入和原始 batch 上，以预先固定的 2×2 benchmark/TF32 组合并各重复前向比较，固定其他设置、记录顺序与运行环境，保存本轮原始输出及 SHA 供独立重算；继续使用 0.001 门，不重训、不覆盖旧产物。该检查是建议，未执行。

也可以保留 warning，先进行已授权的有限同 target 的 `R(real AGRI)` / `R(predicted AGRI)` 与相同 CPP 参考掩膜的只读诊断。应先固定 train/val 交集、抽样和规模，统一两支 R 的显式后端，独立保存诊断产物；真实未来图像/CPP 只作诊断参考，不进入可部署输入、下游训练选模或主 cohort。该诊断可评估误差传播现象，但不关闭旧 cache 数值门，也不证明真实物理 COT 或 AC 负收益的唯一原因。

## 文件绑定

| 文件或资产 | SHA-256 |
|---|---|
| 本次数值见证 JSON（本地=远端） | `482424f4f18a31822cb00a36b605b0abe3464b4cc5f02be0c18f020891e5b3b3` |
| verify_r_cache_numeric_backends.py（本地=远端=报告） | `af5307a1997bae3c14cdddb366eaeae4ccbb36aea1796abefa1dc026177ddd1b` |
| cache_frozen_forecast.py（本地=远端=报告） | `1140d2af5a23b97f5617e1c0c802d9fe7f2845fe744807785d07a2033c69da12` |
| train_cot_repaired.py（本地=远端=报告） | `8af2d0e101b492bf2096223fe788b7f3a8a9b5ece2544db45903b95397620e05` |
| R checkpoint（远端 metadata 验证） | `0cbdb932f49945259d5f1704316edaf6481be7cfdb54692236f829988925799c` |
| R norm（远端 metadata 验证） | `affb70cf0d645402adf6cef89311eb34599074d1856f447fcfd2cb2f3e6b1328` |
| cache contract（远端 metadata 验证） | `0ce045fb25570079fdc06a24dc85fc63cd9a807393d5e92b50108bb0010926a2` |

其余六分片、head norm/audit 的具体 SHA 见原报告 `provenance`；复核时与 metadata-only 的实时输出逐项一致。
