# CPP空间修复执行说明

## 修复边界

用户已授权修复现存湖南站点缓存的CPP标签。原始Combined文件以只读方式打开；AGRI、GHI及原始schema不改写。新数据是一组CPP修正文件和原样本的显式关联，不是原训练脚本可以直接替换目录使用的整包副本。

扫描四里、竺家现存Combined目录的全部年份/月/日，不复用旧85阈值过滤清单。保存完整inventory；范围是现存站点文件，不补建缺失AGRI，也不声称覆盖原缓存尚未包含的日期。

每份修正文件包含COT/CER/CTH/CLP、finite掩膜、COT参考监督掩膜、源CPP路径、原样本SHA256、坐标映射和合同SHA256。每次写入均读回检查hash、CPP值和掩膜；源NC在读取前后检查大小和mtime。原始NC本身未全文件hash。

按LAT/LON轴逐文件映射，坐标误差要求≤1e-4°。源缺失时保留全NaN和全False监督掩膜，登记missing_source；读错误、合同变化或新增未解码QA字段登记error，不作为成功。COT主表共同评价样本不得由未来CPP有效性决定。

## 服务器路径与状态

- 项目根：`/home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905`
- 原缓存：`/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_Station16_Combined`
- CPP源：`/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP`
- 18例小批量：项目根下`data/cpp_repair_smoke_20260905_v2`
- 全量新目录：项目根下`data/cpp_aligned_20260905_v2`
- 启动日志：`/tmp/cpp_repair_20260905_v2.log`（当前主机临时盘，完成后应归档至项目audits）
- 启动时间：2026-09-05 22:52:05 BJT；启动器进程3792363。进程号仅是当时快照，不是持续运行保证。

入口脚本`code/repair_cpp_cache.py`。每个输出目录含`run_config.json`、`inventory.txt`、`ledger.jsonl`、`inventory_status.jsonl`、`cot_candidates.csv`、`status.json`及`samples/`。status在每100条及最终导出；JSONL每25条刷新并fsync。进程中断后最多重算未提交批次，不使用部分写入文件充当成功记录。

当前执行状态需要看日志与status，本说明不预先宣称全量完成。启动器先要求18例全部ok，再执行全量。全量零error且所有inventory条目均已登记才表示修复处理结束；missing_source仍是实际缺口，不能描述为有效CPP。

## 读取修正版

```python
from load_repaired_cpp import load_repaired_sample, cot_target

# 两个绝对路径从cot_candidates.csv同一行取得。
sample = load_repaired_sample(row['original_path'], row['sidecar_path'])
target, mask = cot_target(sample)  # COT / 100；mask控制监督
# AGRI、GHI来自原文件，CPP来自修正版。GHI只允许用于标签或评价。
```

读取器核对原文件完整SHA256、站点、时间和修正schema，不允许静默退回旧CPP。删除合并视图中已经过时的cpp_crop_bounds，使用cpp_rows/cpp_cols。无效target中的0仅供masked loss使用，原CPP缺测仍为NaN。

旧COT训练代码仍按85缩放，不能直接使用；后续新训练必须使用此读取器和100缩放合同，并重新在train范围计算归一化。完成修复也不意味着上游教师训练时段或其他论文训练门禁已经通过。

## 断点续跑

先确认该目录没有活动任务；脚本还会用OS文件锁拒绝同目录并发写入。下面命令在服务器执行：

```bash
/home/Data_Pool/zjnu/.conda/envs/swc/bin/python -u \
  /home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905/code/repair_cpp_cache.py \
  --combined-root /home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_Station16_Combined \
  --cpp-root /home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP \
  --contract /home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905/configs/cot_contract.json \
  --output /home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905/data/cpp_aligned_20260905_v2 \
  --workers 2
```

默认重试error并跳过已完成记录；源数据补齐后加`--retry-missing`重试missing_source。数据根、合同、代码或小批量选择改变时必须另建目录；代码hash变化不能静默续跑。同一冻结inventory之外的新样本需新版本。

## 本轮验证

本地隔离NetCDF环境18项测试全部通过；其中新增4项集成测试在服务器swc环境也全部通过，覆盖正确源像元提取、AGRI/GHI保留、源缺失不变晴空、原文件改变后读取拒绝。实际18例及全量结果由运行产物另行记录，不能由合成测试推断。

## 存储适配修正
首轮smoke的SQLite提交遭遇共享盘disk I/O error，未进入全量；已保留该失败目录。改用JSONL账本并新增断点尾记录测试，服务器4项集成测试通过后启动v2独立目录。最后一条未完整写入的记录续跑时丢弃并重算，已完成记录按relative_path去重。上面首轮启动时间和PID不代表v2进程。运行中的候选CSV和JSONL导出是进度快照，仅FINISHED状态可进入最终验收。

## 实际小批量验收
v2于2026-09-05 22:56:36 BJT完成18/18，零错误，耗时3.83秒（缓存及负载条件不同，不据此估计全量速度）。18份原样本hash与修复前审计相同，全部修正坐标误差为0。12份具备当前SOZ/白天/范围掩膜下的有效监督像元，其余仍保留修正版，只不进入COT候选训练清单。验收JSON：audits/cpp_repair_smoke_v2_acceptance.json。全量随后启动，运行进程3798830（快照）。

## 全量自动验收阶段
已启动一次性依赖阶段scripts/finalize_cpp_repair_20260905.sh，等待本次全量进程退出后运行verify_cpp_repair.py。验收要求完整inventory与ledger集合一致、零error、status计数一致、COT候选无重复且与有效监督记录一致，再按站点/split/状态抽样核对修正文件SHA256、原文件SHA256和掩膜。通过后写data/cpp_aligned_20260905_v2/acceptance.json；失败只保留错误日志，不产生PASS。验收阶段日志为/tmp/cpp_repair_acceptance_20260905_v2.log。这是本次修复的后续阶段，不是周期任务或已完成通知。

## 2026-09-06 02:55 BJT检查
全量进程仍运行，最近日志29,900/130,940：ok=7,741，missing_source=22,054，error=105。错误类型汇总为101个RuntimeError(NetCDF: HDF error)和4个OSError(-101, NetCDF: HDF error)。抽查首两份错误源NC，重新读取LAT/LON/COT/CER/CTH/CLP全部成功，只能确认这两份可恢复，不能据此断言全部105份无损。
已替换一次性验收等待器，主修复进程未中断。新等待器PID4050176，服务器脚本/tmp/finalize_cpp_repair_20260906_retry.sh；等待全量退出后按原合同单进程续跑失败项，最多两轮，然后执行原验收。成功和missing_source记录不重跑；仍有错误则验收不通过。新日志/tmp/cpp_repair_acceptance_20260906_retry.log。PID及计数均为本次检查快照。
