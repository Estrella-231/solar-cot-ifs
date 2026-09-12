# 固定同目标 COT 配对的几何一致性预检（2026-09-08）

实际结果为 `PASS_GEOMETRY_EQUALITY_COORDINATE_GATE_PENDING`。固定的 872 个 real/forecast 配对中，`cosSOZ`、`cosRAA`、`day_mask` 三个 `[16,16]` 通道逐元素完全相等。此结论只通过几何一致性门禁；R pack 没有逐行坐标，尚不能由几何相等单独证明物理像元相同，也没有产生 R 推理结果或精度结论。

## 固定选择与结果

选择 SHA-256 始终为 `8a06a7f06cf3339cd8f067111d5397c173ba63868598c6904348362994b08318`。保留原 inventory 的 64 个 validation 站点/目标时刻、872 个可用配对、152 个不可用 lead、59 个 forecast shards；未按照几何结果重新选样，也未删除 6 个无任何 forecast 配对的目标。

绝对容差 `1e-5`、相对容差 `0` 在读取几何值前固定于检查脚本，未放宽。未应用 CPP mask、白天筛选或几何替换。

| 通道 | 配对像元数 | max abs | MAE | RMSE | p0/p50/p90/p95/p99/p100 abs | 绝对误差 ≤ 1e-5 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| cosSOZ | 223,232 | 0 | 0 | 0 | 全部 0 | 223,232 |
| cosRAA | 223,232 | 0 | 0 | 0 | 全部 0 | 223,232 |
| day_mask | 223,232 | 0 | 0 | 0 | 全部 0 | 223,232 |

- 总计 669,696 个配对元素完全相等，872/872 个配对通过全部元素容差。
- 64 个 real 几何 patch 与 872 个 forecast 几何 patch 均全部有限；全部 day_mask 元素属于 `{0,1}`。
- 58 个有 forecast 配对的目标，其不同初始化时刻产生的目标几何完全相等，各通道跨初始化最大范围为 0。
- JSON 保存全部 872 个配对的逐通道统计、分位数与最差位置；最大差为 0，所列最差位置只是同为零的首个位置。

6 个无配对目标保留如下（UTC）：sili 的 `2025-06-30T22:30:00`、`2025-07-03T00:00:00`、`2025-08-24T02:45:00`、`2025-09-22T06:15:00`，以及 zhujia 的 `2025-06-30T22:30:00`、`2025-09-22T06:15:00`。它们的 R 几何已检查有限性和 day_mask 二值性，但没有 real/forecast 差值。

## 读取边界和验证

服务器为 `zjnu-hpc`，工程绝对根目录为：

```text
/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907
```

实际运行耗时 10.13 秒，进程 CPU affinity 为 `[0,1]`，OpenMP/OpenBLAS/MKL/NumExpr 均限制 2 线程，`CUDA_VISIBLE_DEVICES` 为空。脚本不导入 torch，不加载模型，不执行训练或推理。

先重新核验 13 个 inventory 元数据文件 SHA（含 R rows、norm、pack config 和 forecast 索引/合同），再流式重新核验 `x_raw.npy` 与固定 59 个 forecast shard 的完整 SHA；60/60 均与原记录一致，共 SHA-stream 1,095,853,010 字节。完整文件哈希涉及原始字节流，但未把非几何 tensor 解码为科学数值。

语义读取仅有：

- `x_raw.npy` 的 64 个固定行、通道 `[13:16]`，196,608 字节；通过只读 memmap 访问。
- 59 个未压缩 NPZ shard 中固定的 872 个 `[3,16,16]` `geometry.npy` patch，2,678,784 字节；使用被完整 SHA 绑定的 ZIP/NPY offset 定位。
- 所需的 shard `indices.npy` 条目，5,304 字节，用于核验 split-local cache index。

总数值/索引读取 2,880,696 字节。未打开 R `target.npy`/`mask.npy`，未解码 predicted AGRI、COT feature/预测值、CPP 目标或 GHI 标签；没有 test 行或载荷读取，没有源坐标读取。完整文件哈希不能表述为“完全没有读过文件的其他字节”。

完成后，另在本地仅加载保存的几何 NPZ，独立重算 `real_geometry[pair_target_position] == forecast_geometry`、全部 58 个目标跨初始化相等、有限性和文件/脚本 SHA，均通过。

## 原始失败与检查器修正

首次 `cot_pair_geometry_20260908.json` 为 `FAIL_CONTRACT: selected R identity`，在任何 tensor SHA 或几何读取前停止。原因是检查器对同一 UTC 时刻执行原始字符串比较：R rows 中 `2025-06-30T22:30:00Z`，inventory 中 `2025-06-30T22:30:00+00:00`。站点相同，物理时刻相同。

原失败文件完整保留，SHA-256 为 `078a1f1eef2c1c8864df5c44d14cb3b246a4118f625225c9a415b4c0586737f1`。V2 检查器先核验该失败文件的 SHA，再把 `Z` 规范化为 Python 3.10 支持的显式 UTC，比较带时区的真实时刻；无时区或非 UTC 时间仍拒绝。选择、输入数据和阈值均未改变。V2 输出使用新文件名，记录前次失败 SHA 与修正原因，未覆盖失败记录。

## 可复核产物

以下路径均相对于上述服务器工程根目录：

| 产物 | SHA-256 |
| --- | --- |
| `scripts/check_cot_pair_geometry.py` | `4e3d3b4310ec894d3f9deb7159bcc6af93fad4d9bdba070a53c47d099ff890af` |
| `audits/cot_pair_inventory_20260908.json`（本次输入） | `7e7f5f87cced284797ed46b47104c16ad62770c62eb6f611f88036637fa8cf2d` |
| `audits/cot_pair_geometry_20260908_v2.json` | `ebe0ee3e5e06143685eb0a0c6cc4f494d01e7df039b7f90fd61cd500bc31c796` |
| `audits/cot_pair_geometry_values_20260908_v2.npz` | `9beac346aaf6c9b28d90ac10fe7651f8ec8668d8904481564b8439dacc6ed63d` |

几何 NPZ 仅 233,327 字节，保存 64 个 real 几何、872 个 forecast 几何、配对到目标的索引、R pack 行号、选择 SHA、通道名和容差，不含 AGRI/CPP/GHI。

重跑必须使用不存在的新输出文件名；脚本拒绝覆盖。以下在服务器运行，`_recheck` 名称若已存在应再另取新名称：

```bash
ROOT=/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \
  /public/home/slfu/miniconda3/envs/swc/bin/python "$ROOT/scripts/check_cot_pair_geometry.py" \
  --root "$ROOT" \
  --inventory "$ROOT/audits/cot_pair_inventory_20260908.json" \
  --previous-failure "$ROOT/audits/cot_pair_geometry_20260908.json" \
  --output "$ROOT/audits/cot_pair_geometry_20260908_recheck.json" \
  --values "$ROOT/audits/cot_pair_geometry_values_20260908_recheck.npz"
```

下一必要门禁是将固定目标的 original/sidecar `grid_lat/grid_lon` 与 frozen forecast 静态湖南站点 patch 逐像元绑定。几何相等不能替代该坐标证据；本报告也不通过 R backend 数值见证、不构成 CPP source NC 与生成 checkpoint 的绑定证据，不产生任何独立泛化或 GHI 改善结论。
