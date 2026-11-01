# FY4B CPP V2 15 分钟生产合同（2026-09-18）

本批次使用独立的 FD-201 V2 运行目录：

```text
/home/Data_Pool/zjnu/cpp_v2_20260918_fd201
```

不写入历史 `Result/CPP`。首个生产日为 `2025-08-04` UTC，输入清单从该日实际存在的 FY4B 4 km HDF 文件构造，而不是盲目以 96 个固定时刻运行。这样缺失的原始观测会在清单中显式保留为缺口，不触发上一小时回填来伪造 COT。

## 时间与辅助场合同

- 每个清单项为实际 FY4B nominal time，间隔为 15 min；单星 COT 对应各自的卫星观测。
- ERA5 温度、相对湿度与地表温度按同一时刻所在整点读取。因此 `HH:00/15/30/45` 四个 COT 共享该小时的 ERA5 辅助场，但不会共享卫星 AGRI。
- 首日使用独占 GPU 5、`GLOBAL_CPP_BATCH_SIZE=512`、2 个数据加载 worker；若 CUDA OOM，记录失败后以 256 重跑未完成清单，不覆盖已完成 NC。
- 每个输出以文件名 UTC 时间绑定，写入独立 `output/Result/FY/FY4B/Result/{CPP,CLP_P}`。

## 扩展规则

每个后续日先生成同样的实际-HDF manifest，再按不重叠时间分片分配至当时真正空闲的 GPU。多卡只扩大不重叠 manifest 分片，绝不让两个进程写同一个 `(satellite, UTC)` 输出。
