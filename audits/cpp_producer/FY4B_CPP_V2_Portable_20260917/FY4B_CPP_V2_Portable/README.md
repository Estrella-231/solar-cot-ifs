# FY4B CPP 便携运行包（105E / 133E）

本目录是 Global CPP Retrieval V2 的风云卫星专用运行包，当前重点支持 FY4B 105E/133E，保留 FY4A 105E 适配器作为同系列兼容入口，包含：

- FY4A/FY4B 的单星反演入口及公共反演代码；
- FY4B 使用的 4 个模型权重：CLP、CTH、CER、COT；
- 两个星下点所需的索引、太阳几何角和经纬度辅助文件；
- 依赖清单、环境变量模板、完整性检查脚本和运行示例。

包内不包含原始 FY4 AGRI 观测数据、ERA5/LCCS 数据或历史结果。它们通常体量很大，且由运行时按时间读取；换到另一台机器时只需要在 `config/runtime.env` 中配置这些数据根目录。

## 目录结构

```text
FY4B_Portable/
├── Code/                 反演及公共处理代码
├── Model/                FY4B 模型权重和网络结构
├── Auxiliary_Data/       FY4B 105E/133E 专用辅助数据
├── config/               运行配置模板
├── requirements.txt      Python 依赖
└── run_fy4b.sh           便携启动脚本
```

## 换环境运行

1. 安装与目标 Python/CUDA 匹配的 PyTorch，再安装其余依赖：

   ```bash
   python -m pip install -r requirements.txt
   ```

   `torch` 需要根据机器的 CUDA 版本选择对应的官方安装包；不要盲目使用 CPU 或 CUDA 版本不匹配的 wheel。

2. 复制并编辑配置：

   ```bash
   cp config/runtime.env.example config/runtime.env
   vi config/runtime.env
   ```

   至少需要设置：

   - `GLOBAL_CPP_FY4B_AGRI_ROOT`：原始 FY4B AGRI 4 km 数据根目录；
   - `GLOBAL_CPP_ERA5_ROOT` 或 `GLOBAL_CPP_ERA5_PRODUCT_ROOT`：ERA5 温度、湿度和地表温度；
   - `GLOBAL_CPP_LCCS_ROOT`：MCD12 IGBP LCCS 数据；
   - `GLOBAL_CPP_DEVICE`：例如 `cuda:0` 或 `cpu`。

3. 先做单时次预检查：

   ```bash
   set -a; source config/runtime.env; set +a
   python Code/Run_Pipeline.py plan --satellite FY4B_133E --time 2023-07-01T00:00:00
   python Code/Run_Pipeline.py satellite --satellite FY4B_133E \
       --time 2023-07-01T00:00:00 --summary
   ```

4. 使用便携启动脚本：

   ```bash
   ./run_fy4b.sh 133E --time 2023-07-01T00:00:00
   ./run_fy4b.sh 105E --start 2025-08-01T00:00:00 \
       --end 2025-08-31T23:45:00 --freq 15min \
       --retrieval-policy overwrite --device cuda:0
   ```

脚本中的 `105E` 和 `133E` 分别映射到 `FY4B_105E`、`FY4B_133E`，不会混用两个星下点的索引文件。日期范围、频率、覆盖策略、GPU 和并行度均可继续透传给 `Run_Pipeline.py`。GOES、MSG 和 MODIS 专用入口已从本便携包移除。

## 输出位置和覆盖策略

默认输出写入包目录下的 `Test/`，可通过环境变量改到其他位置：

```text
Test/Result/FY/FY4B/Result/CPP/       单星 CPP
Test/Result/FY/FY4B/Result/CLP_P/     单星 CLP_P
Test/Results/                         全球拼接结果（如运行 global）
Test/Check/                           检查报告
```

`--retrieval-policy` 控制单星结果：

- `skip`：有效文件跳过；
- `overwrite`：重新反演并覆盖；
- `repair`：仅处理缺失或损坏文件。

当前 FY4B 单星反演的观测质量控制和上一时刻回退逻辑由 `Code/Quality_Control.py`、`Code/Satellite_Retrieval.py` 统一实现。本包只保留反演功能，不包含全球拼接、完整性检查、结果修复、GIF 和可视化功能。

## 资源完整性

`MANIFEST.sha256` 保存模型和辅助文件校验值。复制、解压或转移机器后可执行：

```bash
sha256sum -c MANIFEST.sha256
```

如果只使用 2022-06 之后的 FY4B 流程，两个 FY4B SAZ 文件均保留是有意设计：当前入口默认使用 `202206010000` 文件，`202403180000` 文件作为同源的新版太阳几何辅助数据备用，不应删除。

## 重要限制

- 模型权重和辅助文件已随包提供，但原始 AGRI、ERA5 和 LCCS 数据没有复制；没有这些输入无法完成实际反演。
- 需要 Python 版本、PyTorch、CUDA、NetCDF4 和 Satpy 兼容；建议在新环境先用 `--time` 跑一个时次验证。
- 包内代码不自动下载数据、不自动覆盖已有结果；覆盖必须显式指定策略。
