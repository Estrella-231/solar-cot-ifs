# CPP 源生产者几何与时间绑定审计（2026-09-08）

## 判定

**状态：`COMPLETE_PRODUCER_BINDING_UNVERIFIED`。** 本轮没有发现证据证明源 CPP 的 COT 数组发生了转置、纬向反转、经向反转或 off-by-one；现有证据也不足以在生产者一侧排除这些错误。可核验的是一份源 NC 明确声明 `COT(LAT,LON)`，`LAT` 严格递减、`LON` 严格递增。尚未核验的是生产者如何把真实模型输出注册到这两个坐标轴，以及文件名时刻是否等于模型实际读取的 AGRI 时刻。

因此，当前不能以本审计单独否定已有 CPP 诊断，也不能把“下游像元链完全一致”升级为“上游 CPP 地理与时间绑定已认证”。CPP 仍应表述为**生产脚本、checkpoint 与源输入绑定未核验的参考产品**。本文件不重做结果到主张判断，也不改变先前的模型停止规则。

机器可读记录见 [`../audits/cpp_producer_geometry_20260908_v1.json`](../audits/cpp_producer_geometry_20260908_v1.json)。

## 有界审阅范围

本轮先核对本地 `CPP_QUALITY_PROVENANCE_20260905.md`、旧源快照、当前经纬度映射/缓存修复/12 行像元链脚本；随后只读访问 FD-107 上明确给出的相邻 COT 目录、三份明确路径的 Result/CPP 文件属性、一份 NC 的 header 与 LAT/LON 轴，以及两个小型候选代码文件。没有运行全树 `find`，没有读取 COT 大数组，没有读取 test payload，没有载入模型或启动训练，也没有修改服务器文件。权限为 700 的私有代码目录未尝试绕过。

## 已核验的事实

抽查文件 `/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP/2025/20250804/FY4B_AGRI_20250804064500.nc` 的 header 明确给出：

- `LAT=4051`、`LON=4051`，`COT` 的维名和顺序为 `(LAT,LON)`；
- `LAT` 从 `81` 严格递减到 `-81`，4051 个值唯一；
- `LON` 从 `24` 严格递增到 `186`，4051 个值唯一；
- 全局属性为空，未见生成脚本、checkpoint、原始 AGRI、采集时刻或处理版本字段。

这足以核验**文件声明的坐标模式**。当前 `cpp_grid_mapping.py` 与 12 行空间像元脚本均按坐标值重新求最近行列，并显式检查行列维、单调性、唯一性、转置/翻转与 ±1 候选，因此它们可以发现消费端把递减 LAT 当作递增、交换行列或沿用旧一像元偏移等错误。旧缓存的一像元偏移已在既有质量审计中被定位和修正，这是下游裁剪问题，不是本轮新发现的上游生产者错误。

## 实际 Result/CPP writer 仍未找到

相邻文件 `train-16y-batchsize128-100nan/main.py` 的本地与远端 SHA256 均为 `4db1521dc58cb7c929109c9d7d7b98af2d273a20d62ec7e7788999478f959a70`。它训练/评价 `SmaAt_UNet`，写出 PyTorch checkpoint、loss 和 `predict.npy`/`real.npy`；没有导入 netCDF4/xarray，没有引用 Result/CPP，也没有 NC 写入、4051×4051 推理、重投影、padding、切块或拼接代码。相邻 `cot_best.pth` 的 SHA256 是 `6990c2817e4639a724bf5cd94045c3e063256f6c4322539b4a1ff14fd8905b63`，但没有清单把它绑定到抽查的 NC。

网络文件中的 `view`/`permute` 是成对的 pixel shuffle/unshuffle，外部张量保持 `N,C,H,W`；代码检索未见全图 `transpose`、`flip` 或 `rot90`。这只说明该候选网络内部没有显式全局翻转，不能替代缺失的生产写入路径。

邻近位置还存在一个不同合同的 `AGRI_COT.ipynb`：仅使用 AGRI，COT 上限 150，batch 32，训练 30 epoch，保存名 `AGRI_V1`。它也没有基础 Result/CPP writer。`TP_CPP.py` 会读取 Result/CPP 后再裁剪写 NC，属于下游子集器；其中在 FY4B 目录拼出 `FY4A_AGRI_...` 名称是这个消费者的局部缺陷，不能拿来推断基础产品的几何方向。多个候选合同进一步说明，不能因代码“在旁边”就认定实际生产 checkpoint。

相邻训练代码与 checkpoint 的属主为 `zhaozj`，时间是 2025-06-29；抽查 Result/CPP 文件属主为 `yangzx`，生成时间是 2026-02-28/03-01。属主和时间不同本身不能证明使用了别的模型，但足以说明目录邻接不是不可变的生产绑定。明确命名的公开目录搜索至此停止；可能相关的私有目录权限为 700，实际 writer 未识别。

## 现有空间检查的盲区

| 层级 | 当前能核验 | 仍不能核验 |
|---|---|---|
| NC schema | `COT(LAT,LON)`、LAT 递减、LON 递增 | COT 内容是否确实注册到这些轴 |
| NC → sidecar/pack | 按声明坐标取出的源像元可逐点比较 | 生产者若连同一个错误数组写入，后续一致性仍会全部通过 |
| 模型 → NC | 相邻网络内部无显式全图翻转 | 实际模型、归一化、切块、padding、拼接、写入顺序 |
| 时间 | 路径/文件名与下游 `timestamp_utc` 可一致 | 文件内容实际来自哪个 AGRI 输入/scan 时刻 |

最关键的反例是“坐标轴看起来正确，但生产者把 `COT.T`、`COT[::-1]`、`COT[:,::-1]` 或偏移一格的数组写入同一文件”。所有下游坐标检查仍会忠实读出这个错误数组。`4051×4051` 又是方阵，转置不会改变 shape。相同文件内的 LAT/LON 字段与 COT 一起出现，因此二者一致命名不能构成独立地理真值。

时间也有同样边界：当前脚本能证明“选中行的 UTC 字符串按规则指向同名 NC”，但 NC 内没有独立时间变量或源 AGRI 标识，不能证明文件内容不是前一帧、后一帧或另一个输入被重命名后的结果。

## 可执行闭环证据

下一步应向实际产出者取得一份最小、不可变的生产包，而不是继续无界搜索相邻目录。包内至少需要：

1. 生成 Result/CPP 的实际脚本或其只读快照，以及对应命令/作业记录；
2. 代码、checkpoint、归一化/配置的 SHA256，原始 AGRI 路径或稳定标识，以及文件名时间的解析规则；
3. 从模型输出张量到 `COT(LAT,LON)` 的完整几何路径，包括重投影、纬向方向、转置、padding、crop、tile 顺序和拼接边界；
4. 在已经冻结的 12 行源样本上，导出写 NC 前的中间 COT 张量或做有界复现，再按预先固定的 identity、transpose、上下/左右翻转、行列 ±1 和固定时间平移候选比较，不能看结果后重新选样本或候选；
5. 对每份复现记录原始输入、输出 NC 与中间张量的哈希。若生产者无法恢复，CPP 的科学措辞继续保留“producer/checkpoint 与源数组地理/时间绑定未核验”。

根节点正在协调的 12 行实际像元检查若通过，可以关闭“源 NC → sidecar → pack 被下游错误转置/翻转/偏移”的疑问；它不会单独关闭“模型输出 → 源 NC”这一生产者盲区。两类结论必须分别记录。

## 审计快照

- [`../audits/cpp_producer_geometry_20260908_v1/snapshots/adjacent_training_code.txt`](../audits/cpp_producer_geometry_20260908_v1/snapshots/adjacent_training_code.txt)
- [`../audits/cpp_producer_geometry_20260908_v1/snapshots/result_nc_header_and_axes.txt`](../audits/cpp_producer_geometry_20260908_v1/snapshots/result_nc_header_and_axes.txt)
- [`../audits/cpp_producer_geometry_20260908_v1/snapshots/candidate_inventory.txt`](../audits/cpp_producer_geometry_20260908_v1/snapshots/candidate_inventory.txt)
