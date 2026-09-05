# CPP质量来源、掩膜与空间对齐

## 结论

已追通湖南站点缓存到原始CPP NetCDF的读取链。`cpp_valid_mask` 的真实含义是数值有限性，不是官方QA位。抽查18个源文件没有独立质量标志、检索不确定性或置信度字段；因此不存在可以凭空补写的“QA bit=good”。CPP只能按其实际属性作为检索参考，不能称独立观测真值或已经通过官方质量筛选的产品。

核查同时发现两个影响COT监督的问题：旧训练上限85低于源文件声明上限100；旧CPP裁剪与AGRI网格约相差一像元。旧缓存及其候选清单不得直接放行正式COT训练。源码和原始文件未被覆盖，本轮修正仅产生独立审计样例和可复用映射/掩膜函数。

## 证据链

1. 源文件：`/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP/YYYY/YYYYMMDD/FY4B_AGRI_YYYYMMDDHHMMSS.nc`。时间按UTC，与Combined样本`timestamp_utc`匹配；此Combined命名合同不能套用旧matched文件名BJT规则。
2. 构建脚本：`/home/Data_Pool_3/chenyi/GHI/scripts/legacy_root/build_hunan_station16_samples.py`，`cpp_path()`与`crop_cpp()`（约187–213行）。按写死的站点矩形读取COT/CER/CTH/CLP，将netCDF masked值填NaN，返回`np.isfinite(data)`。
3. 缓存：`/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_Station16_Combined/{station}/.../*.npz`，字段cpp、cpp_valid_mask、cpp_present、cpp_crop_bounds；不保存独立源QA。
4. 旧COT训练：`/home/Data_Pool_3/chenyi/GHI/cloud_retrieval_hunan/dataset.py`，`build_target_and_mask()`（约66–80行）叠加存在、finite、白天、0≤COT≤85，按85缩放标签。85是该训练实现的截断，不能冒充源产品有效范围。

构建脚本SHA256：`8a38195d276199a1a9000ad5a9c33798221a865ba4231ff55afdd1e36bcd3be9`。
代码快照保存在本工程audits/source_snapshots；完整原始审计JSON在audits/cpp_quality_20260905_v1.json和v2.json。

## 原始产品实际字段

| 变量 | 网格/范围属性 | 单位与含义 |
|---|---|---|
| LAT / LON | 各4051个坐标 | 独立一维纬度/经度轴 |
| COT | 4051×4051；Range `0 - 100` | Unit `None`，无量纲云光学厚度 |
| CER | Range `0 - 60` | μm |
| CTH | Range `0 - 17.5` | km |
| CLP | `0: Clear; 1: Water; 2: Ice` | 分类，不是QA标志 |

这些变量的`_FillValue=NaN`，全局属性为空。字段名为自定义`Range`，并非netCDF标准`valid_range`；读取库不会自动执行该范围筛选，必须显式检查。未见质量位表、发行版本、生成checkpoint哈希。不能将另一个FY-4B官方CPP产品的QA位表直接套在这些文件上。

## 抽样证据与限制

按既有候选清单的train/validation/test×四里/竺家六组，每组选最早、中间、记录COT最大样本，共18条。选择不使用GHI或预测误差。
18/18缓存四变量与原始矩形裁剪逐点相等（含NaN），18/18掩膜等于源finite掩膜，0/18具有独立QA变量。
在这批样例中发现26个COT像元位于(85,100]，旧训练会排除；这是刻意包含厚云样例的诊断计数，不能外推成全数据比例。

每条记录保存源绝对路径/大小、完整元数据、缓存文件SHA256、源/缓存patch内容SHA256、时间、站点、旧/新坐标映射、数值与掩膜比较。源大NC未全文件hash；提取patch hash不能冒充整个源文件hash。

## 空间映射修正

旧矩形四里为`[1369,1385,2202,2218]`，竺家为`[1279,1295,2204,2220]`。直接比较源LAT/LON与缓存AGRI的grid_lat/grid_lon，18例均出现约0.04°错位；不能因为矩形源值与缓存一致就称空间对齐通过。
本产品已经是LAT/LON规则网格。正确处理是由目标AGRI网格的经纬度匹配源LAT/LON，并要求每个像元坐标误差≤1e-4°，无重复索引或插值。不得把原始FY-4B HDF的row_index/col_index直接用于该4051×4051再网格化产品。
可复用实现：scripts/cpp_grid_mapping.py。审计v2按此坐标原则重新读取18份对齐样例，18/18的经纬度最大误差均为0。正确的零基、末端不包含矩形为四里`[1368,1384,2201,2217]`、竺家`[1278,1294,2203,2219]`；生产构建仍须逐文件匹配坐标，不能再次仅依赖写死矩形。合同见configs/cot_contract.json。

完整审计JSON与18份修正样例位于服务器`/home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905/audits/cpp_quality_20260905_v2/`。这些sidecar是数值参考与坐标审计产物，不是全量重建结果；未重写全部Combined缓存。

## 可执行的数值质量合同

新候选参考掩膜：源COT有限、0≤COT≤100、白天、太阳天顶角有效且<80°。其中0–100来自源Range；SOZ<80°是本研究保守几何筛选，不是源产品QA。任何以后发现的独立QA需要单独解码与新版本，不得静默忽略。
CLP作为云相解释字段保留，不当质量位；不自动把Clear像元COT写成0，不把缺测写成晴空，不裁剪>100异常值进有效范围。
scripts/cpp_quality_mask.py明确返回各类剔除原因，并拒绝形状广播及与finite不符的掩膜。新增质量/空间测试6项，连同已有时间和IFS测试共14项通过。
此掩膜只用于R的参考监督和COT诊断；GHI主表不能依据未来CPP有效性筛样本。

## 上游检索算法的追溯边界

已找到相邻COT模型代码：`/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Model/COT/train-16y-batchsize128-100nan/main.py`，SHA256 `4db1521dc58cb7c929109c9d7d7b98af2d273a20d62ec7e7788999478f959a70`。代码使用SmaAt_UNet、AGRI与ERA5_LCCS输入、MODIS_CPP监督；cot项xmax=100、scal=100、sozn=80。它支持“该目录包含学习式CPP检索”的判断，但仅凭相邻目录不能证明18个源NC由此具体checkpoint生成。

候选文献：[Zhao等FY4B云属性应用论文](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025GL118433)和[作者公开数据/代码](https://zenodo.org/records/15321424)。出版商描述ResUnet云检索及MODIS监督；其公开代码与本地SmaAt_Unet命名并不相同。本轮未证明两者为同一模型版本，不将候选文献直接登记成已核验源产品DOI。

尚缺的是**源NC产出批次→实际生成脚本与checkpoint哈希**的关联记录，而不是一个等待我们猜测的QA位。没有该关联时，可描述已追溯的数值参考，不能宣称独立检索精度、官方质量等级或全部上游训练时段已核验。

## 下一步与放行状态

质量字段语义与源到缓存链已核清；旧空间缓存判定FAIL。全量COT数据需要从原始CPP按正确LAT/LON映射重建，并重新统计0–100和SOZ合同下的样本，不能沿用先前按旧裁剪与85上限筛出的22,462条就声称完整新清单。新R与norm仍须按论文train/val/test边界重新训练和选择。
本轮不启动训练，不修改M0站点GHI标签或第一篇工程。当前放行仅限“按已明确合同重建候选参考数据”，不放行旧COT缓存正式训练或检索质量科学主张。
