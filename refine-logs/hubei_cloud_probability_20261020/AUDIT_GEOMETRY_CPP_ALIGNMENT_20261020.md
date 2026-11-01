# 湖北判云器来源与太阳几何审计

日期：2026-10-20。范围：湖北牛首、王寨、米庄的train/validation AGRI、CPP和站点GHI输入；没有读取test/buffer载荷或test标签。

## 判云模型盘点

在湖北专属实验目录、当前COT/GHI项目代码和学院服务器可见的Hubei实验根下，未找到湖北云分类checkpoint或湖北 `p_cloud` bank。湖南分类器和判云侧车确有存在，但其权重与训练来源不属于湖北，不能直接当作湖北分类模型。本实验因此需用湖北AGRI与湖北CPP-CLP标签独立训练一个分类器。

## 太阳几何

AGRI输入样本的原始通道包含SAA、SOA、SOZ和day mask。既有几何通道定义为：

\[
\cosSOZ=\cos(SOZ\pi/180),\qquad
\cosRAA=\cos((SOA-SAA)\pi/180).
\]

对9个跨train/validation时段原始AGRI样本逐像元复算，`cosSOZ`、`cosRAA`和day-mask转换的最大绝对误差均为0。将BJT键减8小时转为名义UTC后，与独立太阳位置计算的SOZ误差中位数1.12°、最大2.39°；错误地把BJT作为UTC会得到中位63.90°、最大90.69°。因此AGRI的太阳角通道可以沿用；不得将CPP文件名的UTC时间当成BJT，也不得对几何通道再加8小时。

冻结站点中心索引（16×16 patch局部坐标）为牛首(8,6)、王寨(9,8)、米庄(7,10)。

## CPP时间与空间映射

湖北输入文件键和GHI样本键用BJT；CPP原始NetCDF文件名使用UTC。对每一输入键必须先计算 `t_cpp_utc = t_agri_bjt - 8 h`，再定位 `FY4B_AGRI_YYYYMMDDHHMMSS.nc`。旧R路径manifest按相同数字串直接查找，因此它虽有3,479/3,479个文件存在，实际标签源时刻错了8小时。旧R target/mask、旧R权重以及由其生成的COT bank不能进入本轮实验；保留为历史产物，不覆盖或删除。

原始CPP根目录：`/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP`（FD-201可读，学院服务器不挂载该路径）。最新可见日期目录为2026-03-02。正确UTC转换后，旧R训练/验证清单的原始CPP文件支持为3,479/3,479；fixed-v2裁片文件现存1,050/3,479，故重建以原始NetCDF为准。一个独立抽查的正确键样本中，按AGRI `grid_static`经纬度映射得到的16×16 CLP/COT与corrected fixed-v2裁片逐像元相同，最大绝对差0；错误时刻的旧裁片对应位置为空值。

CPP原网格采用NetCDF的LAT/LON坐标映射到AGRI patch，不使用HDF row/col偏移代替经纬度。已有映射例的最大经纬坐标差满足网格匹配精度；新批量抽取仍须逐文件检查坐标轴单调性、16×16映射唯一性及偏差阈值，任何缺文件或映射失败都保持缺测，不填晴空。

## 共同GHI支持与覆盖边界

学院服务器当前湖北同期GHI包含29,351条站点train/validation行，12,301个唯一站点时刻键，BJT范围2025-08-01 06:00至2026-07-14 08:45；split为26,895 train、2,456 validation。另一train/validation AGRI路径清单有24,705帧（22,061原始data16帧 + 2,644 HDF补帧），不得把它与GHI站点行数混为同一支持。

原始CPP到2026-03-02，早于GHI输入结束时间。R训练/验证及判云验证只能在确有正确CPP/CLP参考的日期报告；R与判云推理可用于剩余train/validation AGRI键，但这些键的分类/反演精度没有同期CPP参考验证。四组GHI主评价保留所有原pack共同有效train/validation GHI行，并单列CPP支持覆盖率；test/buffer载荷关闭。

## 结论与后续

1. 湖北太阳几何输入的公式、单位和BJT/UTC解释通过复核。
2. 湖北分类器不存在于已搜索的Hubei资产范围；按用户要求用湖北train监督独立训练。
3. 旧R/ COT bank因CPP时间连接错8小时作废于本轮输入。先用正确UTC原始CPP重建R监督并训练湖北R，再产生GHI四臂的COT输入。
4. CLP标签和COT参考都来自CPP产品，不称为独立真值。四臂回答的是COT与判云得分对同期站点GHI估计的增量；不能据此直接宣称未来GHI预报改善。

远端资源快照（只读）：学院服务器 `zjnu-hpc` hostname=`tc6000`；最新查询 `qstat -u slfu` 仅列出2个状态为C（已完成）的历史任务，没有运行/排队任务。该快照不替代每次qsub前后的实时核验。
