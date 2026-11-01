# 湖北独立测试门禁工程进度（2026-10-18 更新）

## 本轮完成

- 固定一次性测试账本到实验身份路径 `audit_receipts/hubei_frozen_test_once.ledger.json`，不再由调用者提供的split路径推导；合成测试覆盖同内容split副本仍受同一账本拦截。
- 评估器增加合成夹具根目录约束，所有run/output/receipt/split/truth-manifest路径必须位于脚本的 `synthetic_hubei_test_fixtures` 下，且在读取回执前检查；测试验证外部路径被拒绝且没有打开预测载荷。
- 连续两日块配对bootstrap现在要求所有测试日期可被完整连续日块精确划分、各站点每个块均有支持；存在孤立日、缺口或站点无支持时返回明确不可用，不再静默遗漏日期或截断造成不等权重。目标时间解析改为严格完整时间戳格式并拒绝尾随脏字符。
- 正确执行语法编译，并运行两个相关模块，`29/29`通过；`git diff --check`通过。未读取真实test/buffer的GHI、AGRI、CPP/COT或预测/模型载荷，没有提交训练任务。
- 在 tc6000 通过现有 SSH key 对已训练预测头做远端文件级 `sha256sum`（不反序列化 checkpoint）：确认 FA/FAC/FAP × 三站 × 三seed 共27个权重存在；此前模型清单只含18个FA/FAC权重。本轮新增完整头部清单 [`forecast_model_manifest_20261018_v2.json`](audit_receipts/forecast_model_manifest_20261018_v2.json)，含27条权重、6份run metadata、共同训练器/归一化合同、训练pack与split身份摘要。JSON结构复核：27行、组合唯一、SHA格式有效，旧18项哈希全一致；清单SHA-256=`b4873c60aa60a1e87216ecf00062747eae2935bc985a92e28c916f05007b3e78`。
- 本次服务器只读状态：通过`zjnu-hpc`/tc6000 SSH key进入登录节点；`nvidia-smi`在登录节点不可用；账户 PBS 队列只有`214221.tc6000`（CPU队列排队，未提交新训练）。这符合现阶段训练已完成、继续闭合测试来源/推理门禁的安排。

## 代码指纹（SHA-256）

- `scripts/evaluate_hubei_forecast_heads_frozen_test.py`: `2e8bc42d5d22fc8b972a9d1f93d3112643c048cbeb6ba29d49aea71ccc660cf3`
- `scripts/hubei_forecast_metrics.py`: `6189f17d97a642a4a4c1a9c9210a5ae023bde5d7cf738b5169e5edaa9756004a`
- `scripts/test_hubei_forecast_heads_frozen_test.py`: `b76f01e8b5cfc3c781063e4d6ce4aa85cc3c6ee1ba17606b26158d49e164ab8e`
- `scripts/test_hubei_forecast_metrics.py`: `373ae3ecbcbaa588ce3f20ae8b92893831d9065ed86eb521a12f8d3659811688`

## 边界与下一步

- 真实test仍维持 `FAIL / NO-GO`。新鲜独立复核已确认代码修复，但评估器仍明确拒绝非合成数据；本轮新建的head manifest只闭合27个FA/FAC/FAP checkpoint，不等于完整模型链或测试放行凭据。
- 仍需补齐：真值manifest对原始工作簿、标签构建器、15分钟时间窗口合同及来源接触裁决的绑定；把全训练SimVP与冻结R checkpoint/norm/配置、精确测试输入builder、推理源码和receipt逐文件绑定；确保一个共享的一次性release ledger在任何建包/推理payload访问前原子领取（当前账本仅保护scorer阶段且checkout-local）；冻结bootstrap及指标依赖；按计划决定月份/太阳高度/云况分层；补找学院AGRI镜像缺失的2026-06-14/15输入源或在不读取test载荷的独立审计中裁定可用支持。之后再进行新鲜release审计。
- 已完成 fresh independent code review：`audit_receipts/test_chain_audit_20261018_v3/EXPERIMENT_AUDIT.md`。复核确认前述合成门禁与bootstrap修复，但还指出链级ledger、测试来源/6月旧回放接触风险和真正推理receipt尚未闭合。新head manifest是在该复核之后单独创建，当前尚未被独立审阅。
- 湖北A/AC/AP及forecast FA/FAC/FAP的三站三seed训练和验证已完成；当前不应重复训练。下一项应是绑定full-train S/R的历史来源与真实测试源元数据、实现建包/推理前链级ledger与受门禁builder，而不是新的训练seed。远端当前共享AGRI镜像仍缺2026-06-14/15日期目录；仅从已知tc6000可读目录未找到旧`/home/Data_Pool_3/chenyi`树，FD-201/107现有SSH key本轮也均未能建立连接，未尝试明文密码。
- 仪器校准元数据仍未找到。现有数据处理只继承原工程的W/m²操作合同；结果范围需明确限定为该操作量纲和当前15分钟时间聚合定义。

## 2026-10-18 追加：计算节点访问与Hubei镜像复查

- 修正此前一次探测中的SSH别名大小写：配置名为`FD-201/FD-107/FD-172/FD-248`。对用户明确允许尝试的FD-201、FD-107、FD-172分别做密钥模式与密码优先交互模式连接；三者均在认证提示前由远端关闭。由于没有出现密码提示，没有向远端发送口令，故不能据此判断口令正确或错误。FD-248未尝试。
- 通过`zjnu-hpc`跳板到FD-201/FD-107的只读SSH探测均在SSH banner阶段超时。学院tc6000账户当前PBS快照仍只有`214221.tc6000`一个排队CPU任务；未提交新作业。登录节点没有`nvidia-smi`，当前无法由此快照盘点已分配GPU。
- 只检索可读的学院Hubei AGRI镜像路径和文件名（不打开像元载荷）：`HuBei_AGRI_Preprocessed/data/2026/202606/`目前仅见`20260611`、`20260612`两个日目录；Hubei该镜像树下没有文件名命中`20260613/14/15`的文件。Hunan目录内同日文件不等于Hubei站点空间patch，不能拿来替代。
- 该结果只限定于当前可访问的tc6000镜像树，不证明FD节点、未登记挂载、离线备份或其他账户目录中不存在相应数据。训练和validation已完成，不因节点访问受阻重复训练；真实test仍NO-GO，test/buffer载荷没有读取。
- 本机路由诊断补充：`10.158.205.201:22`在当前Meta隧道默认路由下TCP可建立，但SSH在服务端版本横幅/key-exchange前断开；强制绑定本机`WLAN`源地址`192.168.31.191`时则TCP连接超时。未修改路由、代理或VPN配置。该差异表明当前网络出口/隧道路径是优先排查项，不能归因于密码验证。

## 2026-10-18 追加：更新登录验证与测试日镜像覆盖口径

- 收到用户提供的节点口令后，重新以“密码优先”交互方式探测 FD-201、FD-107、FD-172。三者仍均在 SSH 密码提示出现前由远端关闭连接；因此口令没有发送，不能判断其正确性。FD-248 未尝试，且未更改本机 SSH、路由或代理设置。
- 更正镜像覆盖表述：之前“只见 20260611/12”仅描述 `2026/202606/` 月目录，不代表整棵学院 Hubei 镜像只覆盖两天。按冻结的 22 个测试日期逐日只检查路径、文件名和文件数（未解码数组），其中 20 天有镜像文件，缺 2026-06-14/15；共 2,773 个文件名键。每日文件数与候选支持上界见 `TEST_TARGET_SUPPORT_METADATA_ONLY_20261018.md`。该文件另记录 1,305 个候选初始化、10,020 个共同站点目标键上界；这些是元数据上界，不是已冻结的真实评分键。
- 因缺失的两天恰在冻结测试日，且真实测试源、builder/inference与链级一次性账本门禁仍未闭合，真实 test 继续 `FAIL / NO-GO`；本轮没有读取测试载荷，也没有提交训练或推理任务。
- 继续只读核对预测链模型来源：训练pack报告绑定的冻结R哈希 `1e6c58b9…f7d83` 与学院镜像内 `hubei_al_same_time_20261017/assets/frozen_R_20261016/best.pt` 的远端 `sha256sum` 一致。pack报告记录的SimVP哈希为 `022795dc…e7be2e86`，但它不等于旧Hubei基线 `best.ckpt` 的哈希 `db620b62…b9471d0`；在本轮限定的四个Hubei实验目录中按文件名/哈希只读检索，尚未定位到与 `022795dc…` 匹配的 checkpoint 文件。因此 SimVP完整链资产的路径/配置/归一化仍需闭环，不能把旧基线冒充该forecast pack的上游权重。
- 对可访问的stage产物继续做文件级哈希核对：pack报告声明的`source_stage_report_sha256=e76c689f…718f2832`与当前持久stage的`stage_report.json`哈希`0826822a…cc74899b`不同；pack声明的`source_stage_index_sha256=e91693d4…956c763`也不等于持久`index_trainval.csv`哈希`5ccdc3d9…f8215a43`。当前持久`stage_manifest.csv`哈希`73e0f919…e47571a`与stage报告中列出的哈希一致，但stage报告指向的原始索引在`/tmp`临时目录，且该临时目录当前不存在。除四个湖北目录外，又在整个`irradiance_forecast`树内按`stage_report.json`和`index_trainval.csv`文件名核对哈希，仍未找到pack声明的source report/index原件；四个湖北目录内的JSON文件也未找到pack所列的forecast-COT bank报告哈希`64b8d282…6782a50`。故forecast pack引用的上游模型、索引/report及bank报告仍未能在当前持久资产中逐文件闭环。
- 只读检查forecast pack构建脚本`build_hubei_forecast_ghi_heads_pack_20261016.py`：它显式比较canonical SimVP索引、path-rewritten stage index/report、SimVP bank、forecast-COT bank的哈希及train/val-only状态；forecast GHI pack报告再记录这两类bank report的SHA和SimVP checkpoint SHA。当前持久目录保留了打包后的训练/验证阵列与GHI头，但缺少该脚本要求的原始上游bank report，且申报的COT bank哈希`64b8d282…6782a50`在本轮可访问的全工程JSON文件（<2 MiB）哈希盘点中没有命中。结论是来源链回执尚未可复现，不是测试评分器单点问题。
- 另按现有密钥模式探测FD-248；其在认证前同样关闭连接，未尝试该节点口令。当前tc6000实时PBS快照只有`214221.tc6000`一个排队CPU任务；无新作业提交。训练/验证资产不重跑，真实test仍关闭。
- 本机对真值候选清单中的四个原始源容器重新核验`Test-Path`、字节大小和SHA-256：月度ZIP、2026-04/05工作簿、2026-06-01至07-06工作簿、2026-07-07至07-14工作簿全部存在，大小与既有清单一致，四项SHA-256全部匹配。新增只做文件级校验的收据`audit_receipts/test_truth_source_inventory_20261018/local_source_hash_recheck_20261018.json`。这让原件字节身份可复核，但不等于读取/验证标签值，也未解决Apr–May未绑定trainval source report、设备单位凭证和2026-06-14旧回放接触裁决。
- 为补查冻结测试日期缺失的湖北AGRI，只在当前可访问的`/public/home/slfu/ttzhou/Auxiliary_data`按精确日期文件名搜索，过滤`HuBei`路径；未发现2026-06-14/15的湖北AGRI文件。命中的两份同日文件属于IFS NWP，不能替代卫星输入。未打开任何AGRI/IFS载荷；故20/22 test日镜像覆盖缺口仍未解决。
- 继续盘点tc6000可访问的`Auxiliary_data`目录层级：只发现湖北/湖南预处理AGRI与IFS `raw_nc`，未发现独立FY-4B HDF/原始卫星归档目录。对`swc/irradiance_forecast`全工程的Hubei路径按6月14/15日期名再检索也无命中。当前该节点没有可用于补建缺失Hubei卫星输入的已登记副本；不以同日湖南数据或IFS替代。

## 2026-10-18 追加：最新节点凭据连通性复核

- 用户补充了除FD-248外节点使用的新口令。本轮按密码认证方式尝试FD-201、FD-107、FD-172；三台均在认证提示出现前由远端关闭SSH连接，口令未发送，故无法验证其有效性。未将口令写入命令、脚本、SSH配置或工程记录。
- 同时用现有`zjnu-hpc`密钥只读确认学院登录节点为`tc6000`、账户为`slfu`；实时PBS队列仍仅见`214221.tc6000`处于`Q`状态。未提交新训练、推理作业或改动服务器状态。
- 节点接入仍受SSH服务端认证前断连阻塞；这与口令正确与否无关。湖北已完成训练/验证不重复训练，真实test仍`FAIL / NO-GO`，本轮未读取test/buffer载荷。

## 2026-10-18 追加：学院镜像来源链与节点路由复核

- 从`zjnu-hpc`只读测试到FD-201/107/172/248对应`10.158.205.{201,107,122,248}:22`的TCP可达性，四个连接均超时；本机直连的SSH密码认证尝试同样未到认证提示。由此确认当前受阻点是节点SSH网络可达性，而不是已提供口令的验证结果。未改动网络/代理配置。
- 在可访问的`hubei_independent_trainval_stage_20261017`中只盘点checkpoint路径、文件大小并对训练/validation checkpoint做SHA-256（未加载权重）。发现`control/fold_1/best.ckpt`大小131,954,558字节、SHA=`e95a54f44c8d6e5b14935270f616e7a66c9d00921ef2c2ae4d9d2ebeac53871a`；它与forecast pack声明的SimVP checkpoint SHA `022795dc7b17f5b7ca545c630c1c666993233bfa215db3cce32efab6e7be2e86`不一致，不能作为同一权重的替代证明。
- stage目录共盘点到5个checkpoint路径（best/last等重复副本按路径计），哈希未发现pack声明的`022795...`。该结论仅限当前可读的train/validation stage目录；FD-201的原训练目录仍无法从本机或tc6000访问，完整SimVP来源链仍未闭合。

## 2026-10-18 追加：用户再次提供节点口令后的连通性检查

- 按本机SSH配置中的`zjnu`账号，以密码优先模式分别连接FD-201、FD-107、FD-172。三台TCP均建立，但都在SSH密钥交换阶段（`kex_exchange_identification`）被远端关闭，未到身份验证/密码提示阶段；口令未发送，因此不能判断口令是否有效。未尝试FD-248，未将口令写入配置、命令或工程文件。
- 该状态属于节点SSH服务/网络接入阶段阻断，不是认证失败。训练与validation已完成；本轮未提交训练、推理作业，也未读取test/buffer载荷。需恢复可完成SSH握手的接入路径后，才能继续核验服务器原始数据并补齐缺失影像。

## 2026-10-18 追加：缺失AGRI时次归档申请清单

- 将已审计的16个train/validation缺失UTC时次转换为BJT并按原FY-4B 4KM HDF命名规则生成逐文件检索清单：`AGRI_ARCHIVE_REQUEST_16_SLOTS_20261018.csv`。清单仅是请求/检索元数据，不代表文件已经存在或已下载；未更改冻结manifest，也没有替代、插值或打开任何测试载荷。
- 复核官方NSMC公开目录仍列有FY-4B AGRI L1全圆盘4KM产品，目录时间范围至少覆盖到2026-10-08；该目录不证明这16个精确时次均可调取，且下载仍需完成注册审核。官方入口：[FY-4B产品目录](https://satellite.nsmc.org.cn/DataPortal/en/code/FY4B.html)、[官方产品查询](https://satellite.nsmc.org.cn/DataPortal/en/support/search.html)。
- 通过现有`zjnu-hpc`密钥只读确认远端仍为`tc6000`/`slfu`，PBS当前仅有无关CPU排队任务`214221.tc6000`，未提交训练或推理。FD-201/107/172仍在SSH密钥交换前断连，故本轮不能从其原始目录补回HDF。训练/验证已完成的结果不重跑；测试链未通过release gate，test/buffer继续关闭。

## 2026-10-18 追加：替代接入拓扑核对

- 复核本地`SERVER_CONNECTION_GUIDE.md`：工程记录的FD访问方式是本机直接SSH到FD节点；文档中的`FD-107:2233 → 本机 → ZJNU-HPC:22`反向隧道方向是让复旦节点访问学院集群，不是从本机绕过当前FD节点的SSH握手阻断。
- 本机未发现2233/2022/22端口监听，当前没有已运行的该反向隧道可复用。未启动或改动隧道、VPN、路由；因此获取16个原始HDF的下一步需要恢复校园网/VPN的FD节点访问，或经正式NSMC授权查询/下载，不应把“跳板隧道”误当作已验证的替代下载路径。

## 2026-10-18 追加：缺帧对训练窗口的元数据上界

- 新增可复跑脚本`tooling_20261017/audit_hubei_missing_frame_window_impact.py`，仅读取已验收逐帧manifest、冻结日期表和16时次归档请求清单；校验UTC→BJT与缺帧均属train/validation后，比较同BJT日连续24个15分钟槽位在“现有文件”与“假设16帧全部恢复”下的窗口数量。没有读AGRI数组、站点标签、CPP/COT、预测或test/buffer数据。
- 结果为最多233个额外连续24帧候选窗口，分布在11个冻结train日；validation新增上界为0。**233只是未套用24帧全窗day-mask、有效像元/质量条件前的元数据上界，不是可直接加入训练的样本数。**因此补回文件可能扩大训练支持；如来源恢复且逐帧QA通过，必须另建数据manifest并评估/重训受影响链路，不能热替换已完成run。
- 脚本本地执行通过，输出状态`METADATA_ONLY_UPPER_BOUND`；这项核算不改变当前冻结数据、模型、队列或test gate。
- `qstat -f`确认唯一PBS任务`214221.tc6000`是`cot_interval_pack`，属于`SolarCOTIFS_20260907/experiments/monthly_R_station16_20261013_v1/interval_alignment_20261015`的8核、6小时CPU tensor-pack任务，当前排队；它不是湖北GPU训练任务，保持原样未干预。湖北A/AC/AP与forecast FA/FAC/FAP既有三seed训练/验证不重跑。缺失测试输入与完整模型链未闭合，真实test继续`FAIL / NO-GO`；未读取test/buffer载荷、未发起新训练或推理。

## 2026-10-18 追加：历史逐日产物与冻结测试日期接触核对

- 仅按文件名将本机旧目录`.codex_tmp/hubei_rerun_corrected/formal_npz/`中的42个`YYYYMMDD_ghi.npz`日期，与冻结split的22个test日期交叉；两者仅重叠`2026-06-14`和`2026-06-15`，其余20个test日期无该目录下的旧逐日npz。旧inventory把这两日记为每站11个issue、首末issue 08:15–18:00、legacy lead 0–225；这些是文件/清单元数据。
- 阅读历史评估脚本`rerun_hubei_corrected_horizons.py`（未运行）确认它把`formal_npz`预测与`station_observations_minute.csv`实测配对，并计算逐站、逐时效指标，日期循环为`2026-06-13`至`2026-07-14`。已有结果文件及`provenance.json`的mtime均为2026-08-08；provenance标记`station_direct_ghi_v3`、`ecmwf_mode=legacy`、`cot_source=predicted`、`issue_schedule=hourly_0815`，且标为offline replay。因此，6/14–15实测在本次10月冻结split前已进入一次历史评估，不能称为从未评价的盲测日。
- 另在`.codex_tmp/server_daily_all/逐日三方对比_20260601-20260714/`仅按目录项核验，发现`20260614_多云`与`20260615_晴天`均各有41个起报Excel、三方对比CSV/JSON与PNG；文件时间均为2026-08-08。本轮没有打开这些表格、指标文件或图像。该独立历史输出目录与上述旧评估脚本共同佐证两日历史评估产物已在冻结split之前生成。
- 本轮对工作区仅执行文件名/扩展名扫描：含`20260614`或`20260615`的205个路径中，没有同时匹配Hubei路径标记且扩展名为HDF/H5/NetCDF的候选卫星影像。此结论仅限本地工作区文件清单，不代表其他服务器或归档没有数据；未打开任何文件载荷。
- 只查官方产品目录的元数据，发现国家卫星气象中心 FY-4B/AGRI L1全圆盘500 m数据集目录覆盖时间自2022-06-01至2026-10-08/14（目录页面更新时间存在差异），官方说明在线下载需实名注册/审核。该目录是6/14、6/15缺帧的可追溯候选来源，但搜索结果没有逐文件确认这两个日期/目标时次可下载；账户授权与逐键目录确认未完成，也没有发起下载或打开影像。官方页面：[产品查询](https://satellite.nsmc.org.cn/DataPortal/en/support/search.html)、[FY-4B产品目录与注册说明](https://satellite.nsmc.org.cn/DataPortal/en/code/FY4B.html)。
- 本轮只读取了该历史脚本、provenance字段/非评分模型元数据，以及输出文件的名称/大小/时间属性；没有打开观测CSV、NPZ、CSV评分内容或图像，也未运行历史脚本。现有证据证明历史评估流程使用过这些日期的实测值，但不能证明任何人查看过某个具体数值或由此选择了本轮模型。
- 该发现不改变既定test日期，也未据此挑选或删除样本。真实release前需由统一合同明确如何报告两日（保留原22日并披露历史评价接触，或经用户批准预先改为其余20日并完整披露），真实test仍未放行。

## 2026-10-18 追加：缺帧候选UTC/BJT键更正与重新核对

- 收到用户说明除FD-248外节点使用同一新口令后，按本机`zjnu`账号探测FD-201、FD-107、FD-172；均在SSH密钥交换阶段被远端关闭，未出现密码提示，口令没有发送，不能判断口令是否有效。FD-248未尝试。另经现有`zjnu-hpc`密钥确认登录节点为`tc6000`/`slfu`，队列仍只有无关CPU任务`214221.tc6000`（Q）；没有提交训练/推理。
- 审计`build_hubei_frozen_simvp_index.py::official_data_path`后发现学院`HuBei_AGRI_Preprocessed/data/`缓存键按UTC文件时刻组织。镜像中的`202606111615.npy`、`202606111630.npy`实际对应UTC 6月11日16:15/16:30，冻结帧manifest已把它们作为BJT 6月12日00:15/00:30验收时次；此前将它们误配为缺失UTC 08:15/08:30属于时区错误，现已在`AGRI_GAP_RECONCILIATION_20261016.md`追加更正，不再把这两帧算作补档。
- 按规范UTC键在可访问学院镜像和baseline派生数据树中逐项检查16个缺档，共32个精确`.npy`路径全部不存在。此前同数字检索命中的6个文件按工程索引规则属于UTC键，不是对应这些缺档的证据；该说明已更正到`AGRI_GAP_RECONCILIATION_20261016.md`。此前2+6个文件名线索均不计作缺档候选。
- 未读取test/buffer载荷，未改manifest、索引或checkpoint。已验收AGRI仍为2,644/2,660；原工程单位/旧5分钟标签和本工程15分钟右闭窗口定义不变。当前要补齐须取得精确原始HDF或有逐文件来源凭据的官方预处理；没有新合规数据，故不重跑已完成训练。

## 2026-10-18 追加：本地冻结测试评估器合成回归

- 在工作区本地运行`test_hubei_forecast_heads_frozen_test`、`test_hubei_forecast_metrics`和`test_hubei_forecast_pack_gate`三个测试模块；33项通过。覆盖合成夹具下的release fail-closed、路径限制、一次性scorer ledger、共同键支持、指标复算和两日bootstrap等行为。
- 这只是代码/合成夹具回归，不是湖北真实test授权、全链路数据访问账本证明，也没有读取真实test/buffer载荷或产生可报告的测试指标。包含PyTorch依赖的另一个合同测试模块未在本机运行，因本机Python环境没有`torch`；不得将其记为通过。
- 通过`zjnu-hpc`只读复查，当前仍在`tc6000`/`slfu`，PBS仅有无关CPU任务`214221.tc6000`排队；没有可运行的湖北GPU训练任务，本轮未提交作业。FD-201/107/172本轮仍在SSH认证提示前断连，提供的口令未发送、有效性未知，FD-248未尝试。
- 下一步仍须取得可追溯的湖北6月14/15测试AGRI来源、裁定6月14日历史评价接触，并完成完整S/R/H推理链、链级一次性ledger和独立测试真值/构建器回执；这些门禁未通过前不解封真实test，也不为重复训练占卡。

## 2026-10-18 追加：NSMC测试日文件计数接口探测未校准

- 官方在线目录页列出FY-4B AGRI L1全圆盘4KM产品，产品级元数据显示覆盖2022-06-01至2026-10-17；这是产品集的总体范围/计数，不证明某个精确日期文件可检索。[官方目录](https://data.nsmc.org.cn/DataPortal/cn/data/dataset.html?satelliteCode=FY4B)
- 只读查看官方页面公开JavaScript后，尝试使用`/DataPortal/v1/data/selection/file/subcount`按UTC日请求2026-06-14/15，并以06-13/16作阳性对照。四日均返回`FILECOUNT=0`，所以该次请求参数尚未校准、阳性对照未通过；这些0值**不得解释为档案缺失**。没有登录、下单、下载或请求文件/影像内容。
- 本次不改变测试来源结论：6/14、6/15的湖北测试AGRI在已知学院镜像仍缺；官方精确时次可检索性仍未知。后续应先用官方界面/正确查询参数得到一个可预期的正计数控制，再查两测试日，避免把错误查询当作数据不存在的证据。

## 2026-10-18 追加：新提供节点口令后的SSH握手复查

- 用户说明除FD-248外节点可使用新提供的`zjnu`账号口令。仅对本机已配置的FD-201、FD-107、FD-172别名做一次密码优先SSH连接检查；三者均在SSH密钥交换阶段被远端关闭，没有到达认证/密码提示，因此口令未发送、有效性未知。FD-248未尝试，口令未写入命令、脚本、SSH配置或工程文件。
- 当前从本机到FD节点的握手路径仍阻断，无法据此访问原始数据目录或补齐湖北影像。现有已验收训练/验证结果不重跑；测试链继续`FAIL / NO-GO`，未读取test/buffer载荷、未提交作业。下一步需恢复FD节点SSH握手通路后继续查找6月至7月14日原始AGRI；同时继续使用本地可访问的官方归档检索，但其精确文件计数接口需先校准阳性对照。

### 2026-10-18 09:55 BJT：复旦SSH入口与学院侧路由复查

- 对FD-201进行一次详细SSH握手诊断：TCP连接成功，但服务端在交换SSH版本/密钥阶段立即关闭（`kex_exchange_identification`），未进入口令认证；提供的口令没有发送，不能判定其有效性。FD-107与FD-172同样关闭在SSH认证前；FD-248未尝试。
- 从可达的`zjnu-hpc`登录节点只读探测同三台FD节点22端口，均为`TCP_CLOSED_OR_FILTERED`，未发现可复用的直连/跳板路径。当前PBS队列只有无关CPU排队任务`214221.tc6000`；登录节点未提供`nvidia-smi`，没有分配GPU作业。本轮没有提交或重跑训练。
- 训练/验证已完成的A/AC/AP、FA/FAC/FAP资产与既有结果保持不变；16个AGRI缺档仍未补回。下一步是恢复复旦SSH握手或通过正式归档取得精确源文件，再建新manifest并判断需否重训；缺源期间不插值、不替代、不为重复占卡而重跑既有数据合同。

## 2026-10-18 追加：预报时COT主张审计

- 用确定性证据预检对照4个验证差值，4/4均在对应机器收据中找到；独立result-to-claim审查结论为`no`（同模型家族、provisional）：现有证据不支持“预测COT提升可部署湖北+15～+240 min GHI预报”。
- 区分了两类实验：同刻AC相对A/AP的validation点估计较低（AC−A −3.340、AC−AP −2.846 W/m²，日期簇95%区间均低于0），只能作为验证期诊断关联；实际forecast-input验证FAC−FA为+1.005 W/m²，FAP−FA为+1.830 W/m²，置信区间都跨0，且test链完整性审计FAIL。故不将同期关联写作未来预报增益，test仍关闭。
- 下一项依赖16个train/val源AGRI缺档的恢复，或预先另立完整排除合同；之后才评估是否需从随机初始化重训FA/FAC/FAP。此处不重跑现有训练、不读取test/buffer载荷。可复核材料见`../findings.md`、`.aris/traces/result-to-claim/2026-10-18_run01/`及`.aris/evidence_precheck_hubei_cot_20261018.json`。

## 2026-10-18 追加：NSMC缺档查询经阳性对照校准

- 按官方页面当前JavaScript复核真实参数名后，重跑公开查询：2026-06-13阳性对照返回95个逐文件记录，且精确找到08:15、08:30两个已知HDF文件名，确认`subcount`/`subfile`查询及UTC起始时刻匹配有效。
- 用同一参数逐日读取16个训练/验证缺档对应的公开文件名元数据。12个UTC日期的记录数分别为93或94；16/16个精确UTC起始时刻均无对应归档文件名。故早先“四天均为0”的未校准请求已被这次有效查询取代；新结论是阳性对照通过、公开在线目录未列出这16个精确时次。
- 新增机器可读回执：`audit_receipts/nsmc_gap_recheck_positive_control_20261018.json`。只读取门户文件名/大小/时间等元数据，未登录、下单、下载或读取像元。该结果仍不代表离线磁带、私有镜像或其他授权归档不存在这些文件。
- 当前没有新增可验收影像，活动manifest仍为2,644/2,660；不插值、不近邻替代、不热改已完成训练。数据源覆盖没有变化，故不重跑已有训练。后续如无新的原始归档入口，按当前合规样本保持训练/验证资产，并继续修复独立test链门禁，不把公开目录查询当作全局不存在证明。

## 2026-10-18 追加：冻结test两日的公开归档元数据

- 时间口径更正：冻结日期是BJT，不可把同日UTC目录直接当作完整test日。按`HDF UTC=BJT-8h`，两个BJT窗口分别对应UTC`6/13 16:00–6/14 15:45`与`6/14 16:00–6/15 15:45`。本轮重新请求官方`subfile`接口后逐项解析`ARCHIVENAME`起始时刻，确认每个UTC整日95/96，缺口均为UTC 17:00；映射到BJT后分别缺6/14、6/15的01:00，故两个BJT窗口均为95/96。先前96/96是把无时区标记的UTC文件名时间误按本机时区转换造成，数学矛盾已消解。回执记录完整查询参数、官方列表脚本哈希和按日文件名清单哈希；原始响应/完整文件名清单仍未持久保存，属于可复查的在线文件元数据、非像元载荷证据。
- 只读取了公开文件清单元数据；两日来源因此有官方目录候选，具备后续经审核账号取回的可能性，但本轮没有认证、加购物车、下单、下载或读取文件内容。
- 机器可读记录：`audit_receipts/nsmc_test_day_metadata_20261018.json`。这两日已有项目历史评价接触风险，不能因原始文件可申请就视作盲测或擅自改写冻结split。后续须先按现有审计处理历史接触，并完成source→builder→inference→scorer release gate；未通过前test/buffer载荷保持关闭。
- 新鲜独立审计此前确认矛盾时证据不足；本轮已用官方列表接口直接重查并按UTC字符串解析，恢复为一致的95/96计数，仍需独立复核更正后的回执。即使元数据覆盖口径已自洽，也不代表Hubei预处理镜像、真实test builder、真值来源、模型推理链和链级ledger已放行；真实test总体仍`FAIL / NO-GO`。

## 2026-10-18 追加：训练数据补档与是否重训决策

- 本轮没有取得新的 train/validation AGRI 原始文件；已验收清单仍为2,644/2,660，公开门户此前对16个精确UTC缺档键的有效阳性对照查询均未列出对应文件名。该结果只描述在线公开清单，不等于离线归档全局不存在；本机到复旦节点先前仍在SSH认证前握手阶段中断，未能从原始服务器拷回文件。
- 2026-10-18 01:36–01:39 UTC 通过同一NSMC `subfile`接口再次逐日查询了16个缺档键所在的12个UTC归档日，16/16仍无精确文件名匹配；daily list数量及按排序`ARCHIVENAME`计算的SHA已追加至`audit_receipts/nsmc_gap_recheck_positive_control_20261018.json`。同时新增只读可重放脚本`tooling_20261017/recheck_nsmc_agri_metadata.ps1`（SHA-256 `0f00be0d…8c067a33`），能复核train/val缺键和6/14、6/15 BJT窗口；默认不含全量文件名，传`-IncludeFileNames`时将列表输出到stdout。脚本现场运行返回gap exact matches=0，两个BJT测试窗口各95/96，缺口为当地01:00。未认证、下载或读取像元。
- 用同一16个UTC键在本地工作区按文件名全树检索，没有找到精确时次文件。2026-10-18 01:40 UTC从`zjnu-hpc`实时查PBS队列，仅有`214221.tc6000`（`cot_interval_pac`，8核CPU，Q状态），没有湖北GPU作业；因此不提交重复训练作业。当前已完成的A/AC/AP及FA/FAC/FAP三seed仅为现有train/validation资产，缺帧来源恢复后另建manifest再决定是否重训。
- 后续只读检查本机到FD-201的SSH：配置账号`zjnu`与用户此前明确提到的`maont`用户名均能建立TCP，但都在服务端SSH密钥交换阶段被关闭，没有到认证提示；本轮未发送或记录密码，节点访问仍受网络/服务端握手阻断。为推进补源，新增`NSMC_ARCHIVE_AVAILABILITY_REQUEST_DRAFT_20261018.md`，附16个准确UTC/BJT时次和正式询档问题；只是本地草稿，未向外发送。
- 随后对FD-201、FD-107、FD-172、FD-248四个配置节点并发做密钥模式SSH快照；四台22端口连接均被远端关闭（FD-201输出同上，另三台同样`Connection closed`），未到用户认证阶段。本轮未尝试口令。现阶段不能从复旦任一节点拷回缺档数据或检查各节点GPU/调度状态。
- 湖北同期A/AC/AP和forecast FA/FAC/FAP已有三站三seed训练与验证产物。本轮不重复训练，也未提交PBS任务；只有新取得并通过逐文件来源、像元/掩膜及索引QA的缺档数据后，才应建立新manifest并评估受影响模型是否需重训。
- 本轮实际推进是用官方接口重新确认test日元数据覆盖并修正UTC/BJT解析错误；没有下载或读取AGRI、test/buffer、真值或checkpoint载荷。下一步优先查询并获取缺失的16个train/validation时次；test日元数据自洽不代表训练缺档已补齐。在证据到手前不宣称“数据已补齐”。

## 2026-10-18 10:17 BJT：跳板路径与新口令后的接入复查

- 对FD-201、FD-107、FD-172以配置的`zjnu`账号做密码优先SSH连接；均在认证前关闭连接，没有出现密码提示，因此口令未发送，无法判断口令有效性。FD-248未尝试该口令。
- 另经已配置的`focus-cloud`跳板对FD-201进行只读握手探测：跳板公钥登录成功，但到目标节点的banner exchange超时；未向跳板或目标发送密码，也未改动任何代理/隧道配置。
- `zjnu-hpc`实时状态为`tc6000`/`slfu`；PBS仅有无关CPU任务`214221.tc6000`处于Q状态，没有湖北GPU任务。本轮未提交训练、未读取test/buffer载荷。
- 湖北train/val已验收AGRI仍为2,644/2,660，16个缺档未补；现有训练不热换manifest、不重复跑。当前下一步依赖可用原始归档入口或节点SSH服务恢复，随后对补回帧做来源与像元QA并建立新manifest。
