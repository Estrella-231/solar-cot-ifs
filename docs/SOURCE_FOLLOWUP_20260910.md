# CPP生产来源与新增IFS库存：恢复连接后的有界核查

用户已授权按来源追溯→空间表示对照→独立IFS审计推进。FD-107于本轮成功SSH登录；共享盘部分目录及NC读取缓慢/超时，不能把SSH成功当作数据读取已完全恢复。未提交PBS、未重建数据、未打开test或持续监控。

## CPP

实查/home/Data_Pool_3/yangzx/Code和Project仍为yangzx属主、700权限；没有绕过权限或访问私有内容。FY4B目录只有Model/Result/Sample/TP/TP_CPP.py，公开SmaAT-Unet/Code只有四个既有AGRI云属性训练notebook，没有新的基础NC生成记录。Global_CPP/Realtime一次浅层枚举超时；AGCPPD浅层为年份目录，未扫描数据全树。当前仍不能绑定实际Result/CPP writer、模型权重、归一化、源AGRI输入或几何/时刻注册。

已向用户请求具体生成脚本/批次日志路径。所需最小生产包：对应现有抽查NC的实际命令或作业日志、writer及配置、checkpoint/归一化、输入AGRI标识和写出前后张量映射。若权限限制，由生成者导出一份可读快照即可，不要求开放整个私有目录。没有该证据，不启动新的COT表示训练。

## IFS：出现有用的新来源，但尚未放行

新候选根/home/Data_Pool_3/chensr/ifs_hres_china/raw_nc，实查含2024/2025/2026；2024列有202404～202412，2025列有202501～202512。这是目录覆盖，不是逐文件完整性证明。

发现明确文件ecmwf_hres_hubei_hunan_20240402_t0000.nc、ecmwf_hres_hubei_hunan_20250804_t0000.nc。部分同目录包含.part.nc和不同命名版本，后续必须仅用固定产品模式并排除未完成文件，不混合命名产品。

logs/ifs_202404_202608.log头部记录2024-04-02下载请求：class=od、stream=oper、expver=1、type=fc、levtype=sfc、time=00:00:00，step=1..90逐小时与93..120每3小时，area=34/108/24/117、grid=0.25/0.25，param=169/164/176/167/168/134/228。日志时间为2026-08-12，属于历史回填下载，不能用作2024年的release/arrival。

20250804文件ncdump -h成功：time=100、latitude=41、longitude=37，含ssrd/tcc/ssr/t2m/d2m/sp/tp；ssrd单位J m**-2，有scale_factor/add_offset与缺测码，读取时需解包。只有time坐标，没有独立release/init字段；history显示2026-08-14的grib_to_netcdf转换，不能当作当年的发布时刻。请求范围覆盖两站，但本轮未成功完成坐标值与全部步长核验。

随后对上述2024/2025两文件读取time/latitude/longitude各30秒均超时，原错误保存audits/ifs_extended_source_20260910.json。仅确认2025文件header的100个时刻，不能将请求中的100步直接当已逐值核验的有效时间序列。不重复盲读。

## 公开发布时间证据与边界

已查ECMWF官方历史页面Version17（2025-07-02）：00UTC Set I-i的0～90h产品计划05:45～06:12 UTC分发，93～144h计划06:12～06:27 UTC；这不是00UTC初始化时已可用，也不是本站实际到达证明。

来源：https://confluence.ecmwf.int/pages/viewpage.action?pageId=540563877

该版本不能无条件覆盖整个2024～2026研究期。后续需补适用历史版本，区分计划可用性回放与实际到达回放；可采用有依据的保守发布上界，但必须明确假设，不伪造逐文件release。SSRD还需核查累计定义、相邻step及首步起点；不使用ERA5的不同累计语义替代HRES。

## 已做的独立实施准备

见COT_SPATIAL_REPRESENTATION_NEXT_20260910.md。现有缓存代码保存完整cot_log1p[sequence,station,lead,16,16]，HPC相应complete.json存在；可在来源门禁通过后按原head pack行键追加空间特征，不重跑SimVP/R。三组为有效容量控制的AGRI基线、四维统计、COT空间编码，具体输入映射/参数验算须在实现时冻结。本轮未宣称cache全量重验或新模型可训练。

下一步依赖CPP实际生产包和可读IFS有效时间坐标；IFS候选线索已推进，旧step1–4缓存不因此变成完整产品。
