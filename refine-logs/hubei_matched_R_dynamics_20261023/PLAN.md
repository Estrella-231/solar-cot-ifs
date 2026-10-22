# 同一折内反演器下的云厚演变信息

承接固定头替换、区域R来源和GHI小头四类负结果，不重复旧validation oracle或再调融合。新内容是将已确定C_fit扩展训练日期及C_select与真实目标云图完整连接，并以每条forecast实际OOF fold的同一R权重/归一化反演真实init和target。旧history bank用full R，不能直接当fold forecast的同R参照。

日期完全沿用hubei_cot_calibration_coverage_20261023/METADATA_AUDIT.json，不基于新误差选样。只读原split=train且wide-C_fit或C_select日期；不读F标签、official val、新7–8月及outer buffer/test载荷。每条station/init/target/lead连接原source_role和sequence index，历史8帧/目标所需raw patch都按prepared_attempt12/stage_manifest.csv的BJT/UTC及SHA绑定，不把缺失目标补成晴空。

原node22缓存提取16×16物理AGRI/三几何通道，固定原C01–C06/C09–C15和patch(103,118)。归档选定帧原文件hash及patchhash，只生成紧凑数据包；不改原缓存。node22单CPU低优先级预处理，不训练。冻结fold_0…4各自R/normalization，须与GHI_pack_attempt12/source_correctutc_Rs哈希一致；GPU三卡短推理并行折。AMP沿用原R推理。先用每fold一条原forecast patch重放R，事前门槛RMSE≤0.05、max≤0.25 COT；若超出先查来源/精度，不把不同R混成相同。

记录所有候选及可用率；共同有效像元比较forecast→R、init持平→同R与真实target→同R。报告1/2/3/4小时和全体的物理COT RMSE、bias、变化量相关、固定变化阈值±3 COT的增厚/消散方向检出与误报。station center主指标、3×3逐像元补充，不将全图统计作GHI输入；同目标/不同init不当独立天气样本。目标日期簇区间，并区分C_fit/C_select，后者仍为已见开发集。

生成全部日期数表和按纯日期规则选取的连续云图/COT轨迹（不按收益挑图）。任何同R真实图COT仍是反演参考，非独立真值；它不进入可部署预报或GHI输入。若参考显示可学习变化，后续再冻结独立COT预测/监督方案；此轮不训练，不以参考替换GHI成绩，不声明未来COT已获收益。
