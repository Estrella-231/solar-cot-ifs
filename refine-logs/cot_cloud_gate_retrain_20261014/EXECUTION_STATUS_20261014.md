# 执行状态：B0/B1完成；B1未通过筛选门槛

更新时间：2026-10-14 BJT。判云分类器训练和独立验证审计已完成。B1未通过预注册门槛，B2及GHI重训未启动。

## 已完成

- 已读取工作区AGENTS.md、IRRADIANCE_ERROR_PREVENTION.md、上游规则、月度划分合同、判云实验协议、R合同和运行实验skill。
- split合同为`hunan_monthly_block_2d_buffer1d_seed42_v1`；现有R CPP训练包状态COMPLETE，训练/验证15414/1398，test_payloads_read=0。
- 固定GHI pack状态COMPLETE，训练/验证50458/4479，test_payload_opened=false，使用同一新月度R checkpoint；远端文件位于`.../data/ghi_head_pack_new_r_v1`，远端新结果不能误按`results/`路径查找。
- 执行`ssh zjnu-hpc qstat -u slfu`，账户仅有一个未结束作业`213898.tc6000`（normal队列、运行中CPU作业、node20、16核）；PBS未结束任务上限为3，当前仍有容量。
- 本地RTX 4060 Laptop GPU为8GB，快照使用约1.98GB；本实验计划优先使用集群GPU并按单卡先profile的规则执行。

## 阻塞门禁

判云监督必须逐像元使用CPP CLP，但现有R训练pack只保存`x_raw.npy`、`cot.npy`、`mask.npy`，没有CLP数组。现有CLP本地sidecar仅覆盖验证样本，不能用于训练。不能用COT是否为零替代CLP类别，因为云类中存在近零COT像元。

CPP原始NC路径记录在pack `rows.csv`，PBS计算节点抽查时不可读。用户随后明确允许使用107/172/201并提供登录凭据；凭据未写入工程。107仍在认证阶段断开，201密码认证成功且能读取CPP NetCDF及湖南`grid_static.npz`。201可见冻结R pack目录：`/home/Data_Pool_3/zjnu/cot_value_first_20261013/monthly_R_station16_20261013_v1/data/full_trainval_pack_v1`；其状态为COMPLETE、train/val=15414/1398、test_payloads_read=0，恰与当前R合同一致。pack含16通道x_raw、COT及监督mask，原mask已联合CPP有效CLP/输入有效/日间/SOZ<80支持。

## 继续执行

- 新增`extract_trainval_clp.py`，按冻结R pack `SUPERVISED` 行、候选样本键、站点坐标和湖南网格，从每个匹配CPP文件提取16×16 CLP；CLP无效值保留为255，不填作晴空。
- 已完成全量train/val CLP提取：8,469个唯一CPP时间文件，train/val=15,414/1,398，4,277,993个R监督有效像元全部匹配合法CLP（晴空1,362,424、水云1,937,389、冰云978,180），最大网格坐标误差0.0°。`test_payloads_read=0`。审计为`clp_trainval.audit.json`，侧车SHA-256=`361c79e40f40fd3c39c50b48e4c003b3fb64c90c32183e8304f54adc7b8cea01`。
- PBS版`pack_trainval_v1`与201的`full_trainval_pack_v1`逐文件核对：pack状态SHA相同，rows=`b270713590f6af793b53df73d27e01f3f62ad8d7e60bb9a8eaace5c43fb2ebcb`、mask=`2614ca0b589ff3e3367f08f0e8f8ed8067226f80446f4e27f3478d5c675c340f`、COT=`43dab24e9b6c5cf9b9db09f6bce16b3dede60750b312c6e808b192ec265fb63c`、x_raw=`06d4606623c1495d63ab015d778b5691d6f0c072e58fad79c6e559c84d450a31`；CLP标签与训练输入同键同数组。
- 新增`train_cloud_gate.py`，使用当前R训练归一化和冻结R验证预测，seed42、30轮、batch512；单卡profile、逐轮KSS选模、R-COT≥0.1基线和门控前后COT/厚云指标均写入结果。PBS脚本已通过`bash -n`，swc环境可导入PyTorch。
- 201六张4090复核时均在高显存占用；172虽有空闲显存但系统Python无PyTorch，未改服务器环境。首次PBS运行`213924.tc6000`完成了单卡profile和epoch1，但因日志调用把`flush`误传给`json.dumps`而以非零状态退出；该不完整输出原样保留在`results/cloud_gate_seed42_20261014/`，不用于模型筛选。profile实测A800 batch512为263.2步/秒，峰值显存0.409GB。
- 已修正日志语句，并令profile样本只从train行选取。作业`213925.tc6000`（`cot_gate_retry1`）正常结束，930次更新，30轮，按验证KSS选中epoch 24；独立输出目录`results/cloud_gate_seed42_20261014_retry1/`。单A800峰值显存783,962,112 bytes。训练脚本SHA-256=`155a3588a03521f55012c54640ca9cfdec655ff43688e5a0712de295a5c56a7c`，PBS脚本SHA-256=`dcbc299e216937e626ca31112f1010498cd0e3f5d71db155ab87db1a0aa816f7`。test仍关闭。
- 独立FP64复核覆盖1,398个验证样本、356,169个有效像元和46个BJT目标日期。站点等权COT RMSE为9.0531→9.0595（Δ=+0.0065，日期簇95%CI `[+0.0011,+0.0137]`）；云FAR为0.1261→0.1137（Δ=−0.0124）；四里云POD下降1.45pp（95%CI −2.45至−0.68pp），竺家上升5.35pp；两站厚云POD/CSI均不变。B1不满足COT RMSE和四里云POD门槛，故停止该候选，不调阈值、不启动B2、不训练GHI头。完整结果见`B1_RESULT_20261014.md`及`independent_audit_fp64.json`。

当前队列复核仅有`213898.tc6000`（他项CPU作业）仍在运行；本计划作业均已结束。B0/B1的标签sidecar和分类筛选已完成。按照预注册规则，当前没有后续训练动作；任何新方案必须另立合同。没有训练GHI头，也没有读取test。
