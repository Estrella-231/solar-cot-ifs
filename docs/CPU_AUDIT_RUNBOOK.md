# P200 CPU复跑说明

仅复用现存FD-107 Python，不安装依赖、不申请GPU、不启动训练。
SSH：FD-107（key-only）；Python：`/home/Data_Pool/zjnu/.conda/envs/swc/bin/python`。
服务器工程：`/home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905`。

在本地PowerShell执行以下原样命令，输出目录必须尚不存在：

```powershell
ssh -o BatchMode=yes -o PasswordAuthentication=no FD-107 '/home/Data_Pool/zjnu/.conda/envs/swc/bin/python /home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905/code/repartition_cot_manifest.py --source /home/Data_Pool_3/chenyi/GHI/cloud_retrieval_hunan/split_cot_hunan_chronological.csv --output /home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905/audits/p200_fresh_agent_split_v1'
```

预期：退出0；input_rows=22484；train=12630、validation=6954、test=2878，排除22；source_sha256=c81de43432f0563e892f1e2c5cd54a2b0b28bd09452f070d2a96d5b20ad9d7f1；status=CANDIDATE_TIME_SPLIT_ONLY。本地Windows与Linux CSV换行可能造成output_sha不同，先比逻辑行内容，不能把平台换行误判为数据变化。
若路径不存在、计数不符或命令失败，保留原始输出并报告，不临时修改脚本/路径后声称原命令通过。本检查仅验证CPU清单处理可复跑，不证明训练环境或CPP质量可用。
