# 修正 CPP 后的湖南 R 训练

学院服务器根目录：`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907`。
SSH 别名 `zjnu-hpc`；运行环境 `/public/home/slfu/miniconda3/envs/swc/bin/python`。
只复用环境，新 R 随机初始化，不载入旧权重或旧归一化。

## 数据与模型合同

来源是 FD-107 `solar_cot_ifs_20260905/data/cpp_aligned_20260905_v2` 已验收的 sidecar。打包程序逐条核验原文件/sidecar SHA、时刻、站点、网格与参考掩膜。
train 11,978，validation 6,536；test 2,626 仅登记引用，未打开 payload。训练期计算输入归一化；缺失输入归一化后填零。
打包 target 为 COT/100；训练程序显式乘回 100 后 log1p，不把缩放值当物理 COT。
模型与 FINAL_PROPOSAL 一致：16 输入、base16 两级 U-Net，softplus 输出 log1p(COT)，有效像元等权 Huber(delta=0.1)，AdamW(1e-3,1e-4)，50 epoch/patience8，seed42。验证损失选模。
CPP 是检索参考，源 NC 批次与生成权重绑定仍未核实；本训练不解决独立真值问题，也不代表 GHI 预报收益。

## 文档复跑检查（登录节点，仅 CPU）

```powershell
ssh zjnu-hpc 'cd /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907 && /public/home/slfu/miniconda3/envs/swc/bin/python -m unittest discover -s tests -p test_cot_training_contract.py -v'
ssh zjnu-hpc 'bash -n /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/scripts/run_cot_repaired.pbs'
```

## 启动与门禁

数据包必须 COMPLETE、全部 SHA 校验匹配才可提交 PBS。先查 `qstat -u slfu` 和本目录日志，禁止重复作业。通过检查后执行一次：

```powershell
ssh zjnu-hpc 'cd /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907 && /opt/gridview/pbs/dispatcher/bin/qsub scripts/run_cot_repaired.pbs'
```

PBS 单 A100，全部 train/val 缓存在单卡，先 profile batch64/128/256/512/1024；取吞吐距最快 5% 内的最小 batch，上限保留每 epoch 至少12次更新。随后8个训练样本400步，eval Huber 必须降至初始的一半以内。门禁通过才重新初始化正式训练。失败停止，不自动绕过门禁。
每 epoch 原子保存 best/last，日志 `train.log`，输出 `runs/cot_repaired_seed42_v1`。
结束后保存验证逐像元预测/参考/掩膜、重算指标与 SHA。测试集关闭，不自动启动 GHI 头。
已有输出不可直接重启覆盖；失败保留现场并另命名运行目录。

## 验证状态

程序已通过学院 Python 语法检查。独立 agent 原样执行上述两条 CPU 检查：2项单元测试通过，PBS语法检查退出码0，未发现文档与环境差异。GPU门禁与正式训练仍须现场核验，不能把提交或排队当成开始训练。

## 自动衔接

Windows `scripts/continue_cot_launch.ps1` 等待 FD 打包 COMPLETE，核对样本数和测试未读取标记，再用 tar/scp 中转到学院。归档在源、本地、目的三处 SHA 必须一致；学院还重算每个数组和代码文件的 SHA。`submit_cot_once.py` 使用文件锁、提交意向记录与 job_id 文件阻止重复提交；提交结果不明时停止，不盲目重试。
本地运行状态：`data/cot_launch_20260907/status.json`。本机关闭或 SSH 失败会停止自动衔接；重新接手前检查状态和服务器 `submitted_job.json`，不能重复启动。该程序不是周期提醒任务，不发送消息。

## 2026-09-07 衔接恢复

数据包18,514条已完成，测试payload读取0，原始及sidecar全部SHA匹配。旧后台PowerShell在下载后找不到Get-FileHash而停止，尚未提交GPU作业；现改用.NET SHA256，并在相同powershell.exe环境中验证结果。直接从已下载归档恢复上传，源/本机/学院三处SHA均为 `df7b70e04d777ac2042bc7a200605868b711eb60926eeaa5a9c903a26e4e705f`。
学院重新核验全部数组和代码后，通过单次提交程序启动PBS207372.tc6000，node21 GPU0。后续状态查train.log及runs/cot_repaired_seed42_v1；不重复运行自动上传程序。
