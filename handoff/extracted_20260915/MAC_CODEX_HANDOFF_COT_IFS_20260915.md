# 第二篇 COT 云物理论文：Mac Codex 交接包

生成时间：2026-09-15（Asia/Shanghai）。本文是可直接带到 Mac Codex 的工作记忆，
不包含密码、私钥、数据、checkpoint 或逐样本预测。

## 0. 先给 Mac Codex 的任务说明

```text
继续 F:/CODE_Classify/辐照度预报/solar-cot-ifs 的第二篇论文工程。
研究范围固定为湖南、+15 到 +240 min、冻结 SimVP 外推 AGRI、COT 云物理属性和站点 GHI。
不要引入第一篇 diffusion 路线，不要使用湖北权重，不要打开 test，不要把 IFS 加入主表，
直到 IFS 的 release_time <= init_time、有效时刻和 SSRD 单位/累计口径完成审计。

开始前先读 AGENTS.md、docs/UPSTREAM_PROJECT_RULES.md、IRRADIANCE_ERROR_PREVENTION.md。
训练或 qsub 前执行 qstat -u slfu；账户所有项目合计最多 3 个未结束 PBS 任务。
不要重训或覆盖任何已有 A/AC、SimVP、COT-R 资产。所有新表示实验另建目录、冻结数据合同、
使用同 seed 的配对配置，保存逐样本预测、checkpoint SHA256 与独立 NumPy 复算。
```

## 1. 工程、分支和远端权威位置

| 项目 | 位置 |
|---|---|
| GitHub | `https://github.com/Estrella-231/solar-cot-ifs.git` |
| Windows 工作区 | `F:\CODE_Classify\辐照度预报\solar-cot-ifs` |
| Git 分支 / 已提交基线 | `main` / `596d07b01782be0d60281026abda2c13359c5708` |
| ZJNU HPC 论文根 | `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907` |
| COT 反事实运行产物 | `runs/ac_cot_counterfactual_20260912_v1/` |
| 当前本地交接说明 | `docs/MAC_CODEX_HANDOFF_COT_IFS_20260915.md` |

GitHub 仓库不保存数据、checkpoint、私钥、PBS 日志、逐样本预测或大体积审计数组。对于实验
复现，HPC 上述绝对路径才是权威位置。

## 2. 两台服务器：Mac 无密码 SSH 配置

将你已有的私钥安全放入 Mac 的 `~/.ssh/`。不要把私钥贴入聊天、仓库、共享文档或代码。

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519_fd107 ~/.ssh/id_ed25519_zjnu_hpc
```

在 Mac 的 `~/.ssh/config` 加入：

```sshconfig
Host FD-107
    HostName 10.158.205.107
    User zjnu
    IdentityFile ~/.ssh/id_ed25519_fd107
    IdentitiesOnly yes

Host zjnu-hpc
    HostName 10.65.129.39
    User slfu
    IdentityFile ~/.ssh/id_ed25519_zjnu_hpc
    IdentitiesOnly yes
    ServerAliveInterval 5
    ServerAliveCountMax 2
```

只读连通性检查：

```bash
ssh -o BatchMode=yes -o PasswordAuthentication=no -o ConnectTimeout=8 FD-107 \
  'hostname; whoami; pwd'

ssh -o BatchMode=yes -o PasswordAuthentication=no -o ConnectTimeout=8 zjnu-hpc \
  'hostname; whoami; pwd'
```

已知身份与用途：

| 主机 | 登录身份 | 已知 home | 主要用途 |
|---|---|---|---|
| `FD-107` | `zjnu` | `/home/Data_Pool/zjnu` | CPP/原始资料来源核查；部分共享盘读取慢且存在权限边界 |
| `zjnu-hpc` | `slfu` | `/public/home/slfu` | COT 论文代码、已冻结数据、checkpoint、PBS A100 运行 |

2026-09-15 快照：`zjnu-hpc` 无密码登录成功，`qstat -u slfu` 未显示未结束任务；
`FD-107` 在本次复查中被远端主动关闭连接。因此先检查网络/主机状态，不要因一次连接关闭而
修改 key、反复输密码或重建数据。此前 FD-107 的 key-only 登录已通过。

如 Mac 没有现成私钥，应在 Mac 生成新 key，并由已经能登录 FD-107/HPC 的终端把对应 `.pub`
追加到各自远端 `~/.ssh/authorized_keys`；不能用密码或把旧私钥传到聊天中替代。

## 3. PBS 操作流程（仅 ZJNU HPC）

GPU 任务只经 PBS 运行，不能直接在 login 节点跑 CUDA 训练。环境为：

```text
conda: /public/home/slfu/miniconda3
env: swc
python: /public/home/slfu/miniconda3/envs/swc/bin/python
queue: gpu
常用资源：node21, 8 CPU, 1 × A100 40 GiB
```

每一次提交均按此顺序：

```bash
# 1. 先实时查看全账户未结束任务；运行、排队、挂起都算。
ssh zjnu-hpc '/opt/gridview/pbs/dispatcher/bin/qstat -u slfu'

# 2. 少于 3 个未结束任务时，提交一个命名清楚、输出目录全新的 PBS 脚本。
ssh zjnu-hpc 'cd /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907 && \
  /opt/gridview/pbs/dispatcher/bin/qsub scripts/<new_job>.pbs'

# 3. 立刻确认 job id、状态和日志。
ssh zjnu-hpc '/opt/gridview/pbs/dispatcher/bin/qstat -u slfu'
ssh zjnu-hpc 'tail -n 80 /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/<job>.log'
```

PBS 模板的关键部分：

```bash
#!/bin/bash
#PBS -N meaningful_name
#PBS -q gpu
#PBS -l nodes=node21:ppn=8:gpus=1
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/<job>.log
set -euo pipefail
source /public/home/slfu/miniconda3/etc/profile.d/conda.sh
conda activate swc
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
GPU=$(/opt/gridview/pbs/dispatcher/bin/qstat -f "$PBS_JOBID" | tr -d '[:space:]' | \
  grep -oE 'gpu/[0-9]+' | cut -d/ -f2 | sort -nu)
test -n "$GPU"
export CUDA_VISIBLE_DEVICES="$GPU"
cd /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907
python -u scripts/<entrypoint>.py --cpu-check
python -u scripts/<entrypoint>.py
```

SSH 偶发在命令返回前断开。若 `qsub` 后连接中断，先运行 `qstat -u slfu` 并检查同名日志；
确认没有提交成功前，不能再次 qsub，避免重复任务。已完成状态 `C` 不计入 3 个任务上限。

同步轻量代码到 HPC：

```bash
scp scripts/new_script.py zjnu-hpc:/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/scripts/
scp docs/new_protocol.md zjnu-hpc:/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/docs/
```

不要用递归 rsync/复制覆盖远端 `data/`、`runs/` 或 checkpoint。每个新实验新建唯一输出目录，
对原始资产只读。

## 4. 不可违反的数据与评价合同

- 湖南是唯一论文训练、验证、测试区域；湖北只可作项目外历史复现。
- 固定 SimVP 任务是 8 帧输入、16 帧输出，真实时效是 `+15...+240 min`；旧 `lead=0...225`
  要换算成标准 `+15...+240 min`。
- `init_time` 是最后一张历史输入；`target_time=init_time+lead`。matched 文件名按 BJT，
  FY-4B HDF 查找按 BJT−8 h。
- 站点观测 GHI 只作标签和评价，不能进入输入。标签是目标结束时刻对应的 15 min 均值，
  `pred_GHI = pred_kt × clear_sky_GHI`。
- 所有方法必须在同一 `(station, init_time, target_time)`、同一实测 GHI、同一白天掩膜上比较。
- `forecast AGRI` 是 SimVP 外推云图，不是未来真实 AGRI；两者不能混报。
- IFS 只有满足 `release_time <= init_time`，且有效时刻、`ssrd` 累计差分和单位均完成核验后，
  才可进入业务因果评价。

详细规则在：`AGENTS.md`、`docs/UPSTREAM_PROJECT_RULES.md`、`IRRADIANCE_ERROR_PREVENTION.md`。

## 5. 已完成的科学状态

### 5.1 固定 SimVP 的 A/AC 多种子验证

- `A`：forecast AGRI + 几何/站点/lead。
- `AC`：同 A，额外拼接四个 `log1p(COT)` 统计量：区域均值、空间标准差、p90、中心像元。
- 验证集 99,849 行；original 与 QC1 各有 seed 42/43/44，使用配对同 seed 和固定学习率。
- 当前四维统计拼接不满足种子稳定性：original 的 AC−A 两站等权 RMSE 为
  `−1.0104 / −1.0806 / +1.0909 W/m²`，均值 `−0.3334 ± 1.2339 W/m²`；QC1 为
  `−0.5069 / +2.7029 / +0.7407 W/m²`，均值 `+0.9789 ± 1.6181 W/m²`。
- 所以不能写“COT 提升精度”，也不能写“COT 无效”；正确状态是：当前四维表示、损失和数据合同下
  的结果不稳定。

### 5.2 固定 AC checkpoint 的反事实归因

PBS `208492.tc6000` 已完成。seed 42/43/44 的原始 AC 前向同保存预测逐值一致；未训练、未读 test。

四里全 validation 行中：

| 干预 | 三 seed 平均 ΔRMSE (W/m²) | 解释 |
|---|---:|---|
| 屏蔽 COT 均值 | `−1.568` | 均值参与了过强的向下修正 |
| 屏蔽 COT p90 | `−1.539` | p90 参与了过强的向下修正 |
| 屏蔽空间标准差 | `−0.243` | 影响较小 |
| 屏蔽中心像元 | `+2.690` | 中心像元是有益信号，必须保留 |

预测晴天的四里子集有 3,046 行：中心像元仍有益，屏蔽后 ΔRMSE `+19.259 W/m²`；
训练集晴天中位 COT 替换有跨三 seed 一致的小幅改善 `−0.221 W/m²`。均值与 p90 在该晴天子集
跨 seed 不稳定。`AC(c=0)` 不是 A checkpoint，不能把 A→AC 差异唯一归因给单一特征。

权威文件：

```text
docs/AC_COT_COUNTERFACTUAL_PROTOCOL_20260912.md
docs/AC_COT_COUNTERFACTUAL_RESULTS_20260912.md
runs/ac_cot_counterfactual_20260912_v1/report.json
runs/ac_cot_counterfactual_20260912_v1/metrics.csv
audits/ac_cot_counterfactual_20260912_audit.json
```

### 5.3 COT/CPP 来源与 IFS 目前仍受阻

- CPP 是 FY-4B AGRI 关联的 COT/CER/CTH/CLP NetCDF 检索参考产品；COT 范围声明为 0–100。
- 已证实的下游问题包括旧裁剪约一格偏移、旧训练在 85 截断 COT、`finite` 不能代替官方 QA。
  下游空间匹配已修复/审计，但**CPP writer、模型权重、归一化、输入 AGRI 绑定、时间/地理注册和 QA**
  仍没有生产者证据，不能宣称官方 COT 真值或物理机制。
- 生产者相关路径部分为权限 `700`：不要绕过权限或做无界全盘搜索。需要生产者提供实际 writer、
  某一现有 NC 的生成命令/日志、模型权重、归一化、输入 AGRI 清单和 QA 说明。
- IFS 候选目录 `/home/Data_Pool_3/chensr/ifs_hres_china/raw_nc` 有 2024–2026 库存，
  但逐文件有效时间、历史发布/到达时间和 SSRD 累计语义尚未通过。现在不做 AN/ACN 融合主表。

## 6. 后续工作应该怎么接

当前不应直接大规模重训或开 test。先由用户或新的实验协议明确允许后，进行一个独立命名、
单因素、参数量匹配的新表示对照：保留中心像元，把区域 mean/p90 改成相对局地的关系特征，
例如 `mean-center`、`p90-center`，或使用带天气门控的空间编码。必须与 A、旧 AC 使用同一行、
同 seed、相同训练预算，并先做单 seed smoke/overfit/前向审计，再做配对三 seed。

IFS 线可并行做只读来源审计；在 release/arrival 合同未通过前，不提交融合训练。

## 7. 常用验证命令

```bash
# 到 HPC 后
cd /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907
git status --short
/public/home/slfu/miniconda3/envs/swc/bin/python -V
/public/home/slfu/miniconda3/envs/swc/bin/python -c 'import torch, numpy; print(torch.__version__, numpy.__version__)'

# 当前反事实结果是否存在、哈希是否可读
sha256sum runs/ac_cot_counterfactual_20260912_v1/report.json \
          runs/ac_cot_counterfactual_20260912_v1/metrics.csv \
          runs/ac_cot_counterfactual_20260912_v1/counterfactual_predictions.npz

# PBS 队列
/opt/gridview/pbs/dispatcher/bin/qstat -u slfu
```

