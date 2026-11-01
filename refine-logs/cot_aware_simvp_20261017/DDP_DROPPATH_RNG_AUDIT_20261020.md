# DDP DropPath 随机流核查

时间：2026-10-20 01:53 CST。对象：正式 fold3 DDP 作业 `215034`。

## 发现

fold3 当前运行的 DDP runner 在两 rank 上以相同 seed 初始化 CUDA RNG。模型含 SimVP DropPath：每个样本各自抽取随机路径掩码，最大丢弃率为 0.1。由于两 rank 的局部 batch 形状和模型调用次序相同，两边对应局部位置可能重复使用相同掩码。此项是由代码路径推得的风险，尚未从运行时直接采样确认。

它不改变 OOF 日期划分、样本顺序、全局 batch=20、有效像元归一化损失、优化器更新数或同步 BatchNorm；但改变了逐样本 DropPath 掩码的相关性。因此，fold3 只通过了 batch/loss/SyncBN 等价见证，不能表述为随机正则过程与单卡完全等价。当前作业保持运行，不中断、不重启；若未来评估结果敏感，应在同一正式划分下补做独立纠正重跑。

## 后续折修正

新 runner `train_monthly_simvp_oof_ddp_20261020_rankrng.py` 在 DDP 同步初始参数后给各 rank 设置独立 CUDA RNG seed；worker RNG 仍按 rank 分离。CPU 语义测试、Python 编译和 PBS shell 语法已在服务器 `swc` 环境通过。双卡运行见证新增跨 rank 随机掩码不重复及丢弃率检查；该 GPU 见证尚未运行，必须等 PBS 获得两卡时执行。

fold4 DDP PBS 已写好并同步至服务器：seed46、相同 global batch20、固定12轮、独立输出目录，包含 fold4 train-only normalization/raw-cache 构建、DDP GPU witness、profile 门和正式训练完成收据门。服务器 fold4 日期清单计 27,237 train、6,116 heldout、0 test；其归一化文件目前尚不存在。作业没有提交：现场 PBS 仍为 3/3，必须等一个任务自然完成并复查全账户队列后再排队。

## 当前训练现场

2026-10-20 02:00 CST：`215021` fold2 为 epoch8/step9600；`215028` fold0 恢复为 epoch10/step11900；`215034` fold3 DDP 已完成 epoch1/step1377，`train_loss_mean=0.3542621`，耗时1317.35秒，并进入epoch2/step1500。首轮checkpoint已出现，后11轮粗估约4小时（以实测首轮线性外推，可能波动）。fold1 OOF bank完成6,430/6,430；其他折OOF bank和全量GHI A/AH/AC结果尚不存在。当前没有未来COT带来GHI增益的证据，继续以全五折同键GHI配对评价和日期块bootstrap为结果门。
