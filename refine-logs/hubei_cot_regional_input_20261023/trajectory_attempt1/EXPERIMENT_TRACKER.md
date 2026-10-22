# 区域轨迹执行记录

|阶段|状态|证据|
|---|---|---|
|旧实验承接与假设|已固定|EXPERIMENT_PLAN_20261023.md；不重开小裁块损失搜索|
|train内日期角色|元数据已计数|19420拟合/2840内部选模配对，未来/历史日期均在同一角色|
|历史R训练材料|待准备|复用原始本地AGRI，不读取共享NAS，不重训R|
|CPU/GPU回溯及DDP核验|未完成|不能以计划代替实际通过|
|六个配对拟合|未启动|同结构、同损失、三种子；内部日期选模含step0|
|未来COT及GHI|未完成|COT判据未通过前不训练GHI|

## CPU与来源准备完成

CPU_WITNESS.json通过：19841参数，初始full/repeat_last都精确等于共同输运基线；改变旧历史只影响full特征，不影响repeat_last；PyTorch回溯和独立SciPy最大log-COT差1.79e-7；掩膜外NaN反向有限。此项不等于GPU/DDP通过。

train内19420拟合/2840select配对固定后，4598张所需历史图及aux从node22本地到node23本地转运，27,734,548,480字节、626.443秒，两端exit0；9196文件hash匹配。目的/dev/shm/hubei_region_trajectory_train_20261023_attempt1。无GPU源读取、无外层val/test载荷。

实时全账户空队列且node23五张空闲A100后，提交215754.tc6000（单PBS四卡）。该作业仅用冻结区域R生成train历史缓存、复核完整太阳几何并CPU构建配对包；不是轨迹模型/GHI训练。正式新候选训练程序已实现但未通过GPU profile，不能称已开始拟合。profile/正式训练将在新独立目录保留。

## 215754缓存完成、CPU枚举错误与独立修复

215754终态C/exit1、约99秒。四个缓存分片各自PASS，1150/1150/1149/1149帧，太阳几何最大差均0；错误发生在CPU build_pair_pack把shard0.log当作目录。未将此工程失败解释成模型负结果。原代码、369MB未完成attempt1包和日志保留。

仅把目录枚举限定为is_dir，在node23通过CPU repair_pack.py写全新/dev/shm/hubei_trajectory_pair_pack_20261023_attempt2。四个缓存SHA重新核对；4598历史帧、1853目标帧、3723起报、19420fit/2840select配对完成。来源/日期/标签/运动算法未改，未重跑GPU。新包使用原CPP物理值，几何仅由时间网格计算；外层val/test/GHI读取0。

215754的PBS C记录已被队列清理，profile提交初次读取qstat遇Unknown，在任何qsub/intent前退出。提交器改为核对活队列无该任务、四缓存与CPU修复实际通过后记录purged状态，不把Unknown当失败或重启理由。

实时空账户队列与四空闲GPU核对后提交215755.tc6000，node23四A100，执行完整2/4卡masked物理损失/梯度、AMP/断点恢复及吞吐短测。正式六个候选拟合仍未启动。代码提供run_paired_fits.py和submit_fits.py，须待profile通过且重新查队列后方可提交；不会从profile权重续为正式模型。

## 215755多卡测试未通过，固定同一权重重验

215755已C/exit1，186.34秒；2/4卡FP32及AMP各自完整断点重放均差0，AMP吞吐574.06/1274.43帧/秒。但跨卡gradient_relative_l2=.001467超过原.001门槛，最大绝对差3.93e-6、损失相对差2.81e-6通过；未生成选择文件，未正式训练。原全部输出保留。

发现原检查在2/4卡各自先更新13步后比较梯度，模型参数状态本来已经不同；实查参数最大差.000197336。不能把这项差异直接当同权重下的梯度聚合误差，也不能直接归因全部数值差。本次不放宽阈值，新增--witness-init仅供witness加载同一已有非零encoder梯度的checkpoint，绑定数据/模型/seed/arm/global batch；在完全相同起始权重下各做一个global update比较梯度与损失。正式训练仍从相同随机seed初始化，不用canonical参数。

215756.tc6000已重新实时查空队列及物理空闲卡后提交node23四A100，profile_attempt2；FP32两组记录相同shared_state_sha256，并重新测试AMP吞吐和恢复。新输出目录不覆盖attempt1。正式train参数、loss、样本/预算/选模规则未改，门槛仍相对梯度<.001、绝对<.0001、损失相对<.0001。需新实际结果才能继续。
