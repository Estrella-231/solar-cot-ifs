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
