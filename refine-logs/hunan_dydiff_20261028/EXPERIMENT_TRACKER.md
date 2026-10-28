# 湖南DyDiff推进记录

## 最新：216590八卡适配启动

### 接续216591及跨区域诊断

### 216594检查点对象通信修复

216591真实pilot完成两次反向传播（约2.24秒/步、第二步reserved35.91GiB）后，在rank0收集Python RNG状态的NCCL gather_object报unhandled cuda error并退出1；尚未到正式启动标记，正式训练目录不存在，不能把该pilot写成正式训练。改为单独Gloo CPU group收集小型Python RNG对象，DDP梯度/验证张量通信仍保持NCCL，模型/优化器/样本设置不变。旧失败日志保留。

重新核对slfu全队列、node21空闲8卡后，提交216594.tc6000。只复测受修改影响的8卡断点保存/恢复和真实pilot，复用216591每卡10容量结果，不再跑输入检查或跨区域16窗口。正式global80、每轮442更新、150轮66300更新。

湖北模型→湖南16窗口诊断已完成PASS，原始metrics/selection、hourly_metrics.csv与两季真实云图对比已回收；结果见cross_hubei_on_hunan/RESULTS.md。1/2/3/4小时C02 RMSE为0.07117/0.09879/0.12862/0.14587，C12亮温为5.565/7.158/8.481/9.942 K，不是COT或GHI误差。所有13通道相对末帧持平的同支持MSE skill均为正；不能据此称全验证、优于SimVP或已有GHI收益。

216590实际退出1，原因仅为合成恢复测试的Adam状态逐位一致断言：8192元素中1个相差4.235164736271502e-22，model/EMA逐位一致已通过。原失败日志/产物保留。verify_resume仅对optimizer moments改FP32数值容差rtol1e-6/atol1e-20，模型/EMA仍要求逐位一致；复用已完成合成测试而不重跑数据检查。

重新查队列和物理GPU后提交216591.tc6000，八张node21 A100、40CPU。修正后的恢复验证PASS；真实湖南每卡10容量PASS：34.701GiB reserved、27.563GiB allocated，global80。随后8卡真实pilot与其续训通过后直接正式150epoch。

用户新增直接用湖北模型试湖南的授权已执行：冻结原第150轮EMA及湖北归一化，Hunan val按月选16窗口并排除湖北train日期；在216591容量测试期间用其他7卡并行推理，未另占第9张卡。部分窗口已完成。见HUBEI_ON_HUNAN_PLAN.md；完整指标以对应metrics.json为准。

用户明确取消重复全量源数据检查，适配湖南输入后node21测试通过即8卡开跑。已完成code_hunan_v1：复用冻结湖南train/val、原始QA和同train归一化，修正NOAA几何按UTC及湖南grid重算，原图UTC连接BJT质量位，按需本地缓存；不重新扫描全部帧，不重新拟合归一化，不读test载荷。训练日志loss修正为跨rank求和再平均，与实际全局有效像元梯度一致。

提交前slfu全队列为空，node21八张A10040GB均0MiB/0%。PBS216590.tc6000已分配node21 GPU0—7，状态R，40CPU，8卡不等/空rank有效像元梯度测试已PASS。后续顺序为8卡断点精确恢复、少量真实批次容量及8卡pilot自动续训，然后随机初始化湖南正式150epoch。以launch日志的HUNAN_FORMAL_8GPU_START和正式run_metadata/progress为正式训练启动证据；仅PBS为R不能代替正式训练开始。原CPU216589保留只读。例行监控交Luna，本轮不创建/恢复自动化。

2026-10-28：用户授权湖南独立底图。source_hubei六核心SHA与同门epoch150来源相同。学院GPU/队列实时核对完成；node21空闲8 A10040GB，node23空闲7，node22无可确认空闲。CPU216589正常队列初请求node21未运行，核对后仅对自身Q任务qalter到node7四CPU，C/exit0约30秒。全元数据和6帧头/网格PASS，但1验证帧IR全部不在有效范围，完整数值QC未完成。source adapter、train norm、本地缓存与2/4/8卡真实吞吐仍待做；未开正式训练、未读test。见EXPERIMENT_PLAN.md、READINESS_CPU.json和RESOURCE_SNAPSHOT.json。
