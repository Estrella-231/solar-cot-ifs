# 第一篇资产借鉴与隔离审计

检查日期：2026-09-05。本轮读取本地代码、文稿与结果审计；没有登录服务器核验文件现存性或当前GPU状态。

| 资产 | 本轮依据 | 第二篇处理 |
|---|---|---|
| 湖南时间/标签/空间合同 | 上层 AGENTS.md、IRRADIANCE_ERROR_PREVENTION.md | 完整继承，矛盾旧条目服从湖南主训练和禁止未来输入规则 |
| M0 manifest/审计代码 | experiments/hunan_paper_m0_20260903/README.md | 复用代码与已封存划分；新增COT/IFS覆盖后另存cohort，不能直接沿用计数 |
| M0 已登记服务器结果 | /home/Data_Pool_3/wangyc/irradiance_paper/hunan_shortterm_m0_20260903_r101_r103_v1 | 下一轮只读核验SHA和文件，不重建不同split |
| M0 本地报告计数 | 48,310序列；27,675/4,258/16,377 | 是既有报告快照，不是第二篇已完成统计 |
| R108/R109/R110/R111 | experiments/hunan_baselines_m1_20260903/README.md | 借鉴站点cache、无实测输入基线、共同样本评价器；旧读出头不能直接充当本篇已在forecast域训练的头 |
| 旧SimVP best.ckpt | solar-energy/results/hunan_simvp_qualitative_20260903_v2/EXPERIMENT_AUDIT.md | 明确湖北训练；不能进入湖南主表。本地尚未确认合规湖南checkpoint |
| COT输入构建 | experiments/cot_fixed_v2/dataset_cot_fixed_v2.py | 13 AGRI+cosSOZ/cosRAA/day_mask之外可能追加metadata/angles，不能只根据COTUNet16名字判断输入16通道 |
| COT推理脚本 | experiments/cot_fixed_v2/infer_cot.py | 默认路径指向湖北，且import dataset_cot；作为代码参考，不原样启动。以checkpoint config、实际依赖与norm向量核对 |
| COT输出范围 | dataset_cot_fixed_v2.py: COT_MAX=85 | 旧合同，不是湖南新模型默认真理；新CPP有效范围、饱和与缺测编码必须审计 |
| 第一篇现状 | 上层refine-logs/EXPERIMENT_TRACKER.md，R222等条目 | 最近记录是湖北有限val诊断，不能复制为湖南泛化或第二篇科学证据 |
| 双语正文 | solar-energy/paper/sections/03_data_task_evaluation* | 借鉴标签公式与章节组织；不复制diffusion贡献与未完成结果 |

## 必须落盘的真实合同

SimVP与COT每个资产须登记：server_absolute_path、sha256、训练区域/时间边界、训练manifest哈希、config/norm哈希、完整输入输出通道顺序/单位、冻结状态。所有训练头使用共同上游缓存，其哈希一致。

IFS登记每个来源的init_time/release_time/valid_time/stepRange/单位与文件哈希。选择 release_time≤forecast init 的最新可用周期。不得把文件名中的有效时刻当发布时间。

不根据现存目录名判断湖南训练。未找到合规权重时先查现有作业和配置，再安排一次湖南SimVP训练；不得因第二篇重复训练同一基线。

## 两篇边界

第一篇检验随机外推/innovation建模；第二篇检验固定外推后的COT表示与IFS融合。共享数据与基线必须披露，同一实验数字可明确作为共同基线，但不可把第一篇条件消融完整复制后声称第二篇独立贡献。若第二篇最终只有重复的+条件小幅提升，需要收窄为第一篇补充研究或重新界定科学问题；不能靠改标题制造两篇贡献。
