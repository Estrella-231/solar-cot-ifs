# 最近邻与论文定位（初查，2026-09-05）

## 技能来源

用户所指为 [ARIS: Auto-claude-code-research-in-sleep](https://github.com/wanshuiyin/auto-claude-code-research-in-sleep)。本地 `.aris/installed-skills-codex.txt` 登记来源 `C:/Users/29728/.codex/aris_repo`，当前使用安装到上层 `.agents/skills` 的 research-refine-pipeline、research-refine、experiment-plan。缺失的共享协议链接已从源仓库 `skills/shared-references` 读取。没有重装插件，也没有创建定时任务。

## 已查来源及边界

| 来源 | 已核查内容 | 对本篇的约束 |
|---|---|---|
| [Miller et al., 2018, Solar Energy](https://repository.library.noaa.gov/view/noaa/49543)，DOI 10.1016/j.solener.2017.11.049 | NOAA原始作者记录/摘要：云检索、NWP风与辐射计算耦合进行短临 | 云物理+NWP并非新概念；本篇区分固定学习式外推后COT表示收益 |
| [Nowcasting of Surface Solar Irradiance Based on Cloud Optical Thickness from GOES-16](https://www.mdpi.com/2072-4292/17/16/2861) | 出版商搜索索引显示COT预测及站点/区域COT用于SSI；页面本轮429，未全文读 | 必须作为直接最近邻；不声称首次COT短临，待补全文训练与消融合同 |
| [A physics-informed machine learning method for short-term solar radiation forecast with satellite-derived cloud optical depth](https://www.sciencedirect.com/science/article/abs/pii/S0306261925020148) | 出版商索引正文片段：预测COD并通过物理相关模型估计GHI；页面直接读取403 | 本篇不能仅以“物理引导+COD”申报新颖性；尚未完成逐表方法核对，不编造作者和DOI |
| [ECMWF radiation quantities](https://www.ecmwf.int/sites/default/files/elibrary/2015/18490-radiation-quantities-ecmwf-model-and-mars.pdf) | 官方文档搜索结果确认累计辐射以J/m²提供 | 审计真实文件的step/单位后差分，不对平均通量重复差分 |

## 可防守的贡献假设

把“COT增加物理信息”改为“固定外推下，显式检索监督的COT表示是否构成有益归纳偏置”。需要在同CPP监督的隐特征对照、相同forecast域训练与IFS四格对照下成立。以应用遥感实证贡献为主，不预设ML架构创新或任何投稿录用可能。

全文精读尚欠：两篇最新COT研究的输入、预测时效、地面观测输入、切分、物理映射和NWP融合表。正式引言和投稿期刊选择前补齐；当前不生成未经核验的BibTeX。
