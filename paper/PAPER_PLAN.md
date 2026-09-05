# 第二篇论文写作方案

## 论文一句话

固定 FY-4B/AGRI 的 SimVP 外推，研究显式云光学厚度表示在预测误差存在时的站点辐照度价值，并以四格对照检验其与 IFS 的互补。

暂定标题：**Cloud-Optical-Thickness Representations for FY-4B Solar Irradiance Nowcasting with IFS Guidance**。
中文：**融合 IFS 的 FY-4B 辐照度短临预报：云光学厚度表示的作用**。
标题暂不写 improves、physics-informed novel framework 或 robust；待结果与最近邻全文核验后定稿。

## 与第一篇的科学分工

第一篇回答外推随机演变如何建模；第二篇回答固定外推结果如何通过云光学表示更准确地映射到站点GHI。SimVP是控制变量，COT是主检验因素，IFS是互补条件。不能用第二篇替第一篇宣称diffusion有效，也不能靠重复第一篇条件表形成独立论文。

## 章节结构（英文与中文相同）

| 章节 | 要回答的问题 | 必须证据 |
|---|---|---|
| 1 Introduction | 为何好的外推图像未必等于好的GHI，显式COT值得检验什么？ | 文献与待检验假设，避免泛泛罗列模块 |
| 2 Related Work | COT短临、卫星/NWP耦合已经做到哪一步？ | 两篇最近邻全文对照；不得宣称首次使用COT |
| 3 Data and Forecasting Task | 哪些站点/区间/标签，信息何时可用？ | 同M0切分；COT/IFS新覆盖；CPP检索参考属性 |
| 4 Fixed-Backbone COT Representation | 如何从同一预测AGRI得到COT、隐特征、IFS与kt？ | 冻结S/R，4维c与同监督4维latent，GHI回算公式 |
| 5 Experimental Protocol | 如何单独识别COT、监督、容量及IFS收益？ | 四格、AL/ALN、wide、统一样本、日期块统计 |
| 6 Results | COT是否有用，在何种lead/云条件下有用？ | 逐样本重算结果；先主对比后误差传播/分层 |
| 7 Discussion | 哪些收益来自显式表示，哪里受检索/外推限制？ | 同监督结果、云边增强、低太阳高度、两站范围 |
| 8 Conclusion | 实验证据实际支持什么？ | 无证据不填写提升率、不预设结论 |

## 图表（4图4表上限）

- Fig.1：历史AGRI→冻结SimVP→预测AGRI→冻结R→显式COT/同监督latent→站点kt；旁接release-aware IFS，画init因果截止线。
- Fig.2：16 lead的A/AC/AN/ACN RMSE差与CI。
- Fig.3：R(real)与R(forecast)的COT误差随lead及厚度变化，对应GHI收益；不把oracle画成可部署方法。
- Fig.4：稳定/转变天气与一个代表性失败日，所有模型同日同轴。
- Table1：数据与来源、分辨率、split、样本覆盖和推理角色。
- Table2：核心四格+校准IFS+climatology的共同样本主指标。
- Table3：同CPP监督AL/ALN、容量匹配及显式表示的归因。
- Table4：COT/IFS交互、分站/分时效的增量与成本。正文版紧凑，完整种子和覆盖表进补充材料。

## 强调COT的正确写法

引言强调云光学状态与GHI联系；方法将COT来源、变换、空间统计和误差传递写清；结果先回答ACN对AN与同监督ALN，而非把IFS提升作为COT贡献；讨论说明CPP检索误差、散射、云边增强和同源信息限制。不要简单添加单调损失或Beer定律后自称物理一致。

## 写作顺序与状态

现在可写任务定义、冻结外推路径、待执行实验协议和空表。先写3/4/5，再在实验完成后写6/7，最后1/2摘要/8与标题。当前 `main.tex` 与 `main_zh.md` 是结构和核心公式同步的提纲骨架，不是结果稿，也未编译为PDF。
摘要暂用研究问题表述；没有实验数字时不能提交。服务器路径、哈希、旧错误和审稿修订仅写docs/refine-logs，不进正文。
