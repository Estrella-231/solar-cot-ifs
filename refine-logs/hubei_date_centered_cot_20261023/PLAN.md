# 日期内残差监督：未来预测COT的内部配对筛选

## 假设与相对既有工作唯一因素

215763只读重放证明：既有神经G3末轮拟合RMSE117.244→106.965，C_select102.889→111.353；拟合基线偏差−23.755、选模+15.123，学到平均+24.791/+22.696的修正。太阳可用性固定头干预仅改变选模约0.00026。不是从未加过AGRI或基线标量，不能重复将这些当新方案。

新因素仅为训练标签中的日期整体残差是否去除。保持原冻结G1、307输入槽加G1标量、9922参数ConditionalHead、两档未来COT值输入、相同C_fit/C_select和预算。原损失/新损失各一对：raw_no_future_COT/raw_future_COT，center_no_future_COT/center_future_COT。只在本四组内部解释目标变换，不把旧不同选模/可用性结果直接当单因素对照。

## 数学定义

仅C_fit每站每seed：冻结G1预测b_i、实测y_i，目标北京时间日期d；m_d=mean_{i in C_fit,date=d}(y_i−b_i)。raw目标为y_i；center目标为y_i−m_d。训练仍最小化mean[((pred_GHI−target)/1000)^2]，等价要求修正学习(y_i−b_i)−m_d。m_d不进入特征、门控、C_select目标或推理；C_select与后续任何评分都对原实测y_i，不作偏差扣除。

本假设可能去掉真实、有用的日尺度云衰减，也可能仍未学到变化关系，因此不能预设有效。变换只定义有正负的残差监督，y_i−m_d可能为负，不称其为真实物理GHI；记录负目标数量并保留原输出clamp。没有采用COT均值/p90输入，没有用验证观测选择修正方向。

## 共同训练/选模规则

两档损失×有无未来COT四组共享同初始化、同C_fit行序、batch1536、AdamW lr1e−3/wd1e−4、30epoch、cosine eta_min1e−5、每组60updates，完整训练，不按组改变预算。每站3seed42/43/44，固定G1不更新。其C日期预测复用215763已核验保存值，归一化保持既有仅train拟合来源。与旧训练不同的是本四组共同修复太阳available=原valid且cosSOZ>cos80、3×3全部可用，否则精确回G1；原未来COT/diff/valid槽在不满足角度处置0，无COT臂只再置零未来值与diff18槽。全部日期和标签保留。

四组都在C_select按原GHI RMSE选模，必须包含epoch0精确零修正，平分时保留较早epoch，保存best和含optimizer/scheduler/updates的last。这项共同规则是修复已知选择遗漏，不把旧不含epoch0结果当本raw配对臂。C_select已使用，只能开发筛选；不打开official val、新时段或test/buffer。

## 事前判定

三站三seed等权RMSE，5000次目标日期块配对，seed20261023。只有center_future_COT同时优于冻结G1、center_no_future_COT及raw_future_COT（各负点差、95%区间上界<0），至少2/3seed改善、相对G1每站退化≤1%、GHI≥800平均绝对bias不增加>5W/m²，才考虑冻结后一次外层开发验证。否则封存该目标变换，不调日期/权重/网络/阈值、不将退回零修正视为目标完成。

保存逐样本C_select预测、全部epoch拟合与选模误差、训练日期m_d、负变换目标计数、checkpoint与源哈希、配对行序/初始化/更新见证和CPU独立评分。预计分钟级三卡按站点并行；PBS≤3/GPU≤8，现有节点/环境，例行监控交给Luna。无论文修改。
