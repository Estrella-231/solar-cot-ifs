# 去除拟合日期整体残差：有所缓解，但未来COT未增益

## 回答的问题

215763显示神经条件修正在拟合日期学到上调，到了基线已高估的C_select仍上调。新单因素是仅在C_fit训练目标中去掉每个目标日期的整体残差；推理及C_select评分不使用观测作偏差修正。不是新输入、加大模型或继续旧训练；数学定义及可能删除真实日尺度信息的风险在PLAN.md中事先固定。

四组raw/center×不读取/读取未来COT，共同使用原冻结G1、相同初始化、行序、batch1536、30epoch/60updates与9922参数。共同修复太阳available和epoch0选择，故只在本四组内部解释目标变换；不能将分数差直接归因于相对旧实验的单一变化。

## 内部筛选结果

三站三seed先各算RMSE再等权，单位W/m²：

|输入与训练目标|C_select RMSE|
|---|---:|
|同一历史COT基底G1，不修正|102.888750|
|原目标，不输入未来COT|102.871998|
|原目标，输入未来预测COT|102.873256|
|日期内残差目标，不输入未来COT|102.477216|
|日期内残差目标，输入未来预测COT|102.504780|

center+未来COT相对G1点差−0.383969（约−0.373%），5000次目标日期块95%CI [−1.705503,+1.063972]；相对同目标无未来COT点差+0.027564，CI [−0.428081,+0.510943]。相比原目标未来COT点差−0.368476，CI仍跨0。三项继续条件均未通过，因此不能声称已获得预测COT的GHI收益。

牛首center+未来COT相对基底点变化−0.739%，米庄−0.442%，王寨全部seed保持epoch0。center无未来COT在牛首、米庄反而略优；现有点估计不能把目标变换的效果唯一归因于COT。全部seed与日期、四个小时及高GHI偏差保存于scores_attempt1/SCREEN_RESULT.json；没有删站或选择有利时效。

## 选模与完整训练检查

每组完整训练60updates；best含epoch0，选择期间只看C_select原观测RMSE。王寨四组均选0；牛首raw两组均选0，center部分seed选6或10；米庄raw除seed42选1外均选0，center选4或5。不能将回到基底当作COT有效，也不把末轮替换为结果。所有初始化/行序/更新见证通过，best/last、优化器与调度器保留远端。

中心化只作用于训练残差，变换后的“y−日期残差均值”不是新的物理GHI标签；负值计数和12个训练日期偏移逐seed保存于COMPLETE.json。它们不进入C_select或未来推理。

## 决策

封存该目标变换，不打开official validation/新时段/test，不扫中心化强度、日期、seed或阈值。已有证据支持两点：原修正明显受日期偏差影响；减少这个捷径并不足以获得未来COT额外价值。后续不能继续把“再加基线标量”“继续训练”“修复太阳mask”当作尚未做过的主要方案；若另立实验，需针对未来COT的条件信息或训练日期代表性提出新假设，并公平保留无未来COT对照。

## 复现

PBS215764，浙师大学院node23三A100，C/exit0、36s。上游和G1未训练；读取既有C_fit/C_select张量，36个小头拟合。全部预测保存在replay/<Station>/seed*_Cselect.npz；独立math.fsum计分与记录差1.42e−14 W/m²，并核验四组配对初始化、行序、60updates、31个含epoch0日志及最小分数。日期簇重采样使用保存DATE_ERROR_SUMS.npz，不虚称另一独立bootstrap实现。

服务器完整根：/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_date_centered_cot_20261023。official val、新7–8月及test/buffer标签均未读取，未改论文。C_select是已使用的开发数据，不是独立确认；总体目标未完成。
