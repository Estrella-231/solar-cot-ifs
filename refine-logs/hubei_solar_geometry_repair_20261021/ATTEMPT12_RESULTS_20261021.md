# 湖北 attempt12：太阳几何修正后的预报链结果

## 几何与数据核验

train 和 validation 均已核对。独立 NOAA 几何角公式复核覆盖 62,802 个时空点：train 57,334、validation 5,468；天顶角最大误差为 `1.75e-13°`，方位角最大环形误差为 `2.72e-10°`。样本包审计覆盖 S 10,473 帧（train 9,560、validation 913）和 R 3,479 行（train 3,178、validation 301）；R 几何逐行差为0，S几何最大差为 `3.58e-7`，目标时刻差为0，train/validation不交叉。test和buffer载荷未读取。

## 预报 COT 验证诊断

评价链为历史 AGRI → 修正几何 SimVP 预测未来 AGRI → 冻结修正 R 反演未来 COT。只用复用 validation；CPP用于定义共同参考支持，不是独立COT真值。下表为按四个15分钟时效合并的像元加权RMSE，单位为无量纲 COT。

| 预报时段 | SimVP→R COT RMSE | 历史COT持平 RMSE | 离线真实未来AGRI→R RMSE |
|---|---:|---:|---:|
| 1 h | 10.255 | 8.946 | 8.169 |
| 2 h | 9.787 | 9.238 | 8.224 |
| 3 h | 10.194 | 9.617 | 9.052 |
| 4 h | 12.063 | 9.086 | 9.304 |

离线真实未来 AGRI→R 仅用于误差归因，不能作为可部署输入或forecast成绩。整体上，预测链在四个时段的共同CPP支持误差均高于历史持平，4小时差距最大。由于CPP为检索参考，此结果是验证诊断，不是独立真值精度结论。

## 湖北三站有无预测 COT 的配对 GHI 预报

输入为历史8帧AGRI → 修正几何SimVP外推未来AGRI → 冻结修正R生成预测COT → 15分钟GHI头。FA仅用AGRI和太阳几何；FAC在同样输入上加预测COT。两臂使用相同架构（24,489参数）、配对初始化、样本顺序、更新数和训练预算，三站分别训练seed 42/43/44。评价使用同一验证集样本，非独立test。

| 站点 | 不加预测COT RMSE | 加预测COT RMSE | 相对RMSE降幅 |
|---|---:|---:|---:|
| 牛首 | 88.062 | 87.570 | 0.56% |
| 王寨 | 96.219 | 96.867 | -0.67% |
| 米庄 | 102.828 | 102.390 | 0.43% |
| 等站点、等种子 | 95.703 | 95.609 | 0.098% |

总体点估计略有改善，但目标日期配对95%差值区间为 `[-2.271,+2.282] W/m²`；连续两日分组区间为 `[-2.007,+2.151] W/m²`，都跨0。王寨退化，因此目前不能说预测COT带来了稳定的GHI预报提升。PBS `215320.tc6000` 在 `zjnu-hpc` 的node22 GPU0（A800）正常退出，`exit_status=0`；test和buffer读取均为0。

## 接下来

先在完全相同的验证键上把预测AGRI→R、历史COT持平和离线真实未来AGRI→R的云场与站点GHI误差逐事件对齐，检查云团位置/到达时刻误差与COT厚度误差分别占多少。CPP参考和离线未来图仅作诊断。确认瓶颈后再设计单因素外推改动。

复现文件：`receipts/attempt12/summary_fp64.json`、`RESULTS.md`、`per_station_seed_arm.csv`、`per_lead.csv`、`forecast_cot_metrics.json`、`forecast_cot_metrics.csv`。几何收据：`receipts/attempt12/LIVE_TRAINING_20261021T0726CST.json` 和 `DATA_AUDIT_PASS_attempt12.json`。
