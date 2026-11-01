# 方法流程图设计卡：同刻价值与当前四臂预报分开

2026-10-30。第一版评审图；不含实验分数，不改正文或原始数据。

## 科学问题、位置与借鉴

该图回答：同刻COT表示价值如何检验，以及在预报初始化之后，历史COT和未来预测COT分别怎样进入实际四臂比较。放在数据节末/方法开头，在结构图前定义任务与输入身份。借鉴Chen Fig.1用数据流连接任务阶段，Fig.4将具体网络留给下一张结构图；借鉴Baguan-solar示例的阶段边界与已知条件侧路。不把真实未来云图塞进部署输入。

## 实际代码与形状

以下路径相对solar-cot-ifs。全部只读代码，无训练或test载荷访问。

| 来源 | 核验行及事实 | 图中职责 |
|---|---|---|
| refine-logs/hubei_solar_geometry_repair_20261021/code_pipeline_attempt12/infer_S.py:172–176,256–265,292–298 | 输入8×16×256×256，输出16×13×256×256；保存R所需16×16，不等于只预测16×16 | 8历史/16未来、区域预测与局地反演尺度 |
| refine-logs/hubei_forecast_cot_increment_20261022/code_attempt1/history_base.py:24–48 | 未来AGRI15×3×3、历史AGRI8×15×3×3、历史COT8×1×3×3，三分支；无COT参照使用相同结构，历史COT槽置零 | 两个基底，不是同期单帧头 |
| 同目录train.py:9–17,89–120 | 冻结历史COT基底；两个修正头gate/delta分别308→16→1，sigmoid(gate)×signed delta；delta末层零初始化；输出max(0,base+available×correction) | 共享历史基底上的等容量修正，不是旧attempt12独立头 |
| 同目录prepare.py:68–92 | 未来AGRI135+历史COT72+历史mask72+未来COT9+相对历史末帧差9+未来mask9+lead1+base1=308；仅未来COT/差值18槽在控制组置零 | 唯一改变及共同mask；缺测回退而非删样本 |
| experiments/monthly_R_station16_20261013_v1/train_ghi_heads.py:33–45,91–101 | 同期3×3输入；kt×区间晴空GHI后计算GHI损失 | 同期表示价值，与未来预测区分 |
| 根AGENTS.md §3.3、§5与IRRADIANCE_ERROR_PREVENTION.md | init为历史末帧，target为未来15分钟区间终点；ground GHI仅监督/评价 | 时间线与物理输出 |

## 设计

(a) 同期观测AGRI及可计算几何→16×16冻结R→对齐3×3 COT，与同刻3×3 AGRI送入同期头。CPP仅监督R，地面GHI仅监督/评价头，区间晴空GHI用于物理缩放。

(b) 初始化可用8帧→固定SimVP→16帧预测AGRI；预测AGRI也经R生成未来预测COT。真实历史AGRI经R得到8帧历史COT，与历史/预测AGRI共同构成历史COT基底。无COT参照采用相同网络且历史COT置零。两个修正头冻结同一历史COT基底，唯一区别是未来COT与相对历史末帧差值18槽为实际预测值或中性零；掩膜共享。图中以门控有符号残差概括实际代码，不虚构注意力。GHI训练分基底F与修正C，官方validation不用于本轮选epoch；不把重复开发验证当独立test。

(c) 8个历史快照与16个未来区间，目标+15…+240 min。区间平均GHI和卫星端点快照不同，流程图不暗示卫星图已经是15分钟平均云场。

S/R冻结指GHI阶段；不代表上游从未训练。两区域各自训练/评价，不画跨区域权重箭头。未来真实AGRI/CPP不进入部署路线。

## 独立审阅后的必要修订

初始草图引用旧attempt12独立预测COT头。独立审阅指出当前主预报图使用hubei_forecast_cot_increment_20261022四臂，未来COT组和同容量控制共同保留历史COT。因此已整体替换b面板、图注和来源，明确共同历史基底、18槽区别及缺测回退。没有重训或改数值。

## 输出

177mm宽、195mm高；8–9.5pt英文。青色为历史/同期反演COT，橙色为未来预测COT，灰色为AGRI/通用算子；颜色配合文字，不是唯一编码。保存SVG/PDF/PNG、FigureSpec与确定性源码、来源哈希。已实际查看PNG并修订；PDF另渲染检查。
