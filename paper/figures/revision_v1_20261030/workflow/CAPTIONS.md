# English caption

**Input sources and temporal availability for COT-assisted GHI mapping.** (a) Same-time analysis retrieves COT from an observed AGRI image on a 16×16 context and maps aligned 3×3 AGRI, geometry, and COT to station GHI. CPP provides retrieval supervision; ground GHI provides head supervision and evaluation. (b) The causal four-arm forecast experiment starts from eight observed frames and sixteen SimVP-predicted frames. Historical and forecast AGRI, together with historical retrieved COT, form a history-aware base. Its paired no-COT reference has the same architecture with historical-COT inputs set to zero. Two equal-capacity correction heads share the frozen historical-COT base. The forecast-COT head receives predicted COT and its local difference from the last historical COT; the matched control sets these same 18 slots to zero. Both retain identical availability masks and apply a gated signed residual; unavailable forecast COT returns the base output. Target geometry and clear-sky GHI are calculated from time and location. No observed future AGRI or CPP is an inference input. (c) Initialization is the last historical snapshot. Target \(\tau_k=t+15k\) min ends the GHI averaging interval \((\tau_k-15\,\mathrm{min},\tau_k]\), for \(k=1,\ldots,16\). Solid arrows carry available inputs; dashed arrows denote training supervision. Teal and orange distinguish historical retrieved and future predicted COT. This workflow specifies the implemented comparisons, not a claim of predictive improvement.

# 中文图注

**COT辅助GHI映射的输入来源与时间可用性。** (a) 同期分析在16×16邻域中从真实AGRI图像反演COT，再将对齐的3×3 AGRI、几何量和COT映射为站点GHI。CPP仅提供反演监督，地面GHI仅提供站点头监督和评价。(b) 因果四臂预报实验使用8帧真实历史图与16帧SimVP预测图。历史/预测AGRI及历史反演COT构成历史感知基底；配对无COT参照采用相同结构，仅将历史COT输入置零。两个等容量修正头共享同一个冻结的历史COT基底。未来COT头读取预测COT及其相对历史末帧COT的局地差值；同容量对照将相同的18个输入槽置零。两者保留相同有效性掩膜并施加门控有符号残差；未来COT不可用时回退到基底输出。目标太阳几何及晴空GHI由时间与位置计算，推理中不含真实未来AGRI或CPP。(c) 历史末帧定义初始化时刻；对于\(k=1,\ldots,16\)，目标时刻\(\tau_k=t+15k\)分钟为GHI平均区间\((\tau_k-15\mathrm{min},\tau_k]\)的结束时刻。实线表示可用输入，虚线表示训练监督；青与橙区分历史反演COT和未来预测COT。本图说明已实现的比较，不代表已证明预报改善。

## 评审说明

历史基底分支尺寸：未来15×3×3、历史8×15×3×3、历史COT8×1×3×3。不能套用单帧结构图的15+1输入。当前b面板绑定湖北hubei_forecast_cot_increment_20261022四臂，修正头输入308维；gate与delta各308→16→1，delta末层零初始化，输出max(0,base+available×sigmoid(gate)×delta)。它配套15,275行/22目标日期的四臂保存评分，不配套旧attempt12独立预测COT头。基底与修正使用不同训练日期子集，官方validation不用于本轮选epoch；仍属于复用开发验证。

## Alt text

Three-panel workflow. Same-time observed imagery feeds a COT retriever and local GHI head. Forecasting uses eight past images and sixteen predicted images. A historical-COT base is compared against a no-COT base; two equally sized residual heads then share the frozen historical-COT base and differ only in the future-COT input slots. A timeline shows eight historical snapshots and sixteen 15-minute forecast intervals.
