# 太阳几何均时差单位错误：已复现及本轮修正范围

2026-10-15独立核查发现：湖南AGRI预处理producer的`compute_solar_angles()`计算均时差时遗漏弧度转角度。

```python
# 三角展开expression的单位是弧度
equation_of_time = 4.0 * expression                 # 原实现，错误
equation_of_time = 4.0 * np.rad2deg(expression)      # 本轮独立修正模块
```

取回的原producer文件SHA256为`74b6bf83f40ebe1a05e1a8ff35963e4eebf0d715e3d7f8a9519125206fd72f22`。它在171个真实train/val窗口、8,455个像元上复现原缓存，最大差0.000061°。修正后对独立NREL SPA名义UTC参考的最大天顶角/方位角误差为0.0101°/0.0215°，原值为3.0897°/5.8915°。不是仅凭与SPA有差异而推测代码错误。

原始修正缺失与“实际行扫描晚于名义UTC约3分钟”必须分开。源SOA在物理夜间置零也是独立合同，不能将夜间余弦差全部归于均时差错误。

本轮只在新命名的连续历史GHI配对头中使用修正几何，原数据、SimVP、R、旧COT bank和旧结果只读保留。冻结R仍来自原几何输入，不能称全链已经修复，不能直接用新几何替换R输入而冒充同一来源。若后续重建资产，须独立命名、保存来源，并重新训练/评价；是否重建按当前实验方案决定，本条不自动要求覆盖或重跑所有历史实验。

本轮两站两组共同使用修正几何，因此配对只检验连续COT增量；不能从新旧模型分数差中单独声称几何修复收益或历史表示收益。

- [完整核查及连续历史配对结果](../results/cot_interval_history_20261015/REPORT.md)
- [修正实现](../refine-logs/cot_interval_history_20261015/solar_geometry_corrected.py)
- [独立SPA与源公式复现](../results/cot_interval_history_20261015/audit/geometry_audit.json)
- [实验合同](../refine-logs/cot_interval_history_20261015/PLAN.md)

当前新test载荷未打开，论文正文未修改。
