# attempt13辅助标签与修正太阳几何支持的只读核查

2026-10-21，在本核查读取实际数组前固定。

检查动机来自源码：prepare_attempt13_inputs.py将辅助CPP sidecar绑定到correct-UTC原R包，而冻结R训练实际使用prepared_attempt12/R_packs/full太阳几何修复包。二者目录不同不等于标签错误；必须比较实际行键、目标和有效掩膜，不能凭路径判定。

只读实际prepared_inputs、其sidecar、原correct-UTC R包、实际prepared_dir/R_packs/full（行表、norm、x_norm的3个几何通道、target、mask）。逐键比较源CPP值和有效掩膜；按原阈值cosSOZ>cos80且day_mask>0.5检查辅助支持是否落在修正R训练范围外，另列1e−6阈值边界像元，不移动阈值以改善结果。

sidecar按实际full.csv的原行序核对init/target时间与train/val；未来16帧必须为历史末帧+15…+240 min，data UTC+8与aux BJT一致。统计原行去重、CPP支持差异、target在共同/新增支持上的差异，以及sidecar中重复窗口加权的差异。

若完全一致，排除本项疑点；若不同，只记录差异及来源，区分产品有效支持和修正R的几何/QA支持，不能立即把差异归因成GHI效果。是否改变训练必须另存新方案、不能修改活动sidecar或原权重；215356/215359由其原控制目录执行，本窗口不停止或改写它们。

不训练、不读GHI或新观测、不打开test/buffer、无GPU操作、不改论文正文。核查输出写到本窗口独立目录，保留源SHA、逐split计数及代表性支持差异行，不从既有验证误差选案例。
