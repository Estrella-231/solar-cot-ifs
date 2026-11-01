# Yi × Sun COT验证集复算结果

日期：2026-10-13（BJT）  
范围：新月度划分 validation；真实目标AGRI经冻结R反演的COT，与同扫描FY-4B CPP COT参考比较。此为同期反演一致性，不是未来COT预报结果，也不是独立地面真值评价。

## 数据与复算

- 1,398个站点样本：四里700、竺家698；覆盖57个BJT目标日期（2024-04-18至2026-03-02）。
- 16×16站点patch共有356,169/357,888个共同有效像元，覆盖率99.52%；test载荷读取数为0。
- 对预测log1p(COT)仅做一次FP64 expm1逆变换；按站点中心像元、中心3×3逐像元、原生16×16 patch分别评分。16×16是站点patch，不代表湖南区域全场。
- 主表为两站等权平均各站RMSE/MAE/bias；pooled像元统计仅作补充。置信区间按BJT目标日期成簇配对重采样2,000次。

## 指标

| 空间支持 | 两站等权RMSE | 两站等权MAE | 两站等权bias | pooled RMSE | 共同支持 |
|---|---:|---:|---:|---:|---:|
| 站点中心像元 | 8.482 | 4.269 | +0.306 | 8.486 | 1,390像元样本 |
| 中心3×3 | 8.522 | 4.310 | +0.279 | 8.526 | 12,510像元样本 |
| 站点16×16 patch | 9.053 | 4.647 | +0.279 | 9.074 | 356,169像元 |

单位为COT（无量纲）。patch RMSE的日期簇bootstrap 95%区间为6.563–11.829；中心像元为6.011–11.144，中心3×3为6.047–11.191。当前验证预测NPZ已有的pooled patch RMSE为9.074，与独立FP64复算一致。

厚云层显示明显低估：以CPP参考COT≥30分层，四里bias为−7.15、RMSE为13.99（26,211像元）；竺家bias为−10.29、RMSE为14.28（24,713像元）。该分层应作为限制报告，不能只用总体误差概括模型。

## 尚不能下的结论与下一步

1. 这批指标只衡量真实目标AGRI→R的同期反演与CPP参考的一致性；不能支持S→R未来预报优于持续性或输运基线的说法。
2. SSIM暂不计算：当前validation文件没有携带仅由train参考确定的log1p-COT q99.9尺度；中心3×3也小于预设的11×11 SSIM窗口。不得用validation/test范围补定尺度。
3. 远端只找到真实AGRI COT bank，没有发现S→R未来16lead预测bank。因此下一项为核验新月度S模型来源、冻结的预测输入/输出键及val推理可用性；只有来源合同通过后，才生成或复用val预测并计算16lead S→R、Persistence-R及可用输运基线。test继续关闭。

## 可复核产物

- 独立评分程序：evaluate_cot_retrieval_validation_yi_sun_v1.py
- 机器可读指标：results/cot_evaluation_yi_sun_v1/retrieval_validation_metrics.json
- 输入预测：validation_predictions_monthly_R_seed42.npz
- 参考及配对审计：clp_reference_monthly_R_seed42.npz、clp_reference_monthly_R_seed42.audit.json
