from pathlib import Path
import csv, hashlib, json
import numpy as np

ROOT=Path(__file__).resolve().parent
BANK=ROOT.parent/'forecast_cot_val_bank_20261014_v1'/'forecast_cot_val_bank.npz'
OUT_JSON=ROOT/'arrival_cohort_diagnostic.json'
OUT_CSV=ROOT/'arrival_cohort_by_window.csv'
OUT_MD=ROOT/'COT_FORECAST_ARRIVAL_COHORT_DIAGNOSTIC_20261014.md'
THRESHOLD=30.0
# The bank contains validation predictions only. All physical metrics below are
# recomputed in float64, with no CPP filling and no test rows.
sha=hashlib.sha256(BANK.read_bytes()).hexdigest()
with np.load(BANK,allow_pickle=False) as z:
    pred=np.expm1(z['forecast_log1p_cot'].astype(np.float64))
    persistence=np.expm1(z['persistence_log1p_cot'].astype(np.float64))
    offline=np.expm1(z['target_agri_r_log1p_cot'].astype(np.float64))
    ref=z['cpp_reference_cot'].astype(np.float64)
    mask=z['cpp_valid_mask'].astype(bool)
    ids=z['sequence_ids'].astype(str)
    times=z['target_time_utc'].astype(str)
rows=[]
for wi,sid in enumerate(ids):
    # Require at least one CPP-common pixel at every lead for both stations;
    # this is exactly the pre-existing complete-support cohort, not an event filter.
    if not np.all(mask[wi,:,:,7:10,7:10].sum(axis=(-1,-2))>0):
        continue
    for si,station in enumerate(('Sili','Zhujia')):
        series={'CPP reference':ref[wi,si], 'S→R forecast':pred[wi,si],
                'Persistence-R':persistence[wi,si], 'Future observed AGRI→R (offline)':offline[wi,si]}
        means={}
        supports=[]
        for method,arr in series.items():
            vals=[]; counts=[]
            for li in range(16):
                m=mask[wi,si,li,7:10,7:10]
                counts.append(int(m.sum()))
                vals.append(float(arr[li,7:10,7:10][m].mean()))
            means[method]=np.asarray(vals,dtype=np.float64)
            if method=='CPP reference': supports=counts
        leads=np.arange(15,241,15,dtype=int)
        refv=means['CPP reference']
        ref_peak_i=int(np.argmax(refv))
        ref_hit=np.flatnonzero(refv>=THRESHOLD)
        for method,v in means.items():
            peak_i=int(np.argmax(v)); hit=np.flatnonzero(v>=THRESHOLD)
            rows.append({
                'sequence_id':sid,'station':station,'target_date_bjt':sid.split('_')[1][:8],
                'series':method,'cpp_event_present':bool(len(ref_hit)),
                'cpp_first_hit_lead_min':int(leads[ref_hit[0]]) if len(ref_hit) else None,
                'method_first_hit_lead_min':int(leads[hit[0]]) if len(hit) else None,
                'method_detected_within_240min':bool(len(hit)),
                'arrival_error_min':int(leads[hit[0]]-leads[ref_hit[0]]) if len(hit) and len(ref_hit) else None,
                'reference_peak_cot':float(refv[ref_peak_i]),'reference_peak_lead_min':int(leads[ref_peak_i]),
                'method_peak_cot':float(v[peak_i]),'method_peak_lead_min':int(leads[peak_i]),
                'peak_amplitude_error_cot':float(v[peak_i]-refv[ref_peak_i]),
                'min_common_pixels_per_lead':int(min(supports)),
                'mean_common_pixels_per_lead':float(np.mean(supports)),
                'target_time_utc':times[wi].tolist()
            })
methods=('CPP reference','S→R forecast','Persistence-R','Future observed AGRI→R (offline)')
summary=[]
for station in ('Sili','Zhujia'):
    base=[r for r in rows if r['station']==station and r['series']=='CPP reference']
    events=sum(r['cpp_event_present'] for r in base)
    for method in methods[1:]:
        rr=[r for r in rows if r['station']==station and r['series']==method]
        event_rows=[r for r in rr if r['cpp_event_present']]
        arrivals=[r['arrival_error_min'] for r in event_rows if r['arrival_error_min'] is not None]
        amp=[r['peak_amplitude_error_cot'] for r in event_rows]
        summary.append({'station':station,'series':method,'complete_windows':len(base),
            'reference_thick_events':events,'detected_events':sum(r['method_detected_within_240min'] for r in event_rows),
            'missed_within_240min':sum(not r['method_detected_within_240min'] for r in event_rows),
            'arrival_error_n_detected':len(arrivals),
            'mean_arrival_error_min_detected_only':float(np.mean(arrivals)) if arrivals else None,
            'median_arrival_error_min_detected_only':float(np.median(arrivals)) if arrivals else None,
            'mean_peak_amplitude_error_cot':float(np.mean(amp)) if amp else None,
            'overlapping_windows_are_independent_events':False})
result={'state':'COMPLETE_VALIDATION_COMPLETE_SUPPORT_ARRIVAL_DIAGNOSTIC','split':'validation_only',
    'test_payloads_read':0,'bank_path':str(BANK),'bank_sha256':sha,
    'bank_windows':int(len(ids)),'complete_support_windows':int(len({r['sequence_id'] for r in rows})),
    'unique_target_dates_bjt':sorted({r['target_date_bjt'] for r in rows}),
    'threshold_cot':THRESHOLD,'scale':'physical COT; fixed 3x3 station neighborhood; per-lead mean over CPP-common valid pixels',
    'arrival_rule':'first +15..+240 minute lead whose 3x3 common-support mean reaches COT>=30; discrete 15-minute lead, no interpolation',
    'methods':['S→R forecast','Persistence-R','Future observed AGRI→R (offline)'],
    'window_level_rows':len(rows),'window_level_results':summary,
    'limitations':['CPP is a retrieval reference, not independent truth.',
      'Only CPP-supported complete windows are evaluated; this is a reference-supported diagnostic, not deployable coverage.',
      'Windows overlap in time and are not independent cloud events; counts and averages are descriptive, no CI is reported.',
      'Arrival error averages only cases where both reference and method cross threshold; misses are reported separately.',
      'Offline future observed AGRI→R is not a deployable forecast.',
      'A 3x3 mean threshold is a local signal proxy, not cloud-object tracking.']}
OUT_JSON.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
with OUT_CSV.open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
lines=['# COT外推到达行为：完整共同支持验证窗口','',
 f"bank窗口数 {len(ids)}；同时覆盖两站全部16 lead且每站每lead至少有1个CPP共同有效3×3像元的窗口数 {result['complete_support_windows']}；目标日期 {', '.join(result['unique_target_dates_bjt'])}。",
 '', '到达定义为每个窗口/站点中，固定3×3共同有效像元的物理COT均值首次达到COT≥30的离散lead。未插值；未达到记为240分钟内miss。仅在参考与方法都达到阈值时计算到达误差，miss另列。', '',
 '| 站点 | 方法 | 参考厚云窗口 | 方法命中 | 240 min miss | 命中样本到达误差均值/中位数(min) | 峰值幅度误差均值(COT) |',
 '|---|---|---:|---:|---:|---:|---:|']
for x in summary:
    avg='—' if x['mean_arrival_error_min_detected_only'] is None else f"{x['mean_arrival_error_min_detected_only']:+.1f} / {x['median_arrival_error_min_detected_only']:+.1f}"
    amp='—' if x['mean_peak_amplitude_error_cot'] is None else f"{x['mean_peak_amplitude_error_cot']:+.2f}"
    lines.append(f"| {x['station']} | {x['series']} | {x['reference_thick_events']} | {x['detected_events']} | {x['missed_within_240min']} | {avg} | {amp} |")
lines += ['', '## 限制与解释','',
 '这些是CPP有支持的重叠窗口数，不是独立云事件数；不计算显著性区间。检测数只针对参考厚云窗口。方法未越过阈值的情况按miss记录，不能只在成功命中子集报告平均到达误差。离线真实未来AGRI→R用于辨别反演链和外推链差异，不能作为业务预报成绩。CPP不是独立真值。',
 '', '本诊断把单例曲线扩展到完整共同支持窗口，但时间/空间支持仍窄。结合主指标（SimVP→R整体RMSE点估计略好于持续性、区间跨0；厚云CSI与SSIM较差），证据仍不支持外推头稳定有skill；不据此启动GHI forecast head训练。机器结果与逐窗明细分别见`arrival_cohort_diagnostic.json`及`arrival_cohort_by_window.csv`。test载荷读取为0。']
OUT_MD.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'state':result['state'],'n_windows':result['complete_support_windows'],'dates':result['unique_target_dates_bjt'],'summary':summary},ensure_ascii=False))
