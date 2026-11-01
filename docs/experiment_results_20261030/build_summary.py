from pathlib import Path
import csv, hashlib, json, statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCES = {
 'latest_hunan_full': 'refine-logs/hunan_dydiff_20261028/full_validation_20261029/evaluation_attempt1/ghi_scores_fp64.json',
 'latest_hunan_audit': 'refine-logs/hunan_dydiff_20261028/full_validation_20261029/AUDIT_SIMVP_20261030.md',
 'latest_hunan_comparison': 'refine-logs/hunan_dydiff_20261028/full_validation_20261029/HISTORICAL_COMPARISON_20261030.md',
 'hunan_retrieval': 'refine-logs/cot_value_first_20261013/results/cot_evaluation_yi_sun_v1/retrieval_validation_metrics.json',
 'hunan_same_time': 'audits/sili_invalid_day_20261019/RESULTS.json',
 'hunan_cpp_audit': 'audits/hunan_cpp_time_alignment_20261020/REPORT.md',
 'hubei_cpp_audit': 'audits/hubei_cpp_time_impact_20261020/REPORT.md',
 'hubei_same_time': 'refine-logs/hubei_cloud_probability_20261020/results_fourarm_attempt2/metrics_by_station_arm.csv',
 'hubei_solar_scope': 'refine-logs/hubei_solar_geometry_repair_20261021/EXPERIMENT_PLAN.md',
 'hubei_future_ghi': 'refine-logs/hubei_solar_geometry_repair_20261021/receipts/attempt12/summary_fp64.json',
 'hubei_future_cot': 'refine-logs/hubei_solar_geometry_repair_20261021/receipts/attempt12/forecast_cot_metrics.json',
 'hubei_four_arm': 'refine-logs/hubei_forecast_cot_increment_20261022/evidence_attempt1/scored_attempt1/score.json',
 'hubei_four_arm_cells': 'refine-logs/hubei_forecast_cot_increment_20261022/evidence_attempt1/scored_attempt1/arm_metrics.csv',
 'hubei_history_v1': 'refine-logs/hubei_solar_geometry_repair_20261021/history_cot_forecast_ghi_20261022/results_attempt2/scored_attempt2/global_summary.json',
 'summer_confirmation': 'refine-logs/hubei_history_cot_confirmation_20261022/scored_attempt3/SCORES.json',
 'summer_scope_diagnosis': 'refine-logs/hubei_history_cot_confirmation_20261022/diagnosis_20261023/REPORT.md',
 'summer_fallback': 'refine-logs/hubei_history_cot_missing_fallback_20261023/RESULTS.json',
 'direct_cot': 'refine-logs/agri_direct_cot_20261020/results/direct_cot_cpp_validation_score_v1_local4060/cot_cpp_reference_metrics.json',
 'direct_ghi': 'refine-logs/agri_direct_cot_20261020/results/direct_cot_ghi_validation_score_v1_local4060/direct_cot_ghi_metrics.json',
 'direct_b2_b3': 'refine-logs/agri_direct_cot_20261020/B2_B3_VALIDATION_FINDINGS_20261021.md',
 'conditional_probe': 'refine-logs/hubei_cot_conditional_information_20261023/RESULTS.json',
}
def load(k): return json.loads((ROOT/SOURCES[k]).read_text(encoding='utf-8'))
rows=[]
def row(experiment, task, region, station, method, metric, value, unit, status, source):
 rows.append(dict(experiment=experiment, task=task, region=region, station=station, method=method, metric=metric, value=value, unit=unit, status=status, source=SOURCES[source]))

latest=load('latest_hunan_full')
for method in ['simvp','dydiff']:
 rec=latest['comparisons'][method+'_add_COT']
 for arm,field in [('no_COT','before_macro_rmse_Wm2'),('forecast_COT','after_macro_rmse_Wm2')]:
  row('Hunan_full_frozen_head_20261029','forecast_GHI','Hunan','equal_station_mean',method+'_'+arm,'rmse',rec[field],'W/m2','positive_SimVP_frozen_validation_not_independent_test','latest_hunan_full')
 for st,entry in rec['by_station'].items():
  for arm,field in [('no_COT','before_rmse'),('forecast_COT','after_rmse')]:
   row('Hunan_full_frozen_head_20261029','forecast_GHI','Hunan',st,method+'_'+arm,'rmse',entry[field],'W/m2','frozen_validation_not_both_stations_significant','latest_hunan_full')

r=load('hunan_retrieval')
for scale,m in r['metrics'].items():
 if not isinstance(m,dict) or 'by_station' not in m: continue
 for station,v in m['by_station'].items():
  for metric in ['rmse','mae','bias']:
   row('Hunan_monthly_R_20261013','same_time_COT','Hunan',station,scale,metric,v[metric],'COT','archived_validation_reference_agreement','hunan_retrieval')
 for metric in ['rmse','mae','bias']:
  row('Hunan_monthly_R_20261013','same_time_COT','Hunan','equal_station_mean',scale,metric,statistics.mean(v[metric] for v in m['by_station'].values()),'COT','archived_validation_reference_agreement','hunan_retrieval')

j=load('hunan_same_time')
for exp in ['single_frame_A_AC_AP_AL','current_three_frame_geometry']:
 x=j['experiments'][exp]['after']
 for arm,value in x['station_seed_equal_RMSE'].items(): row(exp,'same_time_GHI','Hunan','equal_station_mean',arm,'rmse',value,'W/m2','dayclean_fixed_weight_rescore','hunan_same_time')
 for station,v in x['by_station'].items():
  for arm,value in v['rmse'].items(): row(exp,'same_time_GHI','Hunan',station,arm,'rmse',value,'W/m2','dayclean_fixed_weight_rescore','hunan_same_time')

with (ROOT/SOURCES['hubei_same_time']).open(encoding='utf-8-sig') as f:
 for x in csv.DictReader(f):
  row('Hubei_correct_UTC_pre_solar_repair','same_time_GHI','Hubei',x['station'],x['arm'],'rmse',float(x['rmse_mean_seeds']),'W/m2','historical_correct_UTC_but_pre_solar_repair','hubei_same_time')

four=load('hubei_four_arm')
for arm,value in four['equal_station_equal_seed_rmse'].items(): row('Hubei_four_arm_forecast','forecast_GHI','Hubei','equal_station_mean',arm,'rmse',value,'W/m2','development_validation','hubei_four_arm')
with (ROOT/SOURCES['hubei_four_arm_cells']).open(encoding='utf-8-sig') as f: cells=list(csv.DictReader(f))
for arm in ['G0','G1','G2','G3']:
 per_seed=[statistics.mean(float(x['rmse']) for x in cells if x['arm']==arm and int(x['seed'])==seed) for seed in [42,43,44]]
 row('Hubei_four_arm_forecast','forecast_GHI','Hubei','equal_station_mean',arm,'seed_sd',statistics.stdev(per_seed),'W/m2','descriptive_SD_not_sampling_uncertainty','hubei_four_arm_cells')

d=load('direct_cot')
for arm,m in d['overall']['station_3x3'].items():
 for metric in ['rmse','mae','bias','thick_pod','thick_far','thick_csi','thick_bias']:
  row('Hunan_direct_COT','forecast_COT','Hunan','pooled_common_3x3_pixels',arm,metric,m[metric],'COT' if metric in ['rmse','mae','bias','thick_bias'] else 'fraction','CPP_reference_limited_coverage','direct_cot')

d=load('direct_ghi')
for station,arms in d['point_metrics'].items():
 for arm,seeds in arms.items():
  values=[v['all']['rmse_wm2'] for v in seeds.values()]
  row('Hunan_direct_COT_GHI','forecast_GHI','Hunan',station,arm,'rmse',statistics.mean(values),'W/m2','development_validation','direct_ghi')
for arm in d['point_metrics']['sili']:
 values=[v['all']['rmse_wm2'] for arms in d['point_metrics'].values() for v in arms[arm].values()]
 row('Hunan_direct_COT_GHI','forecast_GHI','Hunan','equal_station_mean',arm,'rmse',statistics.mean(values),'W/m2','development_validation','direct_ghi')

summer=load('summer_confirmation')['scopes']['primary_shared_dates']
for method,field in [('no_COT','no_cot_rmse'),('history_COT','history_cot_rmse')]:row('Hubei_new_summer','forecast_GHI','Hubei','equal_station_mean',method,'rmse',summer[field],'W/m2','new_dates_with_issue_hour_domain_expansion','summer_confirmation')
fallback=load('summer_fallback')['summary']['primary']
for method,value in fallback['rmse'].items():row('Hubei_new_summer_fallback','forecast_GHI','Hubei','equal_station_mean',method,'rmse',value,'W/m2','posthoc_engineering_diagnostic','summer_fallback')

with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
index={k:{'path':v,'sha256':hashlib.sha256((ROOT/v).read_bytes()).hexdigest(),'size_bytes':(ROOT/v).stat().st_size} for k,v in SOURCES.items()}
(OUT/'SOURCE_INDEX.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'SUMMARY_DATA.json').write_text(json.dumps({'scope':'Repository saved train/validation scores and existing independent audits; no new inference or training; no test/buffer payload opened','latest_hunan_full_validation':latest,'hunan_same_time':j['experiments']['single_frame_A_AC_AP_AL']['after'],'hubei_four_arm':four,'summer_primary':summer,'summer_fallback':fallback,'metric_rows':len(rows)},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'metrics_rows':len(rows),'bound_sources':len(index),'four_arm_seed_sd':{a:next(x['value'] for x in rows if x['experiment']=='Hubei_four_arm_forecast' and x['method']==a and x['metric']=='seed_sd') for a in ['G0','G1','G2','G3']}},ensure_ascii=False))
