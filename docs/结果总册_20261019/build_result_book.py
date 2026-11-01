"""Build an evidence-linked Chinese LaTeX results atlas; no scientific model is run."""
from pathlib import Path
import re, json, hashlib, shutil, csv, datetime, subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
SOURCES=HERE/'sources'; FIGURES=HERE/'figures'; CODE=HERE/'code_snapshot'
for d in (SOURCES,FIGURES,CODE): d.mkdir(exist_ok=True)
groups=[]
def add(title,report,meaning,scope='验证集；同期GHI估计',figs=(),roots=(),status='已完成'):
    p=ROOT/report
    if not p.exists(): raise FileNotFoundError(p)
    groups.append(dict(id=f'E{len(groups)+1:02}',title=title,report=report,meaning=meaning,
                       scope=scope,figs=list(figs),roots=list(roots),status=status))

add('湖南同期估计：COT能否改善紧凑GHI模型？',
 'experiments/monthly_R_station16_20261013_v1/results/GHI_HEAD_THREE_SEED_RESULTS_20261013.md',
 '三种子两站等权RMSE约降低3.02%；评价的是目标时刻真实AGRI辅助的同期估计。',
 figs=['paper/figures/fig4_cot_incremental_value.png'],
 roots=['experiments/monthly_R_station16_20261013_v1'])
add('湖北独立区域：同期COT收益能否重复？',
 'refine-logs/hubei_cot_independent_20261014/SAME_TIME_THREE_SEED_VALIDATION_20261017.md',
 '采用最新归一化重载及2026-10-18复算口径：三站等权RMSE降低约4.49%，与湖南分开报告。',
 scope='湖北三站；22个验证目标日期；三种子',figs=['refine-logs/hubei_cot_independent_20261014/validation_cases_normreload_20261017/*.png'],roots=['refine-logs/hubei_cot_independent_20261014'])
add('反演质量：COT对CPP参考的误差有多大？',
 'paper/analyses/cot_retrieval_quality_20261017/REPORT.md',
 '先量化输入COT的参考一致性及厚云偏差，再解释下游价值；CPP不是独立物理真值。',
 scope='湖南与湖北各自验证；空间与样本范围不同',figs=['paper/figures/fig3_retrieval_quality.png'])
add('厚度分层：GHI收益主要出现在哪些云厚？',
 'refine-logs/cot_value_first_20261013/results/cot_thickness_story_20261014/COT_THICKNESS_GHI_VALIDATION_20261014.md',
 '湖南CPP参考厚云子组RMSE降幅22.45%；其余组较小且区间跨零。分层收益不代表总体收益。',
 scope='湖南同期；中心像元CPP参考分箱；不是预测云况',figs=['paper/figures/fig5_hunan_cpp_cot_strata.png'])
add('太阳资源：高太阳高度与日辐照量是否改善？',
 'paper/analyses/solar_resource_validation_20261017/REPORT.md',
 '高太阳高度和完整日照日期的误差用于检查实际资源估计意义；MAE降低与日偏差更负可以并存。',
 scope='湖南验证；高太阳高度及完整日期敏感性')
add('容量与反演特征：收益是否超过普通辅助表示？',
 'refine-logs/cot_value_first_20261013/results/T18_RAL_RESULTS_20261014.md',
 '区分COT辅助表示、额外头部容量和反演器隐藏特征；不能把全部收益唯一归于显式物理值。')
add('湖北同R特征对照：显式COT是否优于隐藏特征？',
 'refine-logs/hubei_cot_independent_20261014/audit_receipts/al_normreload_20261017/AL_RESULT_AUDIT_20261017.md',
 '采用重载后的隐藏特征对照；COT与隐藏特征差异需按共同样本和区间判断，不能引用被取代的旧汇总。',scope='湖北同期；最新归一化重载')
add('湖北三站共同时间：站点缺测会改变结论吗？',
 'refine-logs/hubei_cot_independent_20261014/audit_receipts/al_normreload_20261017/THREE_STATION_COMMON_TIME_SENSITIVITY_20261017.md',
 '统一三站目标时刻后检验表示差异，避免不同站点样本构成造成误解。',scope='湖北同期；三站486个共同目标时刻')
add('湖北条件差异：月份和太阳高度有何影响？',
 'refine-logs/hubei_cot_independent_20261014/HUBEI_VALIDATION_SUBGROUPS_NORMRELOAD_20261017.md',
 '展示验证增益的条件性；月份和太阳角分层不能替代独立晴空、阴天分类。',scope='湖北最新重载验证；描述性条件分层')
add('更强AGRI残差：COT仍有额外收益吗？',
 'experiments/monthly_R_station16_20261013_v1/crossfit_signed_residual/UNIFIED_FEATURE_RESIDUAL_RESULTS_20261014.md',
 '统一输入残差模型中，AGRI残差RMSE111.277、AGRI+COT113.419，差异区间跨零；COT收益依赖比较模型。',
 figs=['docs/组会原始材料_20261016/10_残差修正/残差修正效果.png'],roots=['experiments/monthly_R_station16_20261013_v1/crossfit_signed_residual'])
add('有符号残差：早期修正结果与后续统一对照',
 'experiments/monthly_R_station16_20261013_v1/crossfit_signed_residual/OOF_SIGNED_RESIDUAL_RESULTS_20261014.md',
 '保留早期残差探索作为演进材料；最终增量判断以统一输入残差结果为准。',status='历史探索；已被统一输入对照补充')
add('三帧COT：历史变化是否真的被利用？',
 'results/cot_interval_history_20261015/REPORT.md',
 '三帧3×3输入确实影响GHI，但总体RMSE点改善仅0.24%，日期区间跨零。',
 scope='湖南三帧离线区间反演；4,263条验证',
 figs=['results/cot_interval_history_20261015/figures/*.png'],roots=['refine-logs/cot_interval_history_20261015','results/cot_interval_history_20261015'])
add('R太阳几何修正：COT误差是否降低？',
 'experiments/r_solar_geometry_corrected_20261015/RESULT_REPORT.md',
 '修正太阳几何后单独评价R；反演指标改善不自动代表GHI改善。',scope='湖南R验证；几何单因素',roots=['experiments/r_solar_geometry_corrected_20261015'])
add('修正后的R进入GHI：上游改善能否传递？',
 'experiments/r_geometry_downstream_ghi_20261016/REPORT.md',
 '在GHI端检查几何修正的实际作用，避免只凭R误差判断下游收益。',roots=['experiments/r_geometry_downstream_ghi_20261016'])
add('高GHI低估：增大高透过损失的收益与代价',
 'experiments/high_kt_loss_20261016/REPORT.md',
 '高GHI低估减轻，但低、中kt误差增加；总体未达到替换要求，高值改善不能归为COT贡献。',
 figs=['experiments/high_kt_loss_20261016/figures/*.png'],roots=['experiments/high_kt_loss_20261016'])
add('真实云事件：COT为什么有时把GHI调错方向？',
 'experiments/cot_underestimate_forensics_20261016/REPORT.md',
 '并列改善与失败案例，检查真实AGRI、COT和GHI；案例用于解释现象，不能估计总体收益。',
 scope='湖南同期；固定失败和匹配案例诊断',
 figs=['paper/figures/fig6_sili_high_change_event.png','experiments/cot_underestimate_forensics_20261016/figures/*.png'],roots=['experiments/cot_underestimate_forensics_20261016'])
add('云高与云影：空间代表性是否有可用证据？',
 'experiments/monthly_R_station16_20261013/cot_shadow_projection_20261014/REPORT.md',
 '利用可追溯CPP云高作视差与云影离线诊断；可视化不等于已证明生产模型完成校正。',
 scope='湖南少量真实案例；离线物理诊断',figs=['experiments/monthly_R_station16_20261013/cot_shadow_projection_20261014/CTH_PARALLAX_SHADOW_VALIDATION_20261014.png'])
add('输入扰动：GHI模型使用了哪些COT信息？',
 'experiments/monthly_R_station16_20261013_v1/results/COT_INPUT_DEPENDENCE_VALIDATION_20261013.md',
 '固定模型输入替换揭示依赖性；替换后的分布变化不能等同公平重训后的COT净收益。',scope='湖南同期；固定模型诊断')
add('晴空屏蔽oracle：COT变准是否让GHI更准？',
 'refine-logs/cot_mask_necessity_20261014/M0_M1_RESULTS.md',
 '参考掩膜离线清零可改变COT误差，却未必改善固定GHI头；oracle不是可部署成绩。',scope='湖南train/val；不可部署oracle诊断',roots=['refine-logs/cot_mask_necessity_20261014'])
add('可部署保守判云：重训GHI头后是否受益？',
 'results/paired_gate_ghi_seed42_20261015_attempt1/REPORT.md',
 '门控COT与原COT各自训练匹配头；seed42未显示净收益，停止无条件扩种子。',scope='湖南验证；seed42门控配对',roots=['results/paired_gate_ghi_seed42_20261015_attempt1','refine-logs/cot_cloud_gate_retrain_20261014'])
add('门控退化归因：误清云能解释多少？',
 'results/gate_ghi_forensics_20261015/REPORT.md',
 '检查被清零与未改变的局部COT记录；误差变化不能全部归于误清云。',scope='湖南固定模型事后诊断',figs=['results/gate_ghi_forensics_20261015/*.png'])
add('站点GHI输入范围：小邻域是否足够？',
 'docs/FORECAST_GHI_WINDOW_ABLATION_20261004.md',
 '空间范围消融解释选择小邻域的依据；旧划分实验不能当成新月度主成绩。',scope='旧划分预报GHI尺寸消融；历史对照',status='历史对照',roots=['experiments/forecast_ghi_window_20261004_v1'])
add('最新COT预报：固定SimVP与等预算监督对照',
 'refine-logs/cot_retrieved_forecast_20261013/FULLTRAIN_RESULTS_20261017.md',
 '3,140验证序列、46个CPP日期的同掩膜COT评价；直接预测COT候选没有超过固定SimVP→R。',
 scope='湖南未来+15～+240分钟；单种子筛选',figs=['paper/figures/fig7_forecast_cot_supervision.png'],roots=['refine-logs/cot_retrieved_forecast_20261013'])
add('直接COT幅值损失：厚云误差能否修复？',
 'refine-logs/cot_retrieved_forecast_20261013/results/SAME_SUPPORT_AMPLITUDE_DIAGNOSIS_20261017.md',
 '相同CPP像元上检查幅值损失；局部厚云误差小改动没有转化为总体COT优势。',scope='湖南未来COT；同掩膜幅值单因素')
add('独立厚云事件：位置、峰值和到达时间',
 'refine-logs/cot_retrieved_forecast_20261013/results/event_cohort_decomposition_20261017/EVENT_COHORT_DECOMPOSITION_20261017.md',
 '七个厚云事件集中在三个日期；用事件显示云团检出与峰值问题，不能作独立总体推断。',scope='湖南验证；七事件描述性分解',figs=['refine-logs/cot_retrieved_forecast_20261013/results/event_cohort_decomposition_20261017/*.png'])
add('离线位置对齐：错位还是厚度演变？',
 'refine-logs/cot_retrieved_forecast_20261013/results/position_amplitude_events_20261017/EVENT_POSITION_AMPLITUDE_DIAGNOSTIC_20261017.md',
 '使用未来参考作位置对齐仅用于归因；对齐后误差不作为预报成绩。',scope='湖南未来COT；不可部署离线诊断',figs=['refine-logs/cot_retrieved_forecast_20261013/results/position_amplitude_events_20261017/*.png'])
add('时间变化：预测COT轨迹是否过平？',
 'refine-logs/cot_value_first_20261013/results/cot_temporal_change_audit_20261017/RESULTS.md',
 '与真实未来AGRI经同R得到的代理轨迹比较，预测变化相关但幅度衰减；代理不是物理真值。',scope='湖南52个目标日期；R到R代理对照',figs=['refine-logs/cot_value_first_20261013/results/cot_temporal_change_audit_20261017/*.png'])
add('固定GHI头2×2替换：图像与COT误差如何传递？',
 'refine-logs/cot_value_first_20261013/results/fixed_head_2x2_allval_agri_cot_attribution_20261014_v1/ALLVAL_FIXED_HEAD_ATTRIBUTION_20261014.md',
 '分别替换预测AGRI与预测COT，定量诊断观测输入头面对预测输入的误差；不是预测输入重训成绩。',scope='湖南2,817序列、52日期；离线固定头归因',figs=['refine-logs/cot_value_first_20261013/results/fixed_head_2x2_allval_agri_cot_attribution_20261014_v1/*.png'])
add('历史COT输运：与Cloud-FNO文献如何比较？',
 'results/history_cot_cloud_fno_20261015/REPORT.md',
 '保留站点像元的1～4小时误差；区域、分辨率及参考不同，只作跨研究描述，不排名。',scope='旧验证集历史输运；文献描述性参照',figs=['results/history_cot_cloud_fno_20261015/hourly_comparison.png'],roots=['results/history_cot_cloud_fno_20261015'])
add('历史COT预报GHI：修正选模流程后的配对结果',
 'refine-logs/history_cot_forecast_ghi_20261015/attempt3_results.md',
 '使用日期留出选模后比较AGRI、历史COT输运及混合输入；该结果不能与同期3%混合。',scope='湖南旧COT输运；attempt3日期留出',figs=['refine-logs/history_cot_forecast_ghi_20261015/forecast_ghi*attempt3*zh.png'],roots=['refine-logs/history_cot_forecast_ghi_20261015'])
add('湖北未来GHI：预测COT是否带来净增益？',
 'refine-logs/hubei_cot_independent_20261014/FORECAST_GHI_NORMRELOAD_THREE_SEED_VALIDATION_20261017.md',
 '预测AGRI基线77.150、加入预测COT78.155 W/m²；差异区间跨零，未显示预报净增益。',scope='湖北三站；22验证日期；最新归一化重载',figs=['refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_three_seed_normreload_20261017/figures/*.png'])
add('湖南最新未来GHI：完整预测COT与收缩输入',
 'refine-logs/cot_aware_simvp_20261017/FUTURE_GHI_COT_PILOT_DIAGNOSIS_20261019.md',
 '最新三种子screen中总体差异很小且区间跨零；训练forecast为in-sample，五折OOF复验尚未完成。',scope='2026-10-19湖南未来GHI筛选；非最终预报结论',roots=['refine-logs/cot_aware_simvp_20261017'])
add('监督来源：CPP与R伪标签是否等价？',
 'refine-logs/cot_aware_simvp_20261017/results/label_source_disagreement_summary.md',
 '共同均值接近掩盖近零虚警与厚云压缩；R回退区与CPP覆盖区的云况构成也不同。',scope='2026-10-19验证抽样1,024窗口；参考差异诊断')
add('云与厚云检出：FAR、POD、CSI分别说明什么？',
 'refine-logs/cot_value_first_20261013/COT_CLOUD_DETECTION_VALIDATION_20261013.md',
 '回归误差与云检出应同时报告：POD看漏报，FAR看虚警，CSI综合命中、漏报与虚警；分母与阈值按原报告。',
 scope='湖南反演验证；CPP-COT阈值识别',figs=['docs/组会原始材料_20261016/02_厚度分层/COT云与厚云检出指标_R验证.png'])
add('数据划分：每个月的训练和验证覆盖如何？',
 'docs/MONTHLY_BLOCK_SPLIT_20261002.md',
 '明确连续日期留出与缓冲，解释四季覆盖；这页是结果评价的样本定义，不是模型技能。',
 scope='新月度日期划分；test仅路径和日期元数据',figs=['experiments/monthly_block_20261002/manifests_v1/monthly_split*.png'])

# Historical branches are retained separately and never pooled with current monthly-split scores.
for title,report,meaning,figs in [
 ('早期统计COT输入','docs/HEAD_GHI_MULTISEED_RESULTS_20260910.md','历史统计表示种子方向不一致；只保留负结果锚点，不恢复该表示。',[]),
 ('空间COT初筛','docs/SPATIAL_B1_PILOT_RESULTS_20260917.md','早期空间图输入只带来很小点改善，属于旧数据和旧头部探索。',[]),
 ('空间排列控制','docs/SPATIAL_B2_CONTROLS_RESULTS_20260917.md','检查空间置换与容量因素；不得据早期头部结论否定全部空间表示。',[]),
 ('AFNO固定GHI头替换','results/afno_fixed_ghi_head_swap_20260923/README.md','替换外推器输入用于检查分布变化；不等于各自匹配训练后的优劣。',['results/afno_fixed_ghi_head_swap_20260923/*.png']),
 ('输运与厚度残差','docs/SILI_TRANSPORT_THICKNESS_PILOT_RESULTS_20260928.md','旧输运与学习厚度残差比较，不与月度新划分合并。',[]),
 ('完整历史与重复末帧','docs/SILI_FULL_COT_HISTORY_PILOT_RESULTS_20260928.md','历史通道堆叠未获得稳定收益；并非显式云轨迹建模的否定证据。',[]),
 ('低COT惩罚消融','docs/SILI_LOW_COT_PENALTY_ABLATION_RESULTS_20260928.md','关闭低COT惩罚后仍未超过输运，说明该项不足以解释全部退化。',[]),
 ('历史信息探针','docs/SILI_HISTORY_INFORMATION_PROBE_RESULTS_20260928.md','固定模型输入替换检查历史是否被使用；属于依赖性诊断。',[]),
 ('竺家残差容量对照','docs/ZHUJIA_RESIDUAL_CAPACITY_CONTROL_RESULTS_20260927.md','历史残差容量匹配结果用于追溯路线，不替代统一输入新对照。',[]),
 ('旧天气分层','docs/STATION_WEATHER_COT_DIAGNOSIS_20260912.md','保留天气与站点差异的早期描述；不同样本和模型不能直接纵向归因。',[])]:
    if (ROOT/report).exists():add('历史探索：'+title,report,meaning,scope='历史划分/模型；不进入当前主成绩',figs=figs,status='历史探索')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def clean(s):
    s=re.sub(r'!\[.*?\]\(.*?\)','',str(s))
    s=re.sub(r'\[([^\]]+)\]\([^)]+\)',r'\1',s)
    s=s.replace('**','').replace('`','').replace('$','').replace('\\','/')
    return s.replace('≤','≤').strip()
def tex(s):
    out=''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}.get(c,c) for c in clean(s))
    return out.replace('/',r'/\allowbreak{}').replace(r'\_',r'\_\allowbreak{}')
def celltex(s):
    labels={'A':'AGRI基线','AC':'AGRI+COT','AP':'AGRI容量对照','AL':'AGRI+R隐藏特征','Seed':'随机种子','seed':'随机种子','Three-seed mean':'三种子均值','FA':'预测AGRI','FAC':'预测AGRI+COT','FAP':'预测AGRI容量对照','AC−A':'COT−基线','AC−AP':'COT−容量对照','AC−AL':'COT−R特征','AC-A':'COT−基线','AC-AP':'COT−容量对照'}
    return tex(labels.get(str(s),s))
def tables(text):
    lines=text.splitlines();out=[];heading='数值对照';i=0
    while i<len(lines):
        line=lines[i]
        if line.startswith('#'):heading=line.lstrip('# ').strip()
        if line.strip().startswith('|') and i+1<len(lines) and re.fullmatch(r'[\s|:\-]+',lines[i+1]):
            block=[line];i+=2
            while i<len(lines) and lines[i].strip().startswith('|'):block.append(lines[i]);i+=1
            rows=[[clean(x) for x in re.split(r'(?<!\\)\|',r.strip().strip('|'))] for r in block]
            n=len(rows[0]);rows=[(r+['']*n)[:n] for r in rows]
            if len(rows)>1:out.append((heading,rows))
            continue
        i+=1
    return out

pages=[];assets=[];invalid=[];seenfig=set();code_paths=set();source_paths=set()
def page(title,meaning,scope,kind,payload,source='',group='总览'):
    pages.append(dict(id=f'P{len(pages)+1:03}',title=title,meaning=meaning,scope=scope,kind=kind,payload=payload,source=source,group=group))

page('当前证据总览：哪些问题已回答，哪些仍待验证？','按科学问题定位材料；各区域与模型设置分开，不将不同实验的百分比拼接。','截至2026-10-19的文件核对；服务器快照14:13 BJT','table',[
 ['问题','当前完成结果','可支持的判断'],
 ['同期COT价值','湖南约3.02%；湖北约4.49% RMSE降低','当前紧凑估计器的验证结果有增益'],
 ['增益是否只因容量','容量对照未复现全部收益','超过本实验增加头部容量的收益'],
 ['显式COT是否优于学习特征','同R隐藏表示、强AGRI残差差异区间跨零','未识别显式COT独有优势'],
 ['何种云厚收益较大','湖南CPP厚云子组约22.45%','条件性收益；不是总体或所有厚云'],
 ['COT未来预报','完整3,140序列、46个CPP日期；优于持续性','存在COT预报技巧，CPP仍为参考'],
 ['未来GHI净增益','湖北匹配预报无明确改善；湖南新pilot区间跨零','暂不能声称COT改善未来GHI'],
 ['10月19日监督改进','混合辅助及CPP-only均未通过预定筛选','厚云CSI小改进伴随虚警增加'],
 ['最新在途','湖南五折OOF输入正在重建','尚无OOF重训后的GHI最终成绩'],
 ['独立测试','当前test/buffer仍关闭','本册不包含独立test成绩']])

# Main paper figures retained with their own purposes, rather than screenshotting manuscript pages.
for name,title,meaning in [
 ('fig1_study_regions_coverage.png','研究范围：五站、两个独立区域与日期覆盖','说明数据范围；两个区域分别训练、分别评价，不称零样本迁移。'),
 ('fig2_cot_estimation_forecast_workflow.png','方法关系：同期COT增量与未来应用如何区分？','图区分观测AGRI反演COT与预测AGRI反演COT，避免混淆同期和预报。')]:
    p=ROOT/'paper/figures'/name
    if p.exists():
        dest=FIGURES/f'overview_{name}';shutil.copy2(p,dest);seenfig.add(sha(p))
        page(title,meaning,'研究设计示意；不作为实验成绩','figure',dest.relative_to(HERE).as_posix(),str(p))

for g in groups:
    p=ROOT/g['report'];text=p.read_text(encoding='utf-8-sig')
    start_pages=len(pages)
    source_paths.add(p)
    snapshot=SOURCES/g['id']/p.name;snapshot.parent.mkdir(exist_ok=True);shutil.copy2(p,snapshot)
    assets.append(dict(id=g['id'],title=g['title'],source=g['report'],sha256=sha(p),status=g['status']))
    for heading,rows in tables(text):
        # Operational provenance tables are catalogued, not presented as scientific results.
        if any(w in '|'.join(rows[0]) for w in ('SHA-256','任务号','服务器','退出码','下一步','文件路径')):continue
        for j in range(1,len(rows),12):
            payload=[rows[0]]+rows[j:j+12]
            suffix=f'（{j}–{min(j+11,len(rows)-1)}行）' if len(rows)>13 else ''
            page(g['title']+'｜'+heading+suffix,g['meaning'],g['scope']+'；'+g['status'],'table',payload,g['report'],g['id'])
    candidates=[]
    for pattern in g['figs']:candidates.extend(ROOT.glob(pattern))
    for rel in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text):
        if rel.startswith(('http:','https:')):continue
        q=Path(rel) if re.match(r'^[A-Za-z]:',rel) else p.parent/rel
        if q.is_file() and q.suffix.lower() in ('.png','.jpg','.jpeg','.pdf'):candidates.append(q.resolve())
    for fig in sorted(set(candidates)):
        if any((par/'INVALID_CPP_JOIN.md').exists() for par in [fig.parent]):
            invalid.append(dict(path=str(fig),reason='目录存在INVALID_CPP_JOIN.md；不收录原错误图'));continue
        digest=sha(fig)
        if digest in seenfig:continue
        seenfig.add(digest);dest=FIGURES/f'{g["id"]}_{len(seenfig):03}{fig.suffix.lower()}';shutil.copy2(fig,dest)
        title=g['title']+'｜配套图 '+str(len(seenfig))
        page(title,g['meaning'],g['scope']+'；'+g['status'],'figure',dest.relative_to(HERE).as_posix(),fig.relative_to(ROOT).as_posix(),g['id'])
    if len(pages)==start_pages:
        # Some newer analysis reports express numerical results in prose only.
        evidence=[]
        for para in re.split(r'\n\s*\n',text):
            v=clean(para)
            if len(v)>35 and re.search(r'\d\.\d|RMSE|MAE|CSI',v) and not any(x in v for x in ('SHA','checkpoint','python ','PBS','```')) and not v.startswith('#'):
                evidence.append(v[:900])
            if len(evidence)==3:break
        rows=[['内容','原报告数值与解释'],['核心判断',g['meaning']]]
        for k,v in enumerate(evidence,1):rows.append([f'报告证据{k}',v])
        page(g['title']+'｜关键证据摘录',g['meaning'],g['scope']+'；'+g['status'],'table',rows,g['report'],g['id'])
    for root in g['roots']:
        for q in (ROOT/root).rglob('*'):
            if q.is_file() and q.suffix.lower() in ('.py','.pbs','.sh','.yaml','.yml') and not any(x in q.parts for x in ('__pycache__','python_deps','node_modules')):
                code_paths.add(q)

# Latest server JSONs were fetched directly in this run. Build numbers from their actual scale fields.
for fname,label in [('mixed_independent_cot_audit.json','CPP/R混合辅助监督'),('cpp_only_independent_cot_audit.json','CPP-only辅助监督')]:
    path=SOURCES/'remote_20261019'/fname;d=json.loads(path.read_text(encoding='utf-8'))
    for scale,stitle in [('station_local_3x3','站点中心3×3'),('patch_16x16','完整16×16')]:
        z=d['scales'][scale];rows=[['方法','COT RMSE','厚云CSI（≥30）','近零虚警率（参考<1、预测≥1）']]
        for key,nm in [('original_simvp','原始SimVP→R'),('mse_4000','等预算MSE微调→R'),('cot_aux_4000',label+'→R')]:
            q=z['scores'][key];rows.append([nm,f'{q["station_lead_macro_rmse"]:.6f}',f'{q["pooled_thick_csi_cot_ge30"]:.6f}',f'{q["near_zero_false_alarm_rate_cpp_lt1_pred_ge1"]:.6f}'])
        page('10月19日最终复算：'+label+'能否保住云厚？｜'+stitle,
             '同一CPP有效像元：总体RMSE未超过原始基线；CSI小改善与近零虚警增大并存，候选未通过预定筛选。',
             f'3,140验证序列；46个CPP目标日期；seed42；COT无量纲；{stitle}', 'table',rows,path.relative_to(HERE).as_posix(),'最新辅助监督')
        rows=[['比较','ΔRMSE（候选−对照）','配对目标日期95%区间']]
        for key,nm in [('cot_minus_original','候选−原始SimVP'),('cot_minus_equal_budget_mse','候选−等预算MSE')]:
            lo,hi=z['paired_bjt_target_date_bootstrap_95ci'][key]
            rows.append([nm,f'{z["point_delta"][key]:+.6f}',f'[{lo:+.6f}, {hi:+.6f}]'])
        page('10月19日不确定性：'+label+'的RMSE差是否明确？｜'+stitle,
             '区间跨零，未证实COT RMSE净改善；厚云CSI与虚警应同时检查，不能仅挑有利指标。',
             '同样本、同日期配对10,000次bootstrap；不是独立test','table',rows,path.relative_to(HERE).as_posix(),'最新辅助监督')

fap_source=ROOT/'refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_fap_three_seed_20261018/forecast_fap_three_seed_validation_recheck.json'
if fap_source.exists():
    d=json.loads(fap_source.read_text(encoding='utf-8-sig'))
    shutil.copy2(fap_source,SOURCES/'hubei_forecast_three_arm_final.json')
    rows=[['未来GHI输入','三站三种子等权RMSE（W/m²）','性质']]
    for key,nm in [('FA','预测AGRI'),('FAC','预测AGRI＋预测COT'),('FAP','预测AGRI＋容量对照')]:
        rows.append([nm,f'{d["equal_station_then_seed_rmse_wm2"][key]:.6f}','同一validation选模期评价'])
    page('湖北最新完整预报对照：COT与容量分支都有效吗？','包含10月18日补齐的容量臂；三组比较均未显示明确未来GHI改善。','湖北三站、三种子、22目标日期；最新归一化重载；不是test','table',rows,fap_source.relative_to(ROOT).as_posix(),'湖北最终三臂')
    rows=[['配对比较','ΔRMSE（W/m²）','日期块95%区间']]
    for key in ('FAC_minus_FA','FAP_minus_FA','FAP_minus_FAC'):
        q=d['paired_date_cluster_bootstrap'][key];lo,hi=q['ci95_wm2']
        rows.append([key.replace('_minus_',' − '),f'{q["estimate_difference_wm2"]:+.6f}',f'[{lo:+.6f}, {hi:+.6f}]'])
    page('湖北未来GHI不确定性：预报收益尚未得到支持','区间跨零既不证明改善，也不证明三种方法等效；保留全部正负结果。','10,000次配对目标日期bootstrap；Δ为前者减后者','table',rows,fap_source.relative_to(ROOT).as_posix(),'湖北最终三臂')

if (HERE/'REMOTE_CHECKPOINT_INDEX.tsv').exists():
    remote_weights=[]
    for line in (HERE/'REMOTE_CHECKPOINT_INDEX.tsv').read_text(encoding='utf-8-sig').splitlines():
        parts=line.split('\t')
        if len(parts)==3:remote_weights.append(parts)
    for offset in range(0,len(remote_weights),8):
        rows=[['远端权重（共同根目录之后）','MiB','远端修改时间']]
        for path,size,stamp in remote_weights[offset:offset+8]:
            rows.append([path.replace('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/',''),f'{int(size)/1024/1024:.1f}',stamp[:19]])
        page('最新服务器权重：COT辅助与未来GHI筛选 '+str(offset//8+1),
             '本次通过SSH读取文件目录和大小；step4000用于已完成辅助监督，OOF过程权重不当作最终模型。',
             '共同根目录见REMOTE_CHECKPOINT_INDEX.tsv绝对路径；本次SSH目录快照','table',rows,'REMOTE_CHECKPOINT_INDEX.tsv','远端现存权重')

# Reproducibility: complete local weight inventory, source/code hashes and explicit remote references.
excluded=('revisions_before','planning_history','python_deps','node_modules','.git')
weights=[]
for q in ROOT.rglob('*'):
    if q.is_file() and q.suffix.lower() in ('.pt','.pth','.ckpt') and not any(x in q.parts for x in excluded):
        rel=q.relative_to(ROOT).as_posix();matches=[g['id'] for g in groups if any(rel.startswith(r+'/') for r in g['roots'])]
        weights.append(dict(id=f'C{len(weights)+1:03}',path=str(q),relative_path=rel,bytes=q.stat().st_size,sha256=sha(q),experiment_ids=';'.join(matches),verification='本次确认本地文件存在并计算字节SHA-256；不是重新运行权重重载测试'))
def writecsv(name,rows):
    with (HERE/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if weights:writecsv('CHECKPOINT_INDEX.csv',weights)
codes=[]
for q in sorted(code_paths):
    rel=q.relative_to(ROOT);dest=CODE/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(q,dest)
    codes.append(dict(path=str(q),snapshot=dest.relative_to(HERE).as_posix(),sha256=sha(q)))
if codes:writecsv('CODE_INDEX.csv',codes)
writecsv('SOURCE_INDEX.csv',assets)
remote_refs=[]
for g in groups:
    text=(ROOT/g['report']).read_text(encoding='utf-8-sig')
    for ref in re.findall(r'`([^`\n]+)`',text):
        if any(k in ref for k in ('/public/home/','/home/Data_Pool','best.pt','best_model','step_4000.pt','checkpoint','train_','analyze_','recompute_')):
            remote_refs.append(dict(experiment_id=g['id'],reference=ref,status='报告登记；远程路径本次未逐个检查',source=g['report']))
if remote_refs:writecsv('DOCUMENTED_ASSET_REFERENCES.csv',remote_refs)

page('复现导航：如何从一页结果找到数据、代码和权重？','图表页标注来源编号；本册sources保留报告快照，code_snapshot保留实际代码。checkpoint只编目，不复制庞大权重。','本次目录盘点，不等同重新完成所有训练','table',[
 ['文件/目录','内容','如何使用'],['PAGE_INDEX.csv','逐页标题、作用、范围、原图/报告路径','按页定位原始证据'],['SOURCE_INDEX.csv 与 sources/','本册采用报告及SHA-256、报告快照','核对数字和原有评价范围'],['CHECKPOINT_INDEX.csv',f'{len(weights)}个本地权重文件：绝对路径、大小、关联实验','找到实际文件；best/last/epoch不能混用'],['CODE_INDEX.csv 与 code_snapshot/',f'{len(codes)}个训练、评价、绘图或执行代码快照及SHA-256','按报告和实验编号定位复现程序'],['DOCUMENTED_ASSET_REFERENCES.csv','报告登记的代码/远程模型位置','标明只登记、未实时逐项远程核验'],['INVALID_RESULTS.csv','作废CPP配对图所在目录','不用于科学成绩'],['main.tex / build_result_book.py','可编辑LaTeX与结果册生成器','使用XeLaTeX编译；一图或一表一页']])

for offset in range(0,len(groups),9):
    rows=[['编号','科学问题/实验','来源报告（仓库相对路径）']]
    for g in groups[offset:offset+9]:rows.append([g['id'],g['title'],g['report']])
    page('结果来源索引：科学问题对应哪个报告？', '所有路径以solar-cot-ifs仓库根目录为基准；全文报告快照另存sources。','来源目录','table',rows,group='复现索引')

for g in groups:
    matched=[w for w in weights if g['id'] in w['experiment_ids'].split(';')]
    related=[c for c in codes if any(Path(c['path']).relative_to(ROOT).as_posix().startswith(r+'/') for r in g['roots'])]
    if not matched and not related:continue
    rows=[['项目','位置/数量','说明'],['报告',g['report'],'报告快照见sources/'+g['id']],['本地权重',str(len(matched))+'个；'+(', '.join(w['id'] for w in matched[:8])),'完整绝对路径见CHECKPOINT_INDEX.csv']]
    # Representative exact paths, with all remaining paths in the accompanying machine-readable inventory.
    selected=[w for w in matched if w['relative_path'].endswith(('best.pt','best_model.pth','best.ckpt'))][:3] or matched[:2]
    for w in selected:rows.append(['权重示例 '+w['id'],w['relative_path'],f'{w["bytes"]/1024/1024:.1f} MiB；本地存在'])
    rank=lambda c:(0 if any(x in Path(c['path']).name for x in ('train','analyze','recompute','audit','plot')) else 1,Path(c['path']).name)
    for c in sorted(related,key=rank)[:4]:rows.append(['代码',Path(c['path']).relative_to(ROOT).as_posix(),'已复制code_snapshot；SHA见CODE_INDEX.csv'])
    page('代码与checkpoint：'+g['title'],'将该实验的结果、实现和现存权重连到一起；示例路径不是自动选模指令。','工程复现附录','table',rows,g['report'],g['id'])

# Complete local checkpoint list, paginated; file sizes are inventory, not scientific scores.
for offset in range(0,len(weights),10):
    rows=[['编号','本地权重（仓库相对路径）','大小MiB','关联实验']]
    for w in weights[offset:offset+10]:rows.append([w['id'],w['relative_path'],f'{w["bytes"]/1024/1024:.1f}',w['experiment_ids'] or '历史/共享；见原目录'])
    page(f'完整checkpoint目录：{offset+1}–{min(offset+10,len(weights))} / {len(weights)}','便于定位已有权重；不将本地存在误写成重新完成重载验收。','本次实际文件盘点；绝对路径保存在CSV','table',rows,group='权重目录')

for marker in ROOT.rglob('INVALID_CPP_JOIN.md'):
    invalid.append(dict(path=marker.relative_to(ROOT).as_posix(),reason=marker.read_text(encoding='utf-8-sig')[:500]))
if invalid:writecsv('INVALID_RESULTS.csv',invalid)
page('作废材料：哪些CPP评价图不能再引用？','旧CPP局部/全局索引错配的图已作废；旧29.13/23.04等数值不作为结果收录。','工程更正记录；不与修复后分数混算','table',[
 ['材料','处理','当前采用来源'],['cot_forecast_eval旧v1–v5及相关bank/个例旧图','INVALID_CPP_JOIN目录标记；不收录原错误图','paper/figures/fig7_forecast_cot_summary_20261017.json'],['旧8日期271窗口修正值','有界历史诊断；不冒充完整验证','最新3,140序列、46CPP日期同掩膜评价'],['湖北旧未保存归一化run汇总','保留历史；不作为最新主表','2026-10-17 normreload与10-18独立复算'],['合成COT示意场','仅用于解释结构','本册优先收录真实数据图'],['未完成OOF训练/预测','无最终成绩可填','单列在途状态，不预写收益']])

directory=[];ndirs=(len(groups)+10)//11
for offset in range(0,len(groups),11):
    rows=[['实验','要回答的问题','本册页码']]
    for g in groups[offset:offset+11]:
        pos=[i for i,p in enumerate(pages) if p['group']==g['id'] and p['scope']!='工程复现附录']
        rows.append([g['id'],g['title'],f'{min(pos)+1+ndirs}–{max(pos)+1+ndirs}' if pos else '见来源索引'])
    directory.append(dict(title='阅读目录：按科学问题查找 '+str(offset//11+1),meaning='页码直接对应本PDF；每页一个表或图。最新辅助监督、资产与权重目录位于后半册。',scope='完整结果＋历史探索＋工程复现',kind='table',payload=rows,source='PAGE_INDEX.csv',group='目录'))
pages[1:1]=directory
for i,p in enumerate(pages):p['id']=f'P{i+1:03}'

# Render deterministic LaTeX. Every page contains exactly one table or one figure.
head=r'''\documentclass[UTF8,a4paper,landscape,10pt]{ctexart}
\usepackage[margin=14mm,headheight=14pt,footskip=8mm]{geometry}
\usepackage{graphicx,adjustbox,booktabs,tabularx,array,xcolor,fancyhdr,url}
\setmainfont{Times New Roman}
\definecolor{ink}{HTML}{18334D}\definecolor{muted}{HTML}{526678}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small COT研究结果总册}\fancyhead[R]{\small 2026-10-19 核对版}\fancyfoot[C]{\thepage}
\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.32}\setlength{\emergencystretch}{3em}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\begin{document}
'''
chunks=[head]
for i,p in enumerate(pages):
    if i:chunks.append(r'\clearpage'+'\n')
    chunks.append(r'{\large\bfseries\color{ink} '+tex(p['id']+'  '+p['title'])+r'}\par'+'\n')
    chunks.append(r'\vspace{2mm}{\normalsize\textbf{本页意义：}'+tex(p['meaning'])+r'}\par'+'\n')
    chunks.append(r'{\small\color{muted}\textbf{评价范围：}'+tex(p['scope'])+r'}\par\vspace{3mm}'+'\n')
    if p['kind']=='figure':
        chunks.append(r'\begin{center}\includegraphics[width=\linewidth,height=0.61\textheight,keepaspectratio]{'+p['payload']+r'}\end{center}'+'\n')
    else:
        rows=p['payload'];n=len(rows[0]);size=r'\small' if n<7 else r'\footnotesize'
        # Explicit weighted widths keep long first-column event labels readable.
        lengths=[max(len(str(r[j])) for r in rows) for j in range(n)]
        if n==3 and '远端权重' in rows[0][0]:column_weights=[2.2,.3,.5]
        elif n==3 and rows[0][0]=='编号':column_weights=[.3,.9,1.8]
        elif n==3 and rows[0][0]=='项目':column_weights=[.35,1.9,.75]
        elif n>=7:column_weights=[1.7]+[(n-1.7)/(n-1)]*(n-1)
        elif n==2:column_weights=[.75,1.25]
        elif n==3 and max(lengths)>60:column_weights=[.45,.8,1.75]
        elif n==4 and max(lengths)>70:column_weights=[.4,2.7,.45,.45]
        else:column_weights=[1.0]*n
        cols=''.join('>{\\hsize='+f'{w:.4f}'+r'\hsize\raggedright\arraybackslash}X' for w in column_weights)
        chunks.append(r'\begin{center}'+size+'\n'+r'\begin{adjustbox}{max totalheight=0.58\textheight,max width=\linewidth}'+ '\n'+r'\begin{minipage}{\linewidth}\begin{tabularx}{\linewidth}{'+cols+'}'+r'\toprule'+'\n')
        chunks.append(' & '.join(r'\textbf{'+celltex(c)+'}' for c in rows[0])+r' \\ \midrule'+'\n')
        for row in rows[1:]:chunks.append(' & '.join(celltex(c) for c in row)+r' \\'+'\n')
        chunks.append(r'\bottomrule\end{tabularx}\end{minipage}\end{adjustbox}\end{center}'+'\n')
    chunks.append(r'\vfill{\scriptsize\color{muted} 来源编号：'+tex(p['group'])+r'。 原始文件：'+tex(p['source'] or '本次综合核对；见来源索引')+r'}\par'+'\n')
chunks.append(r'\end{document}')
(HERE/'main.tex').write_text(''.join(chunks),encoding='utf-8')
index=[{k:v for k,v in p.items() if k!='payload'} for p in pages]
writecsv('PAGE_INDEX.csv',index)
(HERE/'page_manifest.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
(HERE/'main_zh.md').write_text('# COT研究结果总册\n\n'+ '\n\n'.join('## '+p['id']+' '+p['title']+'\n\n本页意义：'+p['meaning']+'\n\n评价范围：'+p['scope']+'\n\n'+ ('![]('+p['payload']+')' if p['kind']=='figure' else '\n'.join(['| '+' | '.join(r)+' |' for r in [p['payload'][0],['---']*len(p['payload'][0])]+p['payload'][1:]]))+'\n\n来源：'+p['source'] for p in pages),encoding='utf-8')
summary=dict(pages=len(pages),figures=sum(p['kind']=='figure' for p in pages),tables=sum(p['kind']=='table' for p in pages),result_families=len(groups),local_checkpoints=len(weights),code_files=len(codes),invalid_records=len(invalid))
(HERE/'BUILD_MANIFEST.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
