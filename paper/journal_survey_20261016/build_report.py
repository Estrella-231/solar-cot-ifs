"""Build an offline, source-linked journal decision report. Python stdlib only."""
import json
from pathlib import Path
from html import escape as E

ROOT = Path(__file__).resolve().parent
D = []
def add(name, field, tier, cas, jcr, feature, value, fit, gap, official, ranking, note='', fee='本轮未核实当前金额；以出版社投稿报价为准'):
    D.append(dict(name=name, field=field, tier=tier, cas=cas, jcr=jcr, feature=feature, value=value, fit=fit, gap=gap, official=official, ranking=ranking, note=note, fee=fee))

add('Solar Energy','能源','重点候选','2区 · 工程技术','Q2 · Energy & Fuels',
    '太阳能资源、辐射测量、估计和应用，是最接近本文问题的读者群。',
    '太阳能专业主流刊，ISES学会期刊；专业匹配和领域积累强，不能因为Q2就视为普通兜底。',
    '首选：以COT表示如何改善地表GHI估计为核心，短临预报作延伸。',
    '补齐两区域重复验证、最终留出评价及云况分层；不把3%本身当作创新。',
    'https://www.ises.org/what-we-do/publications/solar-energy-journal','https://www.ablesci.com/journal/detail?id=5VvYW5',
    'CAS能源与燃料小类3区；JCR公开页2026更新。')
add('Renewable Energy','能源','重点候选','1区 · 工程技术','Q1 · 公开汇总，学科版本需复核',
    '可再生能源技术与应用，涵盖光伏和气候气象；官方范围强调与可再生发电的联系。',
    '可再生能源主流强刊，较适合作为一区目标；比单纯追求通用高影响因子更贴近项目。',
    '若五站两区域结果稳定，可作为首投冲刺；叙事要突出太阳能资源评估的实用意义。',
    '补强跨区域稳定性、天气依赖和预报应用；说明GHI误差改善为何影响太阳能利用，不必硬加调度模型。',
    'https://shop.elsevier.com/journals/renewable-energy/0960-1481','https://www.iikx.com/sci/technology/16712.html',
    'CAS能源与燃料、绿色可持续技术小类均2区。')
add('International Journal of Applied Earth Observation and Geoinformation (IJAEOG / JAG)','遥感','重点候选','1区 · 地球科学','Q1 · 公开汇总，学科版本需复核',
    '应用地球观测、地理信息、数据质量与空间不确定性，欢迎GeoAI方法与应用。',
    '应用遥感强刊。适合有清晰地学价值的遥感研究，不能把“应用”理解成只需搭网络。',
    '有条件的一区候选：突出云光学属性的物理表示与跨区域地表辐射估计。',
    '需要比五站数值表更丰富的真实空间云场、误差机制和区域代表性；范围适合不等于已达到录用水平。',
    'https://shop.elsevier.com/journals/international-journal-of-applied-earth-observation-and-geoinformation/1569-8432','https://www.fabiao.com.cn/sci/03032434/',
    'CAS遥感小类2区。不要与ISPRS Journal of Photogrammetry and Remote Sensing混淆。')
add('IEEE JSTARS','遥感','重点候选','2区 · 地球科学','Q1 · 遥感/自然地理/电子电气；Q2 · 成像',
    '应用遥感、原位观测整合与地球系统建模；明确接收预测信息和应用建模。',
    'IEEE GRSS专业应用期刊，领域识别度强；不是TGRS的同义替代。',
    '遥感路线优先备选：按卫星输入、物理表示、地面验证、误差传播组织论文。',
    '需要新颖且有意义的技术内容；突出受控对照、COT来源和结果可重复性。',
    'https://www.grss-ieee.org/publications/journal-of-selected-topics-in-applied-earth-observations-and-remote-sensing/','https://www.ablesci.com/journal/detail?id=5NQk7p',
    '2025 CAS四个相关小类均3区；JCR为公开页2026更新。','完全OA；官网列2025-01-01起投稿APC US$1,800，实际税费/优惠以账单为准。')
add('Remote Sensing','遥感','重点候选','2区 · 地球科学','Q1 · 地学综合；Q2 · 遥感/环境/成像',
    '覆盖传感器、遥感方法、验证及应用，可容纳完整云属性与辐射实验链。',
    '领域覆盖广、传播方便；在不同单位的考核与声誉评价有差异，需按本单位规则判断。',
    '主题匹配的备选，不宜仅因官网标Q1就认为比其他专业刊更强。',
    '保持清晰科学贡献，避免工程过程堆叠；常规栏目和特刊均需判断编辑方向。',
    'https://www.mdpi.com/journal/remotesensing/stats','https://www.scholay.com/journal/15992/remote-sensing',
    'JCR采用官网截至2026-07-07分类；CAS来自2025版二手汇总。','完全OA；当前APC以MDPI官方APC页面为准，本轮未记录报价。')
add('Atmospheric Measurement Techniques (AMT)','大气','条件候选','3区 · 地球科学','Q2 · 气象与大气科学',
    '大气测量、反演技术、校准和验证；已有云/气溶胶影响太阳辐射短临预报的研究。',
    'EGU专业测量方法期刊，适合严谨反演与不确定性工作；CAS3区不代表专业价值低。',
    '若重点是COT辅助GHI估计及误差来源，匹配度高；不适合只报神经网络性能。',
    '解释检索参考误差、时空代表性、地面测量和辐射物理；需要方法学价值。',
    'https://amt.copernicus.org/articles/17/1851/2024/','https://topj.lib.whu.edu.cn/show.asp?cat=zk&id=177',
    '高校图书馆页CAS年份未明；2025版另由iikx/ais公开汇总交叉核对。','开放获取；收费与文章长度/作者协议有关，以Copernicus当前报价为准。')
add('Atmospheric Research','大气','条件候选','2区 · 地球科学','Q2 · 气象与大气科学',
    '侧重云物理、气溶胶、云动力及大气过程。',
    '大气过程专业主流刊；对真实物理解释的期待高于单纯应用指标比较。',
    '适合把增益随云厚、太阳几何、云团演变变化的原因讲透。',
    '需要云辐射机制证据；不能把外推失败全部笼统归因于骨干局限。',
    'https://www.sciencedirect.com/journal/atmospheric-research','https://www.ablesci.com/journal/detail?id=npBn6p',
    'JCR公开页2026更新为Q2；不能沿用更早年份Q1。')
add('Journal of Applied Meteorology and Climatology','大气','条件候选','3区 · 地球科学','Q3 · 气象与大气科学',
    '应用气象、辐射、观测与气候应用；AMS学会刊。',
    '传统专业期刊，学术读者明确；国际专业认可与国内分区激励不完全一致。',
    '以云况依赖的辐射估计与能源气象应用为主时可以考虑。',
    '须呈现有气象意义的解释、季节与天气评价，不能只有深度学习架构。',
    'https://journals.ametsoc.org/view/journals/apme/apme-overview.xml','https://www.ablesci.com/journal/detail?id=rwX79p',
    'CAS气象小类4区；JCR公开页2026更新。')
add('Weather and Forecasting','大气','条件候选','3区 · 地球科学','Q3 · 气象与大气科学',
    '预报方法、预报检验与对预报员有用的案例；AMS学会刊。',
    '天气预报方向专业辨识度高，价值不只看影响因子。',
    '只有预报成为实质贡献才适合；当前“同期反演主线”不是最自然选择。',
    '先有严格因果的0–4小时收益、时效曲线和可靠基准，不能拿同期估计代替预报。',
    'https://journals.ametsoc.org/view/journals/wefo/wefo-overview.xml','https://www.goingpub.com/journal_detail/weather-and-forecasting.html',
    'CAS2025；JCR Q3参考科研通weather检索页，公开2026更新。')
add('Meteorological Applications','大气','条件候选','4区 · 地球科学','待核 · 不用SJR Q2替代',
    '观测、预报、验证、数据同化与气象服务，官方明确涵盖可再生能源。',
    '皇家气象学会应用期刊，读者贴合能源气象；分区目标较高时不优先。',
    '当贡献以业务应用验证为主时可考虑。',
    '说明气象服务价值，完整比较季节/天气和误差；本轮JCR年份未闭合。',
    'https://www.rmets.org/publication/meteorological-applications','https://journalcompass.com/journals/meteorological-applications-13504827.html',
    '2025CAS4区；网页2026新锐3区不是同一分区。','完全OA（学会官网确认）；当前APC待核。')
add('Sustainable Energy Technologies and Assessments','能源','条件候选','2区 · 工程技术','Q2 · 能源/绿色可持续技术',
    '可持续能源技术及其性能、经济性和应用评估。',
    '能源应用期刊，适合技术评估故事；与纯遥感反演相比需更明确能源落点。',
    '若明确GHI估计对太阳能技术评估的作用，可列备选。',
    '避免只换模型；需要应用指标或可信的能源意义，但不为投刊硬凑经济模型。',
    'https://shop.elsevier.com/journals/sustainable-energy-technologies-and-assessments/2213-1388','https://www.scholay.com/journal/26329/sustainable-energy-technologies-and-assessments',
    'JCR Q2另见Comillas大学期刊页；CAS2025二手。')
add('International Journal of Remote Sensing','遥感','备选','4区 · 地球科学','Q3 · 遥感/成像',
    '遥感方法与应用的广范围传统期刊。',
    '老牌专业期刊，适合完整、可靠的遥感研究；国内分区收益低于前列候选。',
    '科学问题完整但创新幅度有限时，可作备选。',
    '保持地面验证和方法对照完整；不因备选就省略必要证据。',
    'https://www.tandfonline.com/journals/tres20/about-this-journal','https://www.ablesci.com/journal/detail?id=D4bRLp',
    '2025大类4区；不要套用历史3区印象。')
add('Journal of Solar Energy Engineering','能源','备选','4区 · 工程技术','Q4 · 能源；Q2 · 机械工程',
    'ASME太阳能工程专业刊，侧重工程方法、系统和应用。',
    '学会专业背景清楚；适合工程社区，但CAS要求较高时不优先。',
    '若突出辐射资源评估的工程用途可以考虑。',
    '需要太阳能工程意义；不能只引用该刊机械工程Q2来代表能源分区。',
    'https://asmedigitalcollection.asme.org/solarenergyengineering','https://www.ablesci.com/journal/detail?id=p102dr',
    'JCR公开页2026更新；两学科分区不同。')
add('Journal of Renewable and Sustainable Energy','能源','备选','4区 · 工程技术（年份待复核）','Q4 · 公开汇总，版本待核',
    'AIP可再生与可持续能源研究，强调物理和工程应用。',
    '专业出版社期刊，适合规模适中的应用工作；不是高分区首选。',
    '保留为后续备选，不排在Solar Energy、JSTARS之前。',
    '补齐能源应用解释；分区来源证据弱于重点候选，正式选刊前需数据库确认。',
    'https://pubs.aip.org/jrse','https://www.minicod.com/journals/journal-of-renewable-and-sustainable-energy-2ef6b53b')
add('Energies','能源','备选','4区 · 工程技术','Q3 · 能源与燃料（公开汇总）',
    '能源领域综合开放获取期刊，太阳能估计和预测在其广泛范围内。',
    '广覆盖应用传播渠道；具体单位对期刊/特刊的政策须单独核验。',
    '可作备选，但以当前投入程度不建议直接作为首投目标。',
    '不要将易于匹配范围误当成录用保证；保留完整方法、误差与负结果。',
    'https://www.mdpi.com/journal/energies','https://www.xueshu.com/sci/40128/',
    '2025CAS4区，2026新锐3区不可混用。','完全OA；APC本轮未核实。')
add('Applied Energy','能源','高门槛冲刺','1区 · 工程技术','Q1 · 能源/化工',
    '综合能源应用与方法，已发表卫星光谱深度学习辐射估计工作。',
    '能源领域高认可度期刊，竞争强；同题论文存在只说明范围匹配。',
    '可以保留冲刺目标，但目前不宜因为约3%增益直接判定够投。',
    '需要明显新知识、稳定跨区域证据、强基准和能源意义；若新增证据不支持，不强行扩大故事。',
    'https://www.sciencedirect.com/science/article/pii/S0306261923013430','https://www.ablesci.com/journal/detail?id=d5QZRD')
add('Energy','能源','高门槛冲刺','1区 · 工程技术','Q1 · 能源/热力学',
    '广泛能源研究、技术与应用。',
    '综合能源主流强刊，但其大范围不会自动使小样本估计研究更有竞争力。',
    '方法可靠、跨区域与应用解释充分后才值得考虑。',
    '需要超越单一数据集的增益及能源研究意义，当前匹配度低于Solar Energy。',
    'https://www.sciencedirect.com/journal/energy','https://www.ablesci.com/journal/detail?id=DGLvAr')
add('Energy Conversion and Management','能源','高门槛冲刺','1区 · 工程技术','Q1 · 能源/力学/热力学',
    '能源转换、管理与系统分析。',
    '能源工程强刊；专业核心比卫星GHI估计更偏转换与管理。',
    '条件性冲刺，当前不值得为迎合范围强加储能调度支线。',
    '若没有与能源转换或管理的实质关系，编辑初筛适配性会成为问题。',
    'https://www.sciencedirect.com/journal/energy-conversion-and-management','https://www.ablesci.com/journal/detail?id=pe962p')
add('Remote Sensing of Environment','遥感','高门槛冲刺','1区 · 地球科学','Q1 · 遥感/成像/环境',
    '定量环境遥感，重视物理解释、反演方法与可推广的环境认识。',
    '环境定量遥感代表性强刊，科学问题深度与验证广度要求高。',
    '当前更适合作为学习叙事和证据标准的标杆。',
    '仅COT拼接加少量站点增益不足；需广泛独立验证、误差机制和方法推广价值。',
    'https://www.sciencedirect.com/journal/remote-sensing-of-environment','https://www.ablesci.com/journal/detail?id=5jdnjr')
add('IEEE Transactions on Geoscience and Remote Sensing (TGRS)','遥感','高门槛冲刺','1区 · 地球科学','Q1 · 公开汇总',
    '遥感理论、方法、技术与信息处理。',
    'IEEE GRSS旗舰方法刊，技术贡献辨识度高；应用有效并不等于方法创新充分。',
    '除非形成实质遥感方法创新，否则不作为本文当前首选。',
    '不应把第二篇变成新外推网络论文；若需大改技术主线，留给后续工作更合适。',
    'https://www.grss-ieee.org/publications/transactions-on-geoscience-remote-sensing/','https://www.scholay.com/journal/4675/ieee-transactions-on-geoscience-and-remote-sensing',
    '2025大类1区；遥感小类2区，不能只报最优小类。')
add('Energy and AI','能源','条件候选','待核 · 汇总显示工程2区但年份未闭合','待核 · 不把未核年份Q1列为正式依据',
    '能源与人工智能交叉方向；关注AI对能源问题的实质贡献。',
    '年轻交叉期刊，需同时看收录状态、评价年份与机构规则，不能只用高IF判断。',
    '可列观察名单；若侧重物理表示的学习价值，比纯算法堆叠更合理。',
    '本轮官网读取失败、CAS年份和JCR分类未闭合；不据此承诺满足SCI/分区要求。',
    'https://www.sciencedirect.com/journal/energy-and-ai','https://www.aminer.cn/academic/journal/energy-and-ai/63e0caf237a18fe9aa4843d7',
    '保留待核项；未采用仿冒ScienceDirect域名的内容。')
add('IEEE Geoscience and Remote Sensing Letters (GRSL)','遥感','条件候选','3区 · 地球科学','待核 · 不以最佳学科Q1代替逐科查询',
    '遥感短文，适合一个集中、明确的新方法或新发现。',
    '专业学会短文刊；篇幅与本文完整归因链存在张力，不是JSTARS缩短版。',
    '只有收敛为一个独立核心结果才考虑；本稿更适合完整研究论文。',
    '需要压缩科学问题而不删必要验证；先核实当前篇幅和费用规则。',
    'https://www.grss-ieee.org/publications/geoscience-and-remote-sensing-letters/','https://journalcompass.com/journals/ieee-geoscience-and-remote-sensing-letters-1545598x.html',
    '2025遥感小类4区，其他相关小类3区。')

(ROOT/'journals.json').write_text(json.dumps(D,ensure_ascii=False,indent=2),encoding='utf-8')
cards=[]
rows=[]
for i,j in enumerate(D,1):
    j['id']=f'j{i}'
    rows.append(f'<tr data-id="j{i}"><td><a href="#j{i}">{E(j["name"])}</a><small>{E(j["field"])}</small></td><td>{E(j["tier"])}</td><td>{E(j["cas"])}</td><td>{E(j["jcr"])}</td><td>{E(j["fit"])}</td></tr>')
    cards.append(f'''<article id="j{i}" class="journal"><div class="jtop"><span class="serial">{i:02}</span><div><span class="tag">{E(j['field'])} / {E(j['tier'])}</span><h3>{E(j['name'])}</h3></div></div><div class="ranks"><div><small>2025 中科院大类 · 二手证据</small><b>{E(j['cas'])}</b></div><div><small>JCR · 公开证据（非SJR）</small><b>{E(j['jcr'])}</b></div></div><dl><dt>期刊特点</dt><dd>{E(j['feature'])}</dd><dt>含金量判断</dt><dd>{E(j['value'])}</dd><dt>与本文匹配</dt><dd>{E(j['fit'])}</dd><dt>关键差距</dt><dd>{E(j['gap'])}</dd><dt>费用与出版</dt><dd>{E(j['fee'])}</dd></dl><p class="note">{E(j['note']) or 'JCR采用公开页面所示当前版本；正式考核需机构数据库确认。'}</p><div class="sources"><a href="{E(j['official'])}" target="_blank" rel="noopener">出版社 / 学会 / 已发表论文 ↗</a><a href="{E(j['ranking'])}" target="_blank" rel="noopener">分区证据与年份 ↗</a></div></article>''')

html='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>COT论文投稿地图 · 2026-10-16</title><style>
:root{--ink:#17313b;--muted:#62727b;--teal:#087e83;--line:#dce4e5;--bg:#f3f6f5;--gold:#d3a34c}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:90px}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.7 'Segoe UI','Microsoft YaHei',sans-serif}a{color:var(--teal);text-decoration:none}a:hover{text-decoration:underline}header{background:#132f3b;color:white;padding:54px max(24px,calc((100vw - 1240px)/2)) 44px}header p{max-width:870px;color:#c9d9dd}.eyebrow{font-size:13px;letter-spacing:3px;color:#7fe0d5}h1{font-size:42px;line-height:1.25;margin:12px 0 22px}h2{font-size:25px;margin:0 0 20px}h3{font-size:20px;line-height:1.4;margin:6px 0}p{margin:10px 0}main{max-width:1240px;margin:auto;padding:30px 24px 70px}section{margin:0 0 36px}.intro{background:white;padding:27px;border-left:5px solid var(--teal);border-radius:5px}.hero-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-top:24px}.pick{padding:22px;background:#e8f0ef;border-radius:10px}.pick strong{display:block;font-size:21px;margin:8px 0}.pick small{color:var(--teal)}.notice{background:#fff6e4;border:1px solid #e9d8b3;padding:23px;border-radius:10px;font-size:14px}.notice b{color:#7d5520}.controls{position:sticky;top:0;z-index:5;display:flex;gap:10px;flex-wrap:wrap;padding:14px;background:#f3f6f5f5;border-bottom:1px solid var(--line)}input,select,button{font:inherit;border:1px solid #b7c8cd;background:white;border-radius:6px;padding:9px 12px;color:var(--ink)}input{flex:1;min-width:210px}button{cursor:pointer}button:hover{background:#e8f0ef}.tablewrap{overflow-x:auto;background:white;border:1px solid var(--line);border-radius:8px}table{border-collapse:collapse;width:100%;min-width:980px;font-size:14px}th,td{padding:15px 13px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}th{background:#e8efef;font-size:13px}td:first-child{width:23%;font-weight:600}td:last-child{width:29%}td small{display:block;font-weight:400;color:var(--muted)}.cards{display:grid;grid-template-columns:1fr 1fr;gap:22px}.journal{background:white;padding:25px;border:1px solid var(--line);border-radius:10px}.jtop{display:flex;gap:14px}.serial{font-size:27px;color:#9fb4b7;font-weight:300}.tag{font-size:12px;color:var(--teal)}.ranks{display:grid;grid-template-columns:1fr 1fr;background:#f1f6f5;border-radius:8px;padding:14px;gap:12px;margin:20px 0}.ranks small{display:block;color:var(--muted);font-size:11px}.ranks b{font-size:14px}dl{display:grid;grid-template-columns:92px 1fr;gap:10px;margin:0;font-size:14px}dt{font-weight:600}dd{margin:0}.note{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:12px}.sources{display:flex;flex-wrap:wrap;gap:15px;font-size:12px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:25px}.panel{background:white;border:1px solid var(--line);border-radius:10px;padding:25px}.panel li{margin:12px 0}.muted{color:var(--muted);font-size:14px}.count{color:var(--teal);font-size:14px;margin:10px 0}.hidden{display:none!important}footer{border-top:1px solid var(--line);padding:24px 0;color:var(--muted);font-size:12px}#empty{padding:30px;text-align:center}nav{display:flex;gap:24px;flex-wrap:wrap;font-size:14px}nav a{color:#9ae5dc}@media(max-width:800px){h1{font-size:30px}.hero-grid,.cards,.columns{grid-template-columns:1fr}header{padding:35px 22px}main{padding:22px 15px}.journal{padding:20px}.controls{position:static}dl{grid-template-columns:80px 1fr}}@media print{body{background:white;font-size:11px}header{padding:20px;color:black;background:white}header p{color:#333}.controls,button,nav{display:none}.cards{display:block}.journal{break-inside:avoid;margin-bottom:20px}table{min-width:0;font-size:10px}.tablewrap{overflow:visible}.hidden{display:none!important}main{max-width:none;padding:10px}a{color:#17313b}.hero-grid{gap:10px}}
</style></head><body><header><div class="eyebrow">PUBLICATION STRATEGY / COT × SOLAR IRRADIANCE</div><h1>把论文送到对的读者面前</h1><p>22本候选期刊的投稿地图：研究定位、专业价值、分区证据与本文差距。先确定科学问题，再决定投稿梯队。</p><nav><a href="#decision">推荐顺序</a><a href="#ranknote">分区怎么读</a><a href="#compare">全刊对照</a><a href="#details">逐刊分析</a><a href="#evidence">同类论文与策略</a></nav><p style="font-size:12px;margin-top:23px">检索日期 2026-10-16 · 国际英文研究论文 · 公开网络证据版 · 离线可读</p></header><main>
<section id="decision" class="intro"><h2>结论：Solar Energy最贴题；一区目标应补上Renewable Energy与IJAEOG</h2><p>前一版推荐偏窄。若以“云光学厚度表示对地表辐照度估计的价值”为主线，Solar Energy依然是自然的首选。若两区域验证充分、贡献解释扎实，可以先考虑Renewable Energy或IJAEOG。IEEE JSTARS和Remote Sensing是主题明确的替代路线。Applied Energy、RSE、TGRS不应仅凭一个提升百分比就判定适合首投。</p><div class="hero-grid"><div class="pick"><small>主题匹配优先</small><strong>Solar Energy</strong>围绕太阳辐射估计、云况依赖和短临应用，叙事改动最小。</div><div class="pick"><small>有条件争取一区</small><strong>Renewable Energy / IJAEOG</strong>分别强化能源应用价值或定量遥感与空间误差解释。</div><div class="pick"><small>遥感专业路线</small><strong>IEEE JSTARS / Remote Sensing</strong>以物理表示、卫星与站点验证、误差传播组织全文。</div></div><p class="muted">以上是基于选题和现有证据的投稿判断，不是编辑部意见或录用概率。“备选”表示策略顺序，不表示容易录用。</p></section>
<section id="ranknote" class="notice"><h2>“几区”必须带口径</h2><p><b>中科院：</b>本报告固定比较公开可辨认的<strong>2025年升级版大类</strong>，并在有证据时备注小类。未登录官方分区数据库，所有CAS值均为公开二手证据，不能当学校考核证明。网页上的“2026新锐分区”没有混入CAS列。</p><p><b>JCR：</b>按学科报告，不只报最优Q。主要采用公开页2026更新的信息；Remote Sensing使用出版社2026年7月统计页。各网站把指标年和发布年混用，未能闭合年份的单独标记“待核”。不以SJR、CiteScore或JCI分区替代JIF quartile。</p><p><b>含金量：</b>是对专业读者、学会背景、学术定位和本文适配性的判断，不是另造一个官方排名。不同学校的分区版本、预警和奖励名单可能不同；本报告没有获得你所在单位的认定目录。部分出版社页面403，相关条目保留官方入口并依靠学会页或已发表论文核验范围，未声称所有官网全文均已读到。</p><p><a href="https://www.fenqubiao.com/" target="_blank">官方分区表入口 ↗</a> · <a href="https://jcr.clarivate.com/" target="_blank">JCR机构查询入口 ↗</a> · <a href="https://mjl.clarivate.com/" target="_blank">Web of Science收录查询入口 ↗</a></p></section>
<section id="compare"><h2>全刊对照 · 按本文投稿适配排序</h2><p class="muted">范围覆盖能源、遥感和应用大气科学的22个主要候选，不声称穷尽全球所有可能接收交叉研究的期刊。中文期刊、纯综述与明显偏题的刊物不混入首投排序。</p><div class="controls"><input id="q" aria-label="搜索期刊" placeholder="搜索刊名、主题、关键差距…"><select id="field" aria-label="领域"><option value="">全部领域</option><option>能源</option><option>遥感</option><option>大气</option></select><select id="tier" aria-label="投稿梯队"><option value="">全部梯队</option><option>重点候选</option><option>条件候选</option><option>高门槛冲刺</option><option>备选</option></select><select id="cas" aria-label="中科院大类"><option value="">全部CAS大类</option><option value="1区">1区</option><option value="2区">2区</option><option value="3区">3区</option><option value="4区">4区</option><option value="待核">待核</option></select><button id="reset">重置</button><button id="print">打印 / 存PDF</button></div><p class="count" id="count"></p><div class="tablewrap"><table><thead><tr><th>期刊 / 领域</th><th>投稿定位</th><th>CAS 2025大类<br>公开二手证据</th><th>JCR学科分区<br>年份见说明</th><th>对本文的判断</th></tr></thead><tbody>ROWS</tbody></table><div id="empty" class="hidden">没有符合条件的期刊，请重置筛选。</div></div></section>
<section id="details"><h2>逐刊判断 · 不只看分区</h2><p class="muted">“特点”来自期刊范围；“含金量、匹配与差距”是针对本论文的分析。每项附来源入口，费用未核实时不填写猜测金额。</p><div class="cards">CARDS</div></section>
<section id="evidence"><h2>用已发表工作检查方向，而不是凭刊名猜</h2><div class="columns"><div class="panel"><h3>同类研究的三个明确参照</h3><ol><li><a href="https://www.sciencedirect.com/science/article/pii/S0038092X17310435">Solar Energy：Short-term solar irradiance forecasting via satellite/model coupling</a><br>卫星与模型耦合的短临辐射预报，说明该刊读者直接关心我们的目标量与问题链。</li><li><a href="https://www.sciencedirect.com/science/article/pii/S0306261923013430">Applied Energy：Global and direct solar irradiance estimation using deep learning and selected spectral satellite images（2023）</a><br>卫星光谱输入与辐射估计确实在范围内，但这也要求我们解释COT表示相对直接光谱学习的新意。</li><li><a href="https://doi.org/10.5194/amt-17-1851-2024">AMT：Effects of clouds and aerosols on downwelling surface solar irradiance nowcasting and short-term forecasting（2024）</a><br>提供云属性—辐射误差的过程叙事参照；同期反演与预报需要分开讨论。</li></ol><p class="muted">这些例子用于确认选题交集，不是逐刊系统综述，也不能用来推断本稿会被接收。</p></div><div class="panel"><h3>本文需要回答的真正问题</h3><p>“云会影响辐射”本身已经不是新知识。更有价值的问题是：<strong>在同样AGRI观测条件下，通过云产品监督学到的COT表示，何时能改善站点GHI估计，又何时不能？</strong></p><p>现有稿件中的湖南验证约3.02%总体改善与22.45%厚云子集改善，是同一版本、特定支持集的结果；不能直接写成所有版本、所有区域或未来预报的收益。容量/隐特征对照尚不能唯一证明显式COT语义是收益来源。</p><p>两区域独立重复、厚云分层的样本与日期数、日期块不确定性、真实云图案例，比再扩大网络更能服务当前论文。新结果若显示条件性收益，就按条件性结论写。</p><p class="muted">本文状态依据已有稿件与会话证据，仅用于选刊定位；本次没有新跑实验、没有读取test载荷。</p></div></div></section>
<section><h2>投稿路径：根据完成后的证据选，不根据愿望选</h2><div class="columns"><div class="panel"><h3>路径A · 保持论文主旨</h3><p><b>Solar Energy → IEEE JSTARS 或 Remote Sensing</b></p><p>以COT辅助同期GHI估计为主，外推应用为次；保留效果不稳定的边界。适合目前最明确的科学方向。转投需根据拒稿原因修改，不能同时一稿多投。</p><h3>路径B · 争取一区</h3><p><b>Renewable Energy 或 IJAEOG → Solar Energy</b></p><p>前提是两区域证据、独立留出评价与机制解释足够扎实。前者突出能源价值，后者突出定量遥感表示与空间误差。不要同时追两个完全不同的故事。</p></div><div class="panel"><h3>投稿前只补真正关键的证据</h3><ol><li>确认两区域各自独立训练与同支持比较；多站不等于跨区域零样本泛化。</li><li>冻结方案后完成预定最终留出评价，区分调参验证与最终成绩；本报告不授权打开test。</li><li>报告总体与厚云结果的样本量、日期数、种子稳定性和不确定性；不能只突出最大百分比。</li><li>将同期GHI估计与0–4小时预报列成两个问题，分别给指标和限定。</li><li>若预报收益小，只有误差分解支持时才归因到外推；不预设解释。</li></ol></div></div></section>
<section class="panel"><h2>为什么没有把这些刊物列为当前首投</h2><p><b>GRL / JGR: Atmospheres / Atmospheric Chemistry and Physics：</b>需要更明显的大气科学新认识或普适机制；现在的表示增益研究不宜只因出现云物理变量就套入。</p><p><b>ISPRS Journal of Photogrammetry and Remote Sensing：</b>高水平方法与遥感创新方向，当前的站点COT增益主线与其竞争门槛不自然匹配。</p><p><b>Renewable and Sustainable Energy Reviews：</b>本文是原创实验稿，当前叙事不应为了刊名改成综述。<b>Solar Energy Materials and Solar Cells：</b>材料和器件主题与本文不符。</p><p><b>Measurement / IEEE Access / Energy Reports：</b>理论上存在交叉投稿空间，但当前专业读者匹配不优于前列候选；无需为凑清单推荐。<b>Solar Energy Advances / ECM-X：</b>不能因姊妹刊名称或Q1就自动认定与主刊同等收录、声誉和机构认可，需要另行核对。</p><p class="muted">此处是范围筛除，不是期刊质量否定，也不是禁止未来另选。</p></section>
<section class="panel"><h2>证据补充与核验边界</h2><ul><li><a href="https://www.mdpi.com/journal/remotesensing/stats">Remote Sensing官方统计</a>：逐学科JCR分区；不采用其首页最佳Q替代全部类别。</li><li><a href="https://www.iit.comillas.edu/publicacion/info_revista/es/472/Sustainable_Energy_Technologies_and_Assessments">Comillas大学SETA页面</a>：明确区分JCR Q2与SJR Q1。</li><li><a href="https://www.iikx.com/sci/earthscience/10611.html">AMT 2025分区公开汇总</a>：用于补充高校图书馆页年份信息。</li><li><a href="https://www.ablesci.com/journal/index?keywords=weather&amp;order=if_desc">WAF JCR公开检索页</a>：2026更新Q3，与CAS2025证据分别记录。</li><li>未使用论坛自报录用率、审稿周期均值、CAS“估算”值或未来分区预测。没有因未发现预警而声称全部期刊无预警。</li><li>“当前可投”仅指科学主题存在投稿空间。投稿当天仍应从官方入口确认栏目、收录、作者指南和费用；待核字段在取得可靠证据前不作为硬性选刊依据。</li></ul></section>
<footer>研究对象：FY-4B AGRI → COT表示 → 站点GHI估计及短临应用。<br>制作：2026-10-16。HTML不依赖外部脚本、字体或图片；搜索和筛选在本地执行。底层数据：journals.json；可复现生成：build_report.py。学术定位判断与客观分区分列，未修改论文正文。</footer></main><script>
const data=DATA;
function filter(){const q=document.querySelector('#q').value.toLowerCase().trim(),f=document.querySelector('#field').value,t=document.querySelector('#tier').value,c=document.querySelector('#cas').value;let n=0;for(const j of data){const ok=(!q||JSON.stringify(j).toLowerCase().includes(q))&&(!f||j.field===f)&&(!t||j.tier===t)&&(!c||j.cas.startsWith(c));document.getElementById(j.id).classList.toggle('hidden',!ok);document.querySelector(`tr[data-id="${j.id}"]`).classList.toggle('hidden',!ok);if(ok)n++;}document.querySelector('#count').textContent=`显示 ${n} / ${data.length} 本 · 点击刊名查看详细判断`;document.querySelector('#empty').classList.toggle('hidden',n!==0);}
for(const id of ['q','field','tier','cas'])document.getElementById(id).addEventListener('input',filter);
document.querySelector('#reset').onclick=()=>{for(const id of ['q','field','tier','cas'])document.getElementById(id).value='';filter()};document.querySelector('#print').onclick=()=>window.print();filter();
</script></body></html>'''
html=html.replace('ROWS',''.join(rows)).replace('CARDS',''.join(cards)).replace('DATA',json.dumps(D,ensure_ascii=False))
(ROOT/'index.html').write_text(html,encoding='utf-8')
print(f'Built {len(D)} journal profiles: {ROOT / "index.html"}')
