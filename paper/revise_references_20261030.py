"""Integrate verified references into both manuscript languages, preserving results."""
from pathlib import Path
import json
import re
import shutil
import unicodedata

P = Path(__file__).resolve().parent
S = P / 'references/bibliography_expansion_20261030'
D = P / 'revision_records/20261030_bibliography_expansion'
D.mkdir(parents=True, exist_ok=True)
for name in ('main.tex', 'main_zh.md'):
    target = D / ('before_' + name)
    if not target.exists():
        shutil.copy2(P / name, target)
en0 = (D / 'before_main.tex').read_text(encoding='utf-8')
zh0 = (D / 'before_main_zh.md').read_text(encoding='utf-8')
en = en0.split(r'\begin{thebibliography}')[0]
zh = zh0.split('## 参考文献')[0]
oldkeys = re.findall(r'\\bibitem\{([^}]+)\}', en0)
def oldcite(m):
    return '[@' + ','.join(oldkeys[int(x)-1] for x in m[1].split(',')) + ']'
zh = re.sub(r'\[([1-8](?:,[1-8])*)\]', oldcite, zh)

intro_en = r'''\section{Introduction}
Clouds modulate the solar radiation reaching the surface, and changes in their location and optical thickness can produce substantial variability at a ground station. Geostationary multispectral imagery provides frequent observations of these changes and underpins satellite estimates of surface solar irradiance \cite{huang2019}. Cloud optical thickness (COT) describes extinction through a cloud column. Its retrieval from reflected solar radiation has a physical basis, although the inverse problem can be ambiguous for optically thin water clouds \cite{nakajima1990}. These properties make COT a potentially useful representation for irradiance estimation.

Satellite irradiance methods have long represented cloud attenuation explicitly. Early cloud-index methods relate observed reflectance to clear-surface and cloudy references \cite{cano1986}; Heliosat-2 combines calibrated satellite observations with atmospheric-optics modelling \cite{rigollier2004}. These indices are distinct from COT. Physical approaches instead use retrieved cloud properties in radiative-transfer calculations or their approximations. Takenaka et al. developed a neural-network approximation of radiative transfer for surface radiation estimation \cite{takenaka2011}, while Heliosat-4 separates clear-sky irradiance and cloud extinction using lookup tables and satellite cloud properties \cite{qu2017}. These studies establish the role of cloud optical properties in radiation estimation; they do not determine how much an explicit COT input adds to a model that already receives multispectral imagery.

Data-driven retrievals learn spectral, spatial, and temporal relationships with surface irradiance. Chen et al. examined spectral selection for global and direct irradiance estimation \cite{chen2023}. STARNet incorporates neighbouring spatial and temporal information from FY-4A \cite{song2025}, and Schuurman and Meyer combined satellite-product emulation with ground-observation fine-tuning for Meteosat irradiance retrieval \cite{schuurman2025}. Cloud-property information can also be learned separately: Xia et al. retrieved nighttime COT from FY-4B for single-layer non-precipitating clouds \cite{xia2025}. In a hybrid irradiance approach, Liao and Coimbra compared CNNs with and without SCOPE-derived features and extended the approach to short-term forecasting \cite{liao2024}. Together, these approaches motivate testing a cloud-supervised representation alongside the original imagery. In our case, COT is derived from those same observations; it introduces cloud-property supervision without adding an independent measurement.

The contribution of this representation may depend on cloud conditions. Cloud optical properties and aerosols affect radiation errors \cite{pap2024}, while satellite viewing geometry and the displacement between a cloud and its surface shadow complicate comparisons with station measurements \cite{wiltink2025}. Recent FY-4B forecasting work also employs weather-dependent correction of satellite-to-station irradiance differences \cite{shen2026fy4b}. Overall error alone therefore gives an incomplete description of performance. We compare the effect of COT within fixed optical-thickness ranges and examine continuous cloud-passage, growth, and dissipation events. These comparisons link an aggregate gain to the cloud conditions and irradiance variations in which it occurs.

Forecasting introduces uncertainty in the future cloud field. Sirch et al. extrapolated cloud optical thickness for direct-normal irradiance nowcasting \cite{sirch2017}; Miller et al. combined cloud advection, cloud properties, and shadow geometry in solar irradiance forecasts \cite{miller2018}. Learned approaches predict different intermediate quantities: IrradianceNet forecasts satellite-derived irradiance fields \cite{nielsen2021}, Chen et al. use a chain of deep-learning models for satellite-based intraday forecasting \cite{chen2024}, and Cloud-FNO predicts cloud physical properties \cite{zhang2026}. These approaches demonstrate several ways to connect cloud evolution and radiation, with different forecast targets and reference data.

In our image-extrapolation chain, the retrieval and irradiance models receive predicted imagery. Errors in cloud position, boundaries, or growth and decay can reach GHI both through the image input and through retrieved COT. Comparisons of satellite nowcasting methods show that numerical accuracy, spatial appearance, and performance under different sky conditions need not favour the same model \cite{cui2024}. We therefore assess the effect of image prediction through paired input substitutions and aligned cloud and irradiance trajectories, alongside overall error statistics, to examine complementary aspects of forecast performance \cite{yang2020}.

We address three questions: (1) how much does local COT improve GHI estimation beyond AGRI imagery and solar geometry; (2) under which optical-thickness conditions and cloud events does the improvement occur; and (3) how much of that benefit remains in 15--240 min forecasts, and how is it constrained by cloud-image extrapolation errors? Hubei provides the primary station comparisons, while independently trained Hunan models provide complementary regional evidence. The work focuses on the contribution of cloud-property representation and its error propagation; developing an improved image-extrapolation model is a subsequent research question.

'''
intro_zh = '''## 1 引言

云调制到达地面的太阳辐射，其位置和光学厚度变化会引起站点辐照度波动。静止卫星多光谱图像提供对这些变化的高频观测，是卫星估计地表太阳辐照度的重要基础[@huang2019]。云光学厚度（COT）描述穿过云柱的消光，其基于太阳反射辐射的反演具有物理基础，但薄水云条件下的反问题可能存在多解性[@nakajima1990]。这些性质使COT成为潜在有用的辐照度表示。

卫星辐照度方法长期以来就显式描述云衰减。早期云指数方法将观测反射率与晴空地表及有云参考联系起来[@cano1986]，Heliosat-2进一步将定标卫星观测与大气光学模型结合[@rigollier2004]；这些指数与COT不同。物理方法则将反演云属性用于辐射传输计算或其近似。Takenaka等构建了辐射传输的神经网络近似，用于估计地表辐射[@takenaka2011]；Heliosat-4通过查找表和卫星云属性分别描述晴空辐照度与云消光[@qu2017]。这些研究确立了云光学属性在辐射估计中的作用，但并未确定对已经接收多光谱图像的模型，显式COT还能带来多大增量。

数据驱动反演学习光谱、空间和时间信息与地表辐照度的关系。Chen等研究了总辐照度与直接辐照度估计中的光谱选择[@chen2023]；STARNet引入FY-4A的邻近时空信息[@song2025]；Schuurman和Meyer则将卫星产品模拟与地面观测微调结合，用于Meteosat辐照度反演[@schuurman2025]。云属性也可单独学习：Xia等针对单层非降水云，利用FY-4B反演夜间COT[@xia2025]。在混合辐照度方法中，Liao和Coimbra比较有无SCOPE派生特征的CNN，并将其扩展到短时预报[@liao2024]。这些方法促使我们检验在原始图像之外加入云监督表示的作用。本文COT由同源观测得到，引入的是云属性监督信息，并未增加独立测量。

这种表示的作用可能随云况变化。云光学属性与气溶胶影响辐射误差[@pap2024]，卫星观测几何以及云与地面云影之间的位移，则使卫星和站点观测的比较更加复杂[@wiltink2025]。近期FY-4B预报研究也采用与天气条件相关的订正，处理卫星到站点的辐照度差异[@shen2026fy4b]。因此，仅用总体误差不能完整描述模型表现。本文在固定光学厚度范围内比较COT的作用，并分析连续的云团经过、增长与消散事件，将总体改善与具体云况和辐照度变化联系起来。

预报进一步引入了未来云场的不确定性。Sirch等外推云光学厚度，用于直接法向辐照度短临预报[@sirch2017]；Miller等在太阳辐照度预报中结合云平流、云属性和云影几何[@miller2018]。学习型方法预测不同的中间变量：IrradianceNet预测卫星反演辐照度场[@nielsen2021]，Chen等采用深度学习模型链开展卫星日内预报[@chen2024]，Cloud-FNO则预测云物理属性[@zhang2026]。这些方法展示了连接云演变与辐射的不同路径，其预报目标和参考数据各不相同。

在本文云图外推链中，反演器和辐照度模型接收预测图像。云位置、边界或生消误差，既可以通过图像输入直接影响GHI，也可以经过反演COT间接传递。卫星短临方法的比较表明，数值精度、空间形态及不同天空条件下的表现，未必由同一模型占优[@cui2024]。因此，本文通过配对输入替换及对齐的云场和辐照度轨迹分析云图预测误差的影响，并结合总体误差统计，从互补角度评价预报表现[@yang2020]。

本文回答三个问题：（1）在AGRI与太阳几何之外，局地COT能改善多少GHI估计；（2）改善发生在哪些光学厚度条件和云事件中；（3）这种作用在15～240分钟预报中能保留多少，又如何受到云图外推误差限制。湖北开展主要站点比较，湖南独立训练模型提供补充区域证据。本篇重点研究云物理表示的价值及误差传递，更优云图外推模型的开发作为后续研究问题。

'''
en = en[:en.index(r'\section{Introduction}')] + intro_en + en[en.index(r'\section{Study area and data}'):]
zh = zh[:zh.index('## 1 引言')] + intro_zh + zh[zh.index('## 2 研究区域与数据'):]

replacements = []
def replace(lang, before, after):
    global en, zh
    s = en if lang == 'en' else zh
    assert s.count(before) == 1, (lang, before, s.count(before))
    s = s.replace(before, after)
    if lang == 'en': en = s
    else: zh = s
    replacements.append({'language':lang, 'before':before, 'after':after})

replace('en', 'Training, validation, and final evaluation dates are separated at the date-block level.', r'Training, validation, and final evaluation dates are separated at the date-block level to reduce dependence between fitting and evaluation samples \cite{roberts2017}.')
replace('zh', '训练、验证与最终评价按日期块分离。', '训练、验证与最终评价按日期块分离，以降低拟合与评价样本之间的依赖[@roberts2017]。')
replace('en', 'The retriever is a U-Net with two downsampling stages,', r'The retriever follows the U-Net encoder--decoder architecture \cite{unet2015}, with two downsampling stages,')
replace('zh', '采用两级下采样U-Net，特征宽度为32、64、128，', '采用U-Net编码器—解码器结构[@unet2015]，设置两级下采样，特征宽度为32、64、128，')
replace('en', 'with a masked Huber objective,', r'with a masked Huber objective \cite{huber1964},')
replace('zh', '使用CPP的log(1+COT)监督，损失为', '使用CPP的log(1+COT)监督，采用带有效像元掩膜的Huber损失[@huber1964]：')
replace('en', 'Uncertainty is evaluated by paired resampling of date blocks, retaining overlapping forecast records within the same resampled block.', r'Uncertainty is evaluated by paired resampling of date blocks, retaining overlapping forecast records within the same resampled block. This follows the principle of resampling groups rather than treating correlated records as independent observations \cite{field2007}.')
replace('zh', '区间由日期块配对重采样估计，重叠预报记录保留在相应日期块内。', '区间由日期块配对重采样估计，重叠预报记录保留在相应日期块内。这遵循对整组数据重采样、避免将相关记录视为独立观测的原则[@field2007]。')
replace('en', 'Thickness-dependent gains should be interpreted through both attenuation and irradiance variability. A local COT error can affect the inferred radiation level, while cloud edges, diffuse radiation, and changes during a 15-minute interval complicate the mapping to station GHI. The continuous cases connect these processes to the conditions in which the added representation succeeds or fails.', r'Thickness-dependent gains should be interpreted through both attenuation and irradiance variability. Radiative-transfer-based estimation links optical-property errors to surface fluxes, while unresolved broken clouds complicate agreement with point observations \cite{takenaka2011}. Cloud edges, diffuse radiation, and changes during a 15-minute interval further complicate the mapping to station GHI. Cloud-shadow displacement and viewing parallax can also change which cloud pixels are relevant to a station \cite{miller2018,wiltink2025}. The continuous cases connect these processes to the conditions in which the added representation succeeds or fails.')
replace('zh', '厚度依赖的收益应同时从辐射衰减和辐照度变化理解。局地COT误差影响辐射量级；云边界、散射辐射及15分钟内的变化使站点GHI映射更复杂。连续事件将这些过程与表示有效或失效的条件联系起来。', '厚度依赖的收益应同时从辐射衰减和辐照度变化理解。基于辐射传输的估计将光学属性误差与地表通量联系起来，而未解析的碎云使其与点观测的一致性更加复杂[@takenaka2011]。云边界、散射辐射及15分钟内的变化也会影响站点GHI映射。云影位移和观测视差还可能改变与站点相关的云像元[@miller2018,wiltink2025]。连续事件将这些过程与表示有效或失效的条件联系起来。')
replace('en', 'Spatial smoothing can reduce local COT contrasts, while errors in cloud growth and dissipation can suppress or delay changes in optical thickness.', r'Spatial smoothing can reduce local COT contrasts, while errors in cloud growth and dissipation can suppress or delay changes in optical thickness. Prior nowcasting studies distinguish spatial appearance and numerical errors, and identify difficulties in representing cloud evolution \cite{chen2024,cui2024}.')
replace('zh', '空间平滑可能削弱COT的局地对比；云团增长与消散误差则可能压低或延迟光学厚度变化。', '空间平滑可能削弱COT的局地对比；云团增长与消散误差则可能压低或延迟光学厚度变化。已有短临研究区分空间形态与数值误差，并指出表示云演变的困难[@chen2024,cui2024]。')
replace('en', 'CPP-based error measures assess agreement with a retrieval reference, whereas GHI performance is assessed against station observations.', r'CPP-based error measures assess agreement with a retrieval reference, whereas GHI performance is assessed against station observations. Ground-reference quality also matters: a recent FY-4B validation study showed that radiometer quality can materially affect the assessment of satellite irradiance products \cite{chen2025fy4b}.')
replace('zh', 'CPP指标衡量与检索参考的一致性，GHI精度以站点观测评价。', 'CPP指标衡量与检索参考的一致性，GHI精度以站点观测评价。地面参考质量同样重要：近期FY-4B验证研究表明，辐射仪质量会实质影响卫星辐照度产品的评价[@chen2025fy4b]。')

# Normalize bibliographic metadata into a single reusable source index.
refs = {}
root = json.loads((S/'ROOT_METADATA.json').read_text(encoding='utf-8'))
for key, m in root.items():
    refs[key] = dict(key=key, title=m['title'], authors=[a.get('given','')+' '+a.get('family','') for a in m['authors']],
                     year=m['issued']['date-parts'][0][0], venue=m['journal'][0], volume=m['volume'], issue=m['issue'],
                     pages=m['pages'] or m['article_number'], doi=m['doi'], url='https://doi.org/'+m['doi'])
for m in json.loads((S/'CLOUD_SOURCES.json').read_text(encoding='utf-8'))['entries']:
    refs[m['id']] = dict(key=m['id'], title=m['title'], authors=m['authors'], year=m['year'], venue=m['venue'],
                         volume=m['volume'], issue=m.get('issue'), pages=m['pages_or_article'], doi=m['doi'], url='https://doi.org/'+m['doi'])
keymap = {'ronneberger2015unet':'unet2015', 'huber1964robust':'huber1964', 'ineichen2002linke':'ineichen',
          'holmgren2018pvlib':'pvlibclear', 'roberts2017cv':'roberts2017', 'field2007bootstrap':'field2007'}
methods = json.loads((S/'METHOD_SOURCES.json').read_text(encoding='utf-8'))['entries']
for m in methods:
    # Names may vary in source notes; DOI provides the stable identity.
    mapping = {'10.1007/978-3-319-24574-4_28':'unet2015','10.1214/aoms/1177703732':'huber1964',
               '10.1016/S0038-092X(02)00045-2':'ineichen','10.21105/joss.00884':'pvlibclear',
               '10.1111/ecog.02881':'roberts2017','10.1111/j.1467-9868.2007.00593.x':'field2007'}
    key = mapping[m['doi']]
    refs[key] = dict(key=key, title=m['title'], authors=m['authors'], year=m['year'], venue=m['venue'],
                     volume=m['volume'], issue=m.get('issue'), pages=m.get('pages') or m.get('article_number'), doi=m['doi'], url='https://doi.org/'+m['doi'])
refs['simvp'] = dict(key='simvp', title='SimVP: Simpler Yet Better Video Prediction', authors=['Zhangyang Gao','Cheng Tan','Lirong Wu','Stan Z. Li'], year=2022, venue='Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition', volume='', issue='', pages='3170–3180', doi='', url='https://openaccess.thecvf.com/content/CVPR2022/html/Gao_SimVP_Simpler_Yet_Better_Video_Prediction_CVPR_2022_paper.html')
refs['simvpv2'] = dict(key='simvpv2', title='SimVPv2: Towards Simple Yet Powerful Spatiotemporal Predictive Learning', authors=['Cheng Tan','Zhangyang Gao','Siyuan Li','Stan Z. Li'], year=2025, venue='IEEE Transactions on Multimedia', volume='27', issue='', pages='5170–5184', doi='10.1109/TMM.2025.3543051', url='https://doi.org/10.1109/TMM.2025.3543051')

order = []
groups = re.findall(r'\\cite\{([^}]+)\}', en)
for group in groups:
    for key in group.split(','):
        if key not in order: order.append(key)
assert set(order) == set(refs), (set(refs)-set(order), set(order)-set(refs))
assert len(order) == 30, len(order)
doi_lower = [m['doi'].lower() for m in refs.values() if m['doi']]
assert len(doi_lower) == len(set(doi_lower))

def tex(s):
    s = str(s or '').replace('–','--').replace('—','---').replace('‐','-').replace('‑','-').replace('&',r'\&')
    # Use TeX accents rather than requiring a different document engine.
    out=[]
    accents={'\u0301':"'",'\u0300':'`','\u0308':'"','\u0302':'^','\u0303':'~','\u0327':'c','\u030c':'v'}
    for c in s:
        if ord(c)<128: out.append(c); continue
        de=unicodedata.normalize('NFD',c)
        if len(de)==2 and de[1] in accents: out.append('\\'+accents[de[1]]+'{'+de[0]+'}')
        else: out.append(c)
    return ''.join(out)

def authors(m):
    a=m['authors']
    if len(a)>4: return ', '.join(a[:3])+', et al.'
    if len(a)==1:return a[0]
    return ', '.join(a[:-1])+' and '+a[-1]

bib=[]; bibzh=[]
for n,k in enumerate(order,1):
    m=refs[k]
    vol=str(m['volume'] or '')
    if m['issue']:vol+='('+str(m['issue'])+')'
    pages=re.sub(r'(?<=\d)-(?=\d)', '–', str(m['pages'] or ''))
    loc=vol+(':'+pages if pages and vol else pages)
    author_text=authors(m).rstrip('.')+'.'
    item=r'\bibitem{'+k+'}\n'+tex(author_text)+'\n'+tex(m['title'])+'.\n'
    item+=r'\textit{'+tex(m['venue'])+'}'+(', '+tex(loc) if loc else '')+', '+str(m['year'])+'.\n'
    item+=r'\url{'+m['url']+'}.\n'
    bib.append(item)
    bibzh.append(f"{n}. {author_text} {m['title']}. *{m['venue']}*"+(f", {loc}" if loc else '')+f", {m['year']}. [DOI/原始出处]({m['url']}).")

nums={k:i+1 for i,k in enumerate(order)}
zhgroups=re.findall(r'\[@([^\]]+)\]',zh)
assert groups == zhgroups, (groups,zhgroups)
zh=re.sub(r'\[@([^\]]+)\]',lambda m:'['+','.join(str(nums[k]) for k in m[1].split(','))+']',zh)
en+=r'\begin{thebibliography}{99}\small'+'\n'+''.join(bib)+r'\end{thebibliography}'+'\n'+r'\end{document}'+'\n'
zh+='## 参考文献\n\n'+'\n'.join(bibzh)+'\n'

eqbefore=re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', en0, re.S)
eqafter=re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', en, re.S)
assert eqbefore==eqafter
assert re.findall(r'\$\$.*?\$\$',zh0,re.S)==re.findall(r'\$\$.*?\$\$',zh,re.S)
assert en.count(r'\pending{')==en0.count(r'\pending{')
assert zh.count('【待填')==zh0.count('【待填')
assert en.count(r'\figslot{')==en0.count(r'\figslot{')
assert en.count(r'\begin{table}')==en0.count(r'\begin{table}')
for path, content in ((P/'main.tex',en),(P/'main_zh.md',zh)):
    path.write_text(content,encoding='utf-8')
(S/'MANUSCRIPT_REFERENCES.json').write_text(json.dumps([dict(number=nums[k],**refs[k]) for k in order],ensure_ascii=False,indent=2),encoding='utf-8')
(D/'REPLACEMENTS.json').write_text(json.dumps(replacements,ensure_ascii=False,indent=2),encoding='utf-8')
audit=dict(before_reference_count=len(oldkeys),after_reference_count=len(order),citation_groups=len(groups),
           en_zh_citation_sequence_identical=True,unique_dois=len(doi_lower),unresolved_citations=[],uncited_references=[],
           equations_unchanged=len(eqafter),figure_slots=en.count(r'\figslot{'),table_count=en.count(r'\begin{table}'),
           result_placeholders_unchanged=True,ordered_keys=order)
(D/'CHECK.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False,indent=2))
