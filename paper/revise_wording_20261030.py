"""Apply the literature-informed wording revision to the existing bilingual draft."""
from pathlib import Path
import hashlib
import json
import re

root = Path(__file__).resolve().parent
archive = root / 'revision_records' / '20261030_wording_review'
archive.mkdir(parents=True, exist_ok=True)
en_path, zh_path = root / 'main.tex', root / 'main_zh.md'
en, zh = en_path.read_text(encoding='utf-8'), zh_path.read_text(encoding='utf-8')
original_en, original_zh = en, zh
for name, text in [('main.tex', en), ('main_zh.md', zh)]:
    dest = archive / ('before_' + name)
    if not dest.exists():
        dest.write_text(text, encoding='utf-8')
changes = []

def change(lang, old, new):
    global en, zh
    text = en if lang == 'en' else zh
    count = text.count(old)
    if count != 1:
        raise ValueError(f'{lang}: expected one match, got {count}: {old[:90]}')
    text = text.replace(old, new)
    if lang == 'en':
        en = text
    else:
        zh = text
    changes.append({'language': lang, 'before': old, 'after': new})

change('en', en.split('\\begin{abstract}\n')[1].split('\n\\end{abstract}')[0], r'''Cloud optical thickness (COT) characterizes the attenuation of solar radiation by clouds. This study investigates whether including retrieved COT improves global horizontal irradiance (GHI) estimation when multispectral satellite imagery is already available. A retrieval model trained using satellite cloud products generates local COT fields, which are combined with AGRI imagery and solar geometry to estimate 15-minute mean GHI. Models with and without COT are compared at three stations in Hubei and two in Hunan, with separate training for each region. We evaluate overall errors and their dependence on cloud optical thickness, and examine cloud-passage and growth--dissipation events. We then apply the irradiance models to cloud images predicted 15--240 minutes ahead. Controlled input substitutions assess how errors in these images affect retrieved COT and GHI forecasts. \pending{Regional RMSEs with and without COT, relative reductions, and the main difference between cloud conditions.} \pending{Forecasting results and the measured contribution of cloud-image prediction errors.} The analysis examines when COT improves irradiance estimates and how cloud-image prediction errors affect this improvement in forecasting.''')
change('zh', zh.split('## 摘要\n\n')[1].split('\n\n## 1 引言')[0], '''云光学厚度（COT）表征云对太阳辐射的衰减。本文研究在已有多光谱卫星图像的基础上，加入反演COT能否进一步改善总水平辐照度（GHI）估计。利用卫星云产品训练的反演模型生成局地COT场，与AGRI云图及太阳几何共同用于估计15分钟平均GHI。在湖北三站和湖南两站比较有无COT的模型，两区域分别训练。评价总体误差及其随云光学厚度的变化，并分析云团经过和生消过程；随后将辐照度模型用于未来15～240分钟的预测云图。通过控制其余条件、替换输入，评价云图预测误差对反演COT和GHI预报的影响。【待填：分区域有无COT的RMSE、相对降幅及主要云况差异。】【待填：预报结果及云图预测误差的影响。】本文分析COT在什么条件下改善辐照度估计，以及云图预测误差如何影响其在预报中的增益。''')

change('en', 'Data-driven irradiance models exploit spectral and spatial patterns, yet a spectral description and an explicit cloud-property representation need not be equally useful to a finite-capacity estimator.', 'Data-driven models learn spectral and spatial features associated with surface irradiance. Providing COT explicitly may help these models use information about cloud attenuation already present in the imagery.')
change('zh', '数据驱动模型可以利用光谱和空间模式，但对于容量有限的估计器，光谱描述与显式云物理表示未必同样易于利用。', '数据驱动模型学习与地表辐照度有关的光谱和空间特征。显式提供COT，可能帮助模型利用图像中已有的云衰减信息。')
change('en', 'These studies motivate examining the incremental value of an explicit optical-thickness field when the original multispectral observations are already retained. Because COT is inferred from those observations, the additional information takes the form of a representation learned with cloud-property supervision rather than an independent measurement.', 'These studies motivate a comparison of irradiance models with and without retrieved COT while retaining the original multispectral observations. Here, COT is derived from those same observations using a model trained against cloud-product references; it incorporates cloud-property supervision without adding an independent observation.')
change('zh', '这些研究促使我们进一步检验：保留原始多光谱观测时，显式光学厚度场还能增加多少价值。由于COT由这些观测反演得到，它增加的是通过云属性监督学习得到的表示，而非独立观测。', '这些研究促使我们在保留原始多光谱观测的条件下，比较加入和不加入反演COT的辐照度模型。本文COT由同源观测经云产品监督训练的模型反演得到，引入的是云属性监督信息，并未增加独立观测。')
change('en', 'An overall score alone cannot show whether an improvement is broadly distributed or concentrated in particular optical-thickness regimes. Matched comparisons across cloud conditions and continuous events are therefore needed alongside the aggregate evaluation.', 'An overall error statistic can obscure differences between cloud conditions. We therefore compare errors within optical-thickness ranges and examine irradiance changes during individual cloud events.')
change('zh', '总体指标无法说明改善是否广泛出现，还是集中于特定厚度条件。因此，需要将配对总体评价与云况分层和连续事件分析结合。', '总体误差指标可能掩盖不同云况之间的差异。因此，本文比较各光学厚度范围内的误差，并分析具体云事件中的辐照度变化。')

change('en', 'The fitted retriever is frozen during GHI training and forecasting.', 'The retrieval-model parameters remain fixed during irradiance-model training and forecasting.')
change('zh', '训练完成的反演器在GHI训练和预报中冻结。', '反演模型参数在辐照度模型训练及预报期间保持不变。')
change('en', 'Local COT augmentation of the irradiance model', 'Irradiance estimation using local COT fields')
change('zh', '辐照度模型的局地COT增强', '利用局地COT场估计辐照度')
change('en', 'COT retains its nine-pixel arrangement and between-sample magnitude; no patch mean or percentile replaces the field. The paired mappings are', 'The model receives all nine COT values in their original spatial arrangement, preserving local variation and differences in magnitude between samples. The two irradiance models are')
change('zh', 'COT保留九像元排列及样本间幅值，不使用全图均值或分位数代替。两组映射为', '模型接收按原空间位置排列的九个COT值，保留局地变化及不同样本之间的数值差异。两种辐照度模型为')
change('en', 'Local features are encoded before a station readout predicts the clear-sky index. The main estimation comparison uses one target-time frame.', 'The encoded local features are used to predict the station clear-sky index. The main comparison estimates 15-minute mean GHI from one image acquired at the end of that interval.')
change('zh', '局地特征编码后，通过站点读出预测晴空指数。主反演比较使用目标时刻单帧；', '编码后的局地特征用于预测站点晴空指数。主比较利用区间末的一幅图像估计该15分钟区间的平均GHI；')
change('en', 'The same pair of frozen irradiance mappings is applied to imagery with and without forecast-derived COT to measure retention of the estimation representation in forecasting. Heads trained specifically on forecast imagery form a separate comparison. For a three-frame head,', 'We apply the two irradiance models trained on observed imagery to forecast imagery, keeping their parameters fixed, to determine whether the benefit of including COT persists in forecasting. Models trained specifically on forecast imagery are evaluated separately. For the three-frame configuration,')
change('zh', '将同一对冻结辐照度映射用于预测云图及其有无COT输入，检验反演表示在预报中的保留程度。专门使用预测云图训练的GHI头单列比较。三帧头的', '将基于观测云图训练的两个辐照度模型用于预测云图，并保持参数不变，以检验加入COT的作用能否延续到预报。专门使用预测云图训练的模型另行评价。三帧配置的')

change('en', 'Paired evaluation of COT value', 'Evaluation of the contribution of COT')
change('zh', 'COT价值的配对评价', 'COT对辐照度估计的贡献评价')
change('en', 'Equal-capacity imagery features and frozen-retriever features provide complementary controls for model capacity and cloud-supervised representation.', 'Additional image features with matched irradiance-model capacity and features extracted from the fixed COT retrieval model are evaluated to distinguish the effects of model capacity and cloud-property supervision.')
change('zh', '容量匹配的图像特征及冻结反演器特征，分别用于检验额外容量与云监督表示的作用。', '采用辐照度模型容量匹配的额外图像特征，以及固定COT反演模型提取的特征，区分模型容量与云属性监督的作用。')
change('en', 'These comparisons use common reference-supported rows and report sample and date counts. The all-weather GHI comparison retains samples without CPP.', 'All models are compared on the same samples with valid CPP references within each bin; the numbers of samples and dates are reported. The overall GHI evaluation also includes samples without CPP observations.')
change('zh', '分层比较采用共同参考有效样本，并报告样本数和日期数；总体GHI比较保留缺CPP的样本。', '各档内所有模型采用同一组CPP参考有效样本，并报告样本数和日期数；总体GHI评价也包含无CPP观测的样本。')
change('en', 'Image-error propagation through COT and GHI', 'Effects of cloud-image prediction errors on COT and GHI')
change('zh', '底图误差经过COT和GHI的传递', '云图预测误差对COT和GHI的影响')
change('en', 'The mechanism analysis fixes the retriever, the COT-augmented GHI head, and the evaluated sample keys.', 'We keep the retrieval model and the irradiance model with COT unchanged and evaluate all input combinations on the same samples.')
change('zh', '固定反演器、含COT的GHI头及样本键。', '保持反演模型与含COT的辐照度模型不变，在相同样本上评价各输入组合。')
change('en', "All replacements use the same station--initialization--target keys, labels, validity support, weights, and the head's fixed preprocessing, including log-COT standardization.", 'All comparisons use the same stations, forecast initialization times, target intervals, observations, valid samples, and sample weights. The preprocessing, including log-COT standardization, is identical for every input combination.')
change('zh', '全部替换使用同一站点—初始化—目标键、标签、有效支持、权重和头部固定预处理，包括对数COT标准化。', '所有比较采用相同站点、起报时刻、目标区间、观测值、有效样本及样本权重。各输入组合采用相同预处理，包括对数COT标准化。')
change('en', 'It uses MSE, not differences between incompatible RMSE cohorts.', 'All terms are evaluated as MSE on the same samples.')
change('zh', '使用MSE，不将不同样本的RMSE差拼接。', '各项均在相同样本上以MSE计算。')
change('en', 'To distinguish a general forecast-error increase from a reduction in the value of COT, the imagery-only head is also evaluated on observed and forecast imagery on those same keys.', 'To determine whether image prediction errors reduce the improvement from COT, we also evaluate the irradiance model without COT on observed and forecast imagery using the same samples.')
change('zh', '为区分一般的预报误差增加与COT价值下降，还在相同键上将无COT头分别用于真实及预测图像，', '为判断云图预测误差是否削弱COT带来的改善，在相同样本上另将无COT辐照度模型分别用于观测及预测图像，')
change('en', 'A positive value indicates that the COT-augmented head loses more performance under extrapolated inputs than the imagery-only head. A negative value indicates a larger COT advantage in forecasting. This comparison establishes the direction of change before attributing an attenuation of benefit.', 'A positive value indicates that replacing observed imagery with forecast imagery reduces the MSE advantage of the model with COT; a negative value indicates that this advantage increases. The sign is evaluated before interpreting any reduction in forecasting benefit.')
change('zh', '正值表示含COT头在预测输入下损失的性能多于无COT头，负值表示预报中的COT优势更大。先确定变化方向，再归因收益是否衰减。', '正值表示用预测云图替代观测云图后，含COT模型的MSE优势减小；负值表示该优势增大。先判断变化方向，再解释预报增益是否减弱。')

change('en', 'Quality of the retrieved COT representation', 'Evaluation of retrieved COT')
change('zh', 'COT表示的反演质量', '反演COT的评价')
change('en', 'The input-quality analysis compares retrieved COT with the CPP reference at the station pixel and local neighbourhood. It establishes the magnitude and optical-thickness dependence of retrieval errors before evaluating their relevance to GHI.', 'We first compare retrieved COT with CPP references at the station pixel and surrounding neighbourhood, and examine how retrieval errors vary with optical thickness.')
change('zh', '将反演COT与CPP参考在站点像元和局地邻域比较，先明确输入误差及其厚度依赖，再评价其对GHI的意义。', '首先在站点像元及周围邻域比较反演COT与CPP参考，并分析反演误差随光学厚度的变化。')
change('en', 'Incremental value for irradiance estimation', 'Effect of COT on irradiance estimation')
change('zh', '对辐照度反演的增量价值', 'COT对辐照度反演的影响')
change('en', 'Representation controls on common samples. Numerical entries will report matched GHI errors and paired differences.', 'Comparison of COT with alternative features using the same evaluation samples. Entries report GHI errors and paired differences.')
change('zh', '共同样本上的表示对照。数值项填写配对GHI误差与差值。', '相同评价样本上COT与其他特征的比较。数值项报告GHI误差及配对差值。')
change('en', 'Effect beyond added head capacity', 'Effect beyond added model capacity')
change('zh', '是否超出增加头部容量的作用', '是否超出增加模型容量的作用')
change('en', 'Incremental value of COT for synchronous GHI estimation.', 'Effect of including COT on GHI estimation from observed AGRI imagery.')
change('zh', '**图4：逐站配对RMSE与分区域误差差值。**', '**图4：加入COT对观测AGRI云图估计GHI的影响：逐站配对RMSE与分区域误差差值。**')
change('en', 'Sample and date counts accompany each stratum; gains are evaluated within common support.', 'Sample and date counts are shown for each bin; all models use the same valid samples within a bin.')
change('zh', '各档标注样本与日期数，在共同支持内比较。', '各档标注样本与日期数，所有模型在各档内使用相同有效样本比较。')
change('en', 'Forecasting gain and propagation of image errors', 'Effect of COT on forecasting and propagation of image prediction errors')
change('zh', '预报增益与底图误差传递', 'COT对预报的影响及云图预测误差传递')
change('en', 'Report the frozen-head and forecast-trained comparisons separately where both are available.', 'Report results separately for irradiance models trained on observed imagery and those trained on forecast imagery where both are evaluated.')
change('zh', '如两者均有结果，冻结头与预报专训头分别报告。', '如两者均有结果，基于观测云图训练与基于预测云图训练的辐照度模型分别报告。')
change('en', 'four fixed-head input-replacement scores', 'errors for the four input combinations with irradiance-model parameters held fixed')
change('zh', '四种固定头输入替换分数', '保持辐照度模型参数不变时四种输入组合的误差')
change('en', 'Lead-dependent GHI gain and fixed-head input-replacement contrasts', 'GHI error by forecast lead time and effects of input substitution')
change('zh', '分时效GHI增益与固定头输入替换对照', '分时效GHI误差与输入替换的影响')

change('en', 'Why an explicit cloud-property representation can help', 'Contribution of COT to irradiance estimation')
change('zh', '显式云物理表示为何可能有用', 'COT在辐照度估计中的作用')
change('en', 'COT provides a physically defined coordinate for cloud attenuation within a learned irradiance mapping. Its contribution is practical if cloud-product supervision makes relevant information easier for a compact GHI model to use. This interpretation does not require COT to be an independent observation or the only useful representation. Capacity and retrieval-feature comparisons clarify how much of the measured benefit is shared with a richer representation of the same imagery.', 'COT describes optical attenuation by clouds and provides a physically interpretable input to the irradiance model. Training the retrieval model against cloud products may help the irradiance model use cloud information that is implicit in the spectral inputs. Comparisons with additional image features and retrieval-model features assess how much of the improvement is associated with increased model capacity or cloud-property supervision.')
change('zh', 'COT在学习得到的辐照度映射中提供了有明确物理定义的云衰减表征。若云产品监督使紧凑GHI模型更容易利用相关信息，其贡献就具有实际价值。这不要求COT是独立观测，也不要求它是唯一有效的表示。容量和反演器隐特征对照说明收益有多少可以由同源图像的更丰富表示共同解释。', 'COT描述云的光学衰减特征，为辐照度模型提供具有明确物理意义的输入。通过云产品训练反演模型，可能帮助辐照度模型利用光谱输入中隐含的云信息。与额外图像特征及反演模型特征的比较，用于评价模型容量增加和云属性监督分别与改善有多大关系。')
change('en', 'The forecasting chain exposes two routes by which errors in extrapolated imagery can affect GHI. One changes the spectral and spatial input to the irradiance model directly. The other changes the COT inferred from that same imagery. Cloud displacement can shift the inferred attenuation away from the station or to the wrong target time. Spatial smoothing can weaken cloud boundaries, and errors in cloud growth or dissipation can suppress or delay changes in retrieved optical thickness. Consequently, an effective COT representation in observed imagery need not retain the same benefit when supplied with extrapolated fields.', 'Errors in predicted cloud images can affect GHI directly through the spectral and spatial inputs and indirectly through the COT retrieved from those images. Errors in cloud position can change the location and timing of attenuation at the station. Spatial smoothing can reduce local COT contrasts, while errors in cloud growth and dissipation can suppress or delay changes in optical thickness. If the COT-related improvement is smaller with forecast imagery, these processes could help explain the reduction.')
change('zh', '底图外推误差可以通过两条路径影响GHI：一条直接改变辐照度模型接收的光谱和空间输入，另一条改变由同一预测图反演出的COT。云团位移可能使衰减信号偏离站点或出现在错误时刻；空间平滑可能削弱云边界；云团增长和消散误差可能压低或延迟光学厚度变化。因此，在真实云图中有效的COT表示，进入预测云场后未必保留相同程度的增益。', '云图预测误差既可以通过光谱和空间输入直接影响GHI，也可以通过这些图像反演出的COT间接影响GHI。云位置误差可能改变云衰减的空间位置和到站时刻；空间平滑可能削弱COT的局地对比；云团增长与消散误差则可能压低或延迟光学厚度变化。若预测云图条件下加入COT的改善幅度较小，这些过程可为该变化提供可能解释。')
change('en', 'This analysis motivates a distinct next step: improving cloud-image forecasts so that they preserve the location, boundaries, and optical evolution relevant to irradiance. An appropriate future test would hold the retriever and irradiance mapping fixed while improving the upstream image prediction, and evaluate whether better cloud evolution translates into larger GHI gains. The present study establishes the role of COT and examines its sensitivity to forecast inputs; the subsequent study can address how to reduce those upstream errors.', 'Improving the prediction of cloud position, boundaries, and optical-thickness evolution is therefore a direction for further research. Holding the retrieval and irradiance models fixed would allow future work to test whether more accurate cloud-image predictions lead to further reductions in GHI forecast error.')
change('zh', '由此引出独立的下一步问题：如何让云图预报更好地保留辐照度所需的云位置、边界和光学演变。后续研究可以固定反演器与辐照度映射，改进上游底图，并检查云演变改善能否转化为更大的GHI收益。本篇明确COT的作用及其对预报输入误差的敏感性，下一篇进一步研究如何降低这些上游误差。', '因此，改进云位置、边界及光学厚度演变的预测，是值得进一步研究的方向。后续可保持反演与辐照度模型不变，检验更准确的云图预测能否进一步降低GHI预报误差。')
change('en', 'Scope of the interpretation', 'Applicability and limitations')
change('zh', '解释范围', '适用范围与局限')
change('en', 'Regional sampling, reference-product uncertainty, and the finite number of independent weather periods delimit the conclusions.', 'The applicability of the results depends on regional sampling, reference-product uncertainty, and the number of independent weather periods evaluated.')
change('zh', '区域采样差异、参考产品不确定性和有限的独立天气过程决定结论的适用范围。', '结果的适用范围取决于区域采样、参考产品不确定性及参与评价的独立天气过程数量。')

# A wording revision must not alter equations, numeric settings, or citations.
for pattern in [r'\\begin\{equation\}.*?\\end\{equation\}', r'\\bibitem.*', r'\\cite\{[^}]+\}']:
    assert re.findall(pattern, en, re.S if 'equation' in pattern else 0) == re.findall(pattern, original_en, re.S if 'equation' in pattern else 0), pattern
assert re.findall(r'\$\$.*?\$\$', zh, re.S) == re.findall(r'\$\$.*?\$\$', original_zh, re.S)
en_path.write_text(en, encoding='utf-8')
zh_path.write_text(zh, encoding='utf-8')
(archive / 'REPLACEMENTS.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding='utf-8')
(archive / 'CONTENT_CHECK.json').write_text(json.dumps({'equations_unchanged':True, 'citations_unchanged':True, 'replacements':len(changes), 'main_tex_sha256':hashlib.sha256(en.encode()).hexdigest(), 'main_zh_sha256':hashlib.sha256(zh.encode()).hexdigest()}, indent=2), encoding='utf-8')
print(f'Applied {len(changes)} bilingual wording replacements; equations and citations unchanged.')
