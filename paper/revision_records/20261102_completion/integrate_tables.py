from pathlib import Path
import re

root = Path(__file__).resolve().parents[2]
record = Path(__file__).resolve().parent
en = (root / 'main.tex').read_text(encoding='utf-8')
zh = (root / 'main_zh.md').read_text(encoding='utf-8')
es = (record / 'tables_data_training_en.tex').read_text(encoding='utf-8')
zs = (record / 'tables_data_training_zh.md').read_text(encoding='utf-8')

def paragraph_after(text, marker):
    return text.split(marker, 1)[1].strip().split('\n\n', 1)[0]

tables = re.findall(r'\\begin\{table\}.*?\\end\{table\}', es, re.S)
assert len(tables) == 2
table2 = tables[0].replace('Coordinates are latitude', 'Native ground observations are one-minute records in Hubei and five-minute interval means in Hunan. Coordinates are latitude')
table3 = tables[1]
counts = paragraph_after(es, '% Add directly after Table 2 or merge into Temporal sampling.')
split = paragraph_after(es, '% Replace the existing Temporal sampling paragraph; retain the existing section heading.')
cpp = paragraph_after(es, '% Replace/extend the CPP description in Study area and data; producer-version details belong in the supplement.')
training = paragraph_after(es, '% Training details to place immediately after Table 3.')
dy = es.split('The regional DyDiff predictors', 1)[1].strip()
dy = 'The regional DyDiff predictors' + dy

start = en.index('\\pending{Table 2:')
end = en.index('\\end{table}', start) + len('\\end{table}')
en = en[:start] + table2 + '\n\n' + counts + '\n' + en[end:]
start = en.rfind('\\begin{table}', 0, en.index('\\label{tab:training}'))
end = en.index('\\end{table}', start) + len('\\end{table}')
en = en[:start] + table3 + '\n\n' + training + en[end:]
start = en.index('Training, validation, and final evaluation dates are separated', en.index('\\subsection{Temporal sampling}'))
end = en.index('\n\n', start)
en = en[:start] + split + en[end:]
oldcpp = 'CPP provides COT and cloud-phase information for retrieval supervision and cloud-condition analysis. It is a satellite retrieval reference, with uncertainty and incomplete temporal coverage. Its availability does not determine inclusion in the main GHI evaluation.'
assert oldcpp in en
en = en.replace(oldcpp, '\n\n' + cpp, 1)
en = en.replace('examine continuous cloud-passage, growth, and dissipation events', 'examine continuous cloud-field and irradiance evolution', 1)
anchor = 'Future observed imagery and CPP are excluded from forecasting inputs.'
assert anchor in en
en = en.replace(anchor, anchor + '\n\n' + dy, 1)

# Keep provenance limitations in the supplement, not in the result narrative.
s1 = re.search(r'\\subsection\*\{S1\.[^\n]+\}\n', en)
assert s1
prov = ('The archived CPP files provide COT, cloud phase, and geolocation, but the recovered processing records do not uniquely identify a public operational product version or a uniform producer quality flag. We therefore describe them as archived FY-4B/AGRI retrieval references, rather than assigning an unverified NSMC product identity. The numerical and input-validity conditions in Section~2.1 are the actual filters used here. Regional reference availability and validity filters differ; neither COT-reference errors nor reference-covered subsets constitute independent ground-truth validation.\n\n')
en = en[:s1.end()] + prov + en[s1.end():]

# Chinese companion follows the same sections, sample definitions, and settings.
zt2 = zs.split('# 表2：真实云图GHI估计的有效数据\n\n', 1)[1].split('## 时间采样', 1)[0].strip()
zt2 = zt2.replace('坐标为纬度、经度', '地面观测原生采样为湖北1分钟记录、湖南5分钟区间平均。坐标为纬度、经度', 1)
zt2 = zt2.replace('表2　', '**表2：', 1).replace('验证数量沿用选模时采用的观测质量剔除规则。', '验证数量沿用选模时采用的观测质量剔除规则。**', 1)
zsplit = zs.split('## 时间采样（替换现有对应段落）\n\n', 1)[1].split('\n\n## CPP', 1)[0].strip()
zsplit = zsplit.replace('以降低拟合与评价样本的依赖性。', '以降低拟合与评价样本的依赖性[24]。', 1)
zcpp = zs.split('## CPP参考说明（整合至研究区域与数据）\n\n', 1)[1].split('\n\n# 表3', 1)[0].strip()
zt3all = zs.split('# 表3：主线模型训练设置\n\n', 1)[1].strip()
zt3 = zt3all.split('\n\n两区DyDiff预测器', 1)[0].strip()
zdy = '两区DyDiff预测器' + zt3all.split('\n\n两区DyDiff预测器', 1)[1].strip()
zt3 = zt3.replace('表3　', '**表3：', 1).replace('同一地区有无COT的配对模型使用相同设置。', '同一地区有无COT的配对模型使用相同设置。**', 1)
start = zh.index('【表2待填：')
end = zh.index('![图1]', start)
zh = zh[:start] + zt2 + '\n\n' + zh[end:]
start = zh.index('**表3：')
end = zh.index('### 3.3', start)
zh = zh[:start] + zt3 + '\n\n' + zh[end:]
start = zh.index('训练、验证与最终评价按日期块分离', zh.index('### 2.3'))
end = zh.index('\n\n', start)
zh = zh[:start] + zsplit + zh[end:]
old = 'CPP提供COT和云相参考，用于反演监督及云况分析；它本身是具有不确定性和非连续覆盖的卫星检索产品。主要GHI评价不以CPP是否存在作为样本筛选条件。'
assert old in zh
zh = zh.replace(old, '\n\n' + zcpp, 1)
zh = zh.replace('并分析连续的云团经过、增长与消散事件', '并分析连续云场与辐照度演变', 1)
anchor = '未来真实图像与CPP不进入预报输入。'
assert anchor in zh
zh = zh.replace(anchor, anchor + '\n\n' + zdy, 1)
old_captions = [line for line in zh.splitlines() if line.startswith('**图7：两种底图下')]
assert len(old_captions) == 1
zh = zh.replace(old_captions[0] + '\n', '', 1)
z_s1 = re.search(r'### S1[^\n]*\n', zh)
if not z_s1:
    z_s1 = re.search(r'## S1[^\n]*\n', zh)
assert z_s1
zprov = ('\n归档CPP文件包含COT、云相和地理定位信息，但现有生产记录未能唯一确认其对应的公开业务产品版本或统一的生产质量标记。因此，本文将其称为归档FY-4B/AGRI检索参考，不赋予未经核实的NSMC产品身份。实际筛选使用第2.1节所述数值和输入有效性条件。两区参考覆盖和有效性筛选不同；COT参考误差及参考覆盖子集均不构成独立地面真值验证。\n')
zh = zh[:z_s1.end()] + zprov + zh[z_s1.end():]
# Repair pre-existing table-1 citation-number drift while synchronizing references.
zh = zh.replace('| Chen等[1] |', '| Chen等[7] |').replace('| Papachristopoulou等[3] |', '| Papachristopoulou等[12] |').replace('| Zhang等[4] |', '| Zhang等[19] |')

(root / 'main.tex').write_text(en, encoding='utf-8')
(root / 'main_zh.md').write_text(zh, encoding='utf-8')
print('Integrated data/training tables and methods; synchronized Chinese and removed stale Fig.7 caption.')
