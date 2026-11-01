from pathlib import Path
import re, shutil

p = Path(__file__).resolve().parent
en = (p/'main.tex').read_text(encoding='utf-8')
zh = (p/'main_zh.md').read_text(encoding='utf-8')
en = re.sub(r'(\\section\{[^}]+\})\n\\FloatBarrier\n', r'\1\n', en)
en = en.replace('Region & Stations & Observation and partition details', 'Region & Stations and coordinates (latitude, longitude) & Observation and partition details')
en = en.replace(r'Hubei & Niushou; Wangzhai; Mizhuang & \pending{coordinates, periods, sampling, counts}', r'Hubei & Niushou (32.0800, 112.0400); Wangzhai (32.0400, 112.1200); Mizhuang (32.1200, 112.2000) & \pending{periods, sampling, counts}')
en = en.replace(r'Hunan & Sili; Zhujia & \pending{coordinates, periods, sampling, counts}', r'Hunan & Sili (25.9479, 112.3479); Zhujia (29.5667, 112.4500) & \pending{periods, sampling, counts}')
en = en.replace('Coordinates, periods, native sampling, and partition counts will be filled from the final data summary.', 'Coordinates are the station configuration values used for registration; periods, native sampling, and partition counts await the final data summary.')
en = en.replace('Table 2: final station coordinates, observation periods, native sampling, and numbers of dates and samples in each partition.', 'Table 2: final observation periods, native sampling, and numbers of dates and samples in each partition.')
zh = zh.replace('| 区域 | 站点 | 观测与划分信息 |', '| 区域 | 站点与坐标（纬度，经度） | 观测与划分信息 |')
zh = zh.replace('| 湖北 | 牛首、王寨、米庄 | 【待填：坐标、时期、采样、数量】 |', '| 湖北 | 牛首（32.0800，112.0400）；王寨（32.0400，112.1200）；米庄（32.1200，112.2000） | 【待填：时期、采样、数量】 |')
zh = zh.replace('| 湖南 | 四里、竺家 | 【待填：坐标、时期、采样、数量】 |', '| 湖南 | 四里（25.9479，112.3479）；竺家（29.5667，112.4500） | 【待填：时期、采样、数量】 |')
note_en = ' The GHI image branch uses 13 AGRI channels and two geometry channels; the daytime indicator in the shared geometry strip is used by retrieval and validity masking, rather than as an additional GHI image channel.'
en = en.replace('Tensor tiles are schematic, not measurements;', 'Tensor tiles are schematic, not measurements;')
marker = r'\label{fig:framework}'
pos=en.index(marker)
start=en.rfind(r'\caption{', 0, pos)
end=en.rfind('}', start, pos)
if note_en.strip() not in en:
    en=en[:end]+note_en+en[end:]
zh=zh.replace('张量图为示意，不是测量；输入定义见表 S2。', '张量图为示意，不是测量；GHI图像分支为13个AGRI通道加2个几何通道，共享几何栏中的白天指示量用于反演与有效性掩膜，不是额外的GHI图像通道；输入定义见表 S2。')
(p/'main.tex').write_text(en,encoding='utf-8')
(p/'main_zh.md').write_text(zh,encoding='utf-8')
fig=p.parent/'figures'/'paper_visual_preview_v3_20261030'/'data_scope'
for ext in ['pdf','png']:
    shutil.copy2(fig/f'figure.{ext}',p/'images'/f'fig01_regions.{ext}')
print('Bilingual final edits and latest serif locator applied.')
