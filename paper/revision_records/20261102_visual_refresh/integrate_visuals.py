from pathlib import Path
import json, re

root=Path(__file__).resolve().parents[2]
enp=root/'main.tex'; zhp=root/'main_zh.md'
en=enp.read_text(encoding='utf-8'); zh=zhp.read_text(encoding='utf-8')
mapping=json.loads((root/'figures/visual_refresh_20261102/FIGURE_MAP.json').read_text(encoding='utf-8'))
for old,new in mapping.items():
    assert old in en and old in zh, old
    en=en.replace(old,new); zh=zh.replace(old,new)
en=en.replace('figures/hubei_sync_corrected_20261102/three_frame_method.pdf','figures/generated_current_20261102/three_frame_estimation.png')
zh=zh.replace('figures/hubei_sync_corrected_20261102/three_frame_method.png','figures/generated_current_20261102/three_frame_estimation.png')
old='The extra neutral slot is inactive in both models.}'
assert old in en
en=en.replace(old,'The extra neutral slot is inactive in both models. AI-generated visual prototype: cloud textures and feature blocks are schematic, rather than measured fields or an exact layer drawing.}',1)
zh=zh.replace('额外中性槽在两组均不活动。**','额外中性槽在两组均不活动。本图为AI生成的视觉效果稿，云纹理与特征块是示意，并非实测云场或逐层精确绘图。**',1)
# Make room for the forecast workflow in the methods, not after the bibliography.
en=en.replace('The primary estimator uses the ordered three-frame sequence', 'Figure~\\ref{fig:architecture} summarizes the estimation model. The primary estimator uses the ordered three-frame sequence',1)
en=en.replace('With initialization $t$ defined by the last historical image,', 'Figure~\\ref{fig:forecast_application} illustrates the forecasting application and frame alignment. With initialization $t$ defined by the last historical image,',1)
f=r'''\begin{figure}[!htbp]\centering
\includegraphics[width=\linewidth,height=0.70\textheight,keepaspectratio]{figures/generated_current_20261102/forecast_application.png}
\caption{Application of the observed-image-trained irradiance model to forecast imagery. (a) Eight historical AGRI frames enter the image predictor; the resulting sixteen future frames provide both the imagery input and COT retrieved with target-time solar geometry. Registered local fields enter the fixed three-frame GHI model. (b) The three input times for the first three leads: +15 min includes two observed frames, +30 min includes one, and all frames are predicted from +45 min onward. COT is derived from the corresponding observed or predicted frame. The initialization is denoted by $t_0$ in this diagram; each input uses the geometry at its own time. No GHI model is re-trained on forecast imagery. AI-generated visual prototype; cloud textures, maps and station icons are illustrative, not observational or forecast results.}\label{fig:forecast_application}
\end{figure}
\FloatBarrier

'''
assert '\\section{Experimental design}' in en
en=en.replace('\\section{Experimental design}',f+'\\section{Experimental design}',1)
# Renumber existing main-text figure references in Chinese before inserting new Fig.3.
zh=re.sub(r'图([3-7])(?!\d)',lambda m:'图'+str(int(m.group(1))+1),zh)
zh=zh.replace('主估计器使用目标前30分钟','图2概括同期估计模型。主估计器使用目标前30分钟',1)
zh=zh.replace('以最后一张历史图像时刻t为初始化','图3展示预报应用与帧对齐。以最后一张历史图像时刻t为初始化',1)
zf='''![图3](figures/generated_current_20261102/forecast_application.png)

**图3：将真实图训练的辐照度模型用于预测图像。(a)八帧历史AGRI输入底图预测器，输出十六帧未来AGRI；预测图像既直接提供光谱输入，也结合目标时刻太阳几何反演COT，配准后的局地场输入固定三帧GHI模型。(b)前三个时效的输入时间：+15分钟含两张观测帧，+30分钟含一张，+45分钟起三帧均为预测图。COT由对应的真实或预测帧反演。图中以t₀表示初始化时刻，各输入帧采用其自身时刻的几何。GHI模型不在预测图像上重训。本图为AI生成的视觉效果稿，云纹理、地图与站点图标均为示意，不是观测或预报结果。**

'''
assert '## 4 实验设计' in zh
zh=zh.replace('## 4 实验设计',zf+'## 4 实验设计',1)
enp.write_text(en,encoding='utf-8'); zhp.write_text(zh,encoding='utf-8')
print('Integrated two generated method illustrations and six data-preserving restyled figures.')
