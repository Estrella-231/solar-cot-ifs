from pathlib import Path
P=Path(__file__).resolve().parents[2]
e=(P/'main.tex').read_text(encoding='utf-8');z=(P/'main_zh.md').read_text(encoding='utf-8')
e=e.replace('SimVP is the fixed baseline '+r'\cite{simvp,simvpv2}; additional image predictors, if evaluated, use the same downstream comparisons.', 'SimVP is the fixed baseline '+r'\cite{simvp,simvpv2}. The complementary Hunan experiment also uses a diffusion-based image predictor, DyDiff, trained independently on Hunan data. We use the exponential-moving-average weights after 150 epochs and one forecast member generated with 25 sampling steps. Both image predictors use the same downstream comparisons.')
# Assert actual source wording if it differs from the target sentence above.
if 'after 150 epochs' not in e:
    a=e.index('SimVP',e.index(r'\subsection{Short-term forecasting}'))
    b=e.index('Future solar geometry',a)
    old=e[a:b]
    e=e[:a]+r'SimVP serves as the fixed baseline \cite{simvp,simvpv2}. The complementary Hunan experiment also uses a diffusion-based image predictor, DyDiff, trained independently on Hunan data. We use the exponential-moving-average weights after 150 epochs and one forecast member generated with 25 sampling steps. Both image predictors use the same downstream comparisons. '+e[b:]
z=z.replace('SimVP作为固定基线[27,28]；若评价其他底图，则使用相同下游对照。','SimVP作为固定基线[27,28]。湖南补充实验还采用独立在湖南数据上训练的扩散底图模型DyDiff，使用训练150轮后的指数移动平均权重，以25步采样生成一个预报成员。两种底图使用相同下游对照。')
e=e.replace('initialization-specific RMSE differences','random-seed-specific RMSE differences')
z=z.replace('三次初始化RMSE差等权平均','三个随机种子的RMSE差等权平均')
old=r'\pending{Insert observed-derived versus forecast-derived COT trajectory errors and the errors for the four input combinations with irradiance-model parameters held fixed. Quantify the image route, COT route, and interaction using Eq.~\ref{eq:propagation}, and the change in COT advantage using Eq.~\ref{eq:value-loss}.}'
new=r'''A controlled COT replacement indicates that errors in predicted COT still limit the accuracy of the COT-assisted forecast. Keeping the DyDiff AGRI input and the COT-assisted irradiance model fixed, replacing predicted COT at future positions with COT retrieved from observed future imagery decreases RMSE from 122.788 to 108.941 W\,m$^{-2}$; the paired difference is $-13.847$ W\,m$^{-2}$ (95\% interval: $[-17.582,-10.636]$). Conversely, replacing future COT with its last observed value increases RMSE to 129.516 W\,m$^{-2}$, a paired increase of 6.728 W\,m$^{-2}$ ($[3.587,10.035]$). The former is an offline observed-input diagnostic, not an attainable forecast score; together, the contrasts indicate that predicted COT carries useful evolution information while retaining substantial error.

\pending{Complete the aligned COT trajectories and all four input combinations before assigning the image-route, COT-route, and interaction terms in Eq.~\ref{eq:propagation}.}'''
assert old in e;e=e.replace(old,new)
old='【待填：真实图与预测图反演COT的轨迹误差，保持辐照度模型参数不变时四种输入组合的误差；按式(10)量化图像路径、COT路径及交互作用，按式(11)量化COT优势的变化。】'
new='''固定COT替换表明，预测COT的误差仍限制含COT预报的精度。保持DyDiff AGRI输入和含COT辐照度模型不变，将未来位置的预测COT替换为真实未来云图反演的COT，RMSE由122.788降至108.941 W/m²，配对差为−13.847 W/m²，95%区间为[−17.582,−10.636]。反之，以最后观测COT持平替代未来COT，RMSE升至129.516 W/m²，配对增加6.728 W/m²，区间为[3.587,10.035]。前者含未来观测，仅为离线诊断，不是可实现的预报成绩。两组对照说明，预测COT保留了有用的演变信息，同时仍有较大误差。

【待填：完成对齐COT轨迹及四种输入组合后，再按式(10)量化图像路径、COT路径及交互作用。】'''
assert old in z;z=z.replace(old,new)
old=r'\pending{Insert the evidence-backed attribution paragraph: relate the change in COT benefit on common samples to the image/COT replacement contrasts and event errors. Identify image extrapolation as the principal limitation only where the contrasts support that ranking; otherwise describe the measured contribution without assigning an unsupported fraction.}'
new=r'''The input-replacement results distinguish two questions: whether COT helps relative to an imagery baseline, and how accurate the COT-assisted model could be with better COT inputs. On the same Hunan records and with the same three-frame models, observed imagery gives 90.318 W\,m$^{-2}$ without COT and 89.155 W\,m$^{-2}$ with COT, a 1.29\% reduction whose paired interval includes zero ($[-3.345,0.958]$ W\,m$^{-2}$). This point reduction is smaller than the 1.74\% obtained with DyDiff imagery. Hence, these comparisons do not show that extrapolation reduces the relative benefit of COT. Nevertheless, the large improvement after replacing forecast-derived COT with observed-derived COT indicates a sensitivity to errors entering through this input. It does not separate cloud-image prediction error from retrieval behaviour on predicted images or the irradiance model's response to the substituted inputs. The single-image and three-frame percentage gains also involve different samples and representations and cannot be interpreted as an error budget.'''
assert old in e;e=e.replace(old,new)
old='【待填有证据的归因段落：将同一样本上的COT收益变化，与图像／COT替换差异及事件误差联系。对照支持主次排序时，将底图外推识别为主要限制；否则量化已观测贡献，不填写未经证实的占比。】'
new='''输入替换区分了两个问题：COT相对图像基线是否有用，以及含COT模型在更准确COT输入下能达到怎样的精度。相同湖南记录、相同三帧模型下，真实云图输入的无COT和有COT误差为90.318和89.155 W/m²，降低1.29%，其配对区间[−3.345,0.958] W/m²包含零。该点降幅小于DyDiff输入下的1.74%，因此这些比较未表明外推削弱了COT的相对收益。不过，以真实图派生COT替换预测图派生COT后误差明显降低，说明模型对该输入路径中的误差敏感。这一替换没有进一步分离云图预测误差、反演器在预测图上的行为及辐照度模型对替换输入的响应。单帧与三帧的降幅也涉及不同样本和表示，不能作为误差预算。'''
assert old in z;z=z.replace(old,new)
e=e.replace(r'\pending{Summarize the demonstrated influence of image-extrapolation errors on COT and GHI, and distinguish that influence from residual retrieval or mapping errors.}',r'Fixed-model COT replacements indicate both useful predicted evolution and remaining sensitivity to COT input errors; they do not identify a unique physical cause or demonstrate a reduction of relative COT benefit due to image extrapolation.')
z=z.replace('【待填：已证实的底图外推误差对COT和GHI的影响，与剩余反演／映射误差区分。】','固定模型COT替换既显示预测演变信息的作用，也显示对COT输入误差的敏感性；尚不能确定唯一物理原因，也未证明云图外推削弱了COT的相对收益。')
(P/'main.tex').write_text(e,encoding='utf-8');(P/'main_zh.md').write_text(z,encoding='utf-8')
print('Review corrections saved.')
