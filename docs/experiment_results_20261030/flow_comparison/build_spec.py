from pathlib import Path
import json,hashlib

out=Path(__file__).resolve().parent
root=out.parents[2]
spec={'title':'湖南COT预报：正负结果处理流程对照','canvas':{'width':1600,'height':1150},'style':{'font_family':'Microsoft YaHei, Noto Sans CJK SC, sans-serif','font_size':21,'bg_color':'#FFFFFF'},'nodes':[],'edges':[],'groups':[],'labels':[]}
def label(text,x,y,size=24,color='#263238',anchor='middle'):
 spec['labels'].append(dict(text=text,x=x,y=y,font_size=size,color=color,anchor=anchor))
def node(i,text,x,y,w=590,h=78,fill='#F6F8FA',stroke='#BAC3CA',size=22):
 spec['nodes'].append(dict(id=i,label=text,x=x,y=y,width=w,height=h,shape='rounded',fill=fill,stroke=stroke,text_color='#263238',font_size=size))
def edge(a,b):spec['edges'].append({'from':a,'to':b,'color':'#5D6972','thickness':2})
label('湖南COT预报：正、负结果的处理流程差异',800,42,32)
label('左：较早负结果｜预报域训练的单帧头',400,101,25,'#A55A21')
label('右：最新正结果｜同期训练的三帧头冻结回放',1200,101,25,'#087F78')
label('① GHI头怎样训练',800,151,23)
node('nt','H内预测AGRI＋预测COT＋太阳几何\n在预测输入上从头训练配对GHI头',400,217,fill='#FFF8F0',stroke='#CF9B6A')
node('pt','真实同期AGRI＋反演COT＋太阳几何\n先训练配对GHI头，再冻结权重',1200,217,fill='#ECF8F5',stroke='#72B6AA')
node('nh','单帧：目标时刻1张AGRI＋1张COT\n逐时效读取站点3×3',400,332,fill='#FFF8F0',stroke='#CF9B6A')
node('ph','三帧：目标前30、15分钟及目标时刻\n每帧读取站点3×3',1200,332,fill='#ECF8F5',stroke='#72B6AA')
edge('nt','nh');edge('pt','ph')
label('② 验证时实际怎样预报',800,426,23)
node('ni','8帧真实历史AGRI → S_U\n产生16帧未来AGRI',400,488)
node('pi','8帧真实历史AGRI → 全train湖南SimVP\n产生16帧未来AGRI',1200,488)
node('nc','预测AGRI → U-only修正R_U → 未来COT\n另有：历史AGRI＋COT → P-XC → 未来COT',400,605,h=92)
node('pc','预测AGRI → 全train修正R → 未来COT\n与同一时序的预测AGRI逐像元对齐',1200,605,h=92)
node('ng','目标时刻预测AGRI＋COT → 单帧GHI头\n不把真实未来AGRI或CPP送入主预测',400,732,fill='#FFF8F0',stroke='#CF9B6A')
node('pg','对齐三帧AGRI＋COT → 冻结三帧GHI头\n+45～240分钟：三帧全部来自预测场',1200,732,fill='#ECF8F5',stroke='#72B6AA')
for a,b in [('ni','nc'),('nc','ng'),('pi','pc'),('pc','pg')]:edge(a,b)
label('③ 各自共同样本中的结果（RMSE，W/m²）',800,822,23)
node('nr','无COT 135.329 → SimVP→R COT 136.663\nP-XC直接预测COT：136.698；均未获总体收益\n66,086条记录/seed；两站×3种子',400,916,h=130,fill='#FFF8F0',stroke='#CF9B6A',size=22)
node('pr','无COT 127.421 → 预测COT 125.340\n降低1.633%；日期块95%区间[−3.714, −0.561]\n63,480条记录；51日；两站×3种子',1200,916,h=130,fill='#ECF8F5',stroke='#72B6AA',size=22)
label('已核实同时变化：上游训练范围与权重、GHI训练域、1帧/3帧、评价样本',800,1044,23)
label('因此：新结果支持该配置的验证增益，不能单独归因于“三帧”或“修正R”。',800,1091,22)
label('两者均为开发验证；不是独立测试。图示不表示不同实验使用同一套S/R/GHI权重。',800,1129,19,'#61717C')
(out/'flow_comparison.figurespec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
sources=['refine-logs/hunan_dydiff_20261028/full_validation_20261029/AUDIT_SIMVP_20261030.md','refine-logs/hunan_dydiff_20261028/full_validation_20261029/HISTORICAL_COMPARISON_20261030.md','refine-logs/agri_direct_cot_20261020/EXPERIMENT_PLAN.md','refine-logs/agri_direct_cot_20261020/results/direct_cot_ghi_validation_score_v1_local4060/direct_cot_ghi_metrics.json']
(out/'SOURCES.json').write_text(json.dumps({p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},ensure_ascii=False,indent=2),encoding='utf-8')
print('FigureSpec and source hashes written')
