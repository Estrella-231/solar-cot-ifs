from pathlib import Path
import json,re,hashlib
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[2];R=P.parent;D=Path(__file__).resolve().parent
remote=json.loads((D/'REMOTE_1054_SOURCE_20261101.json').read_text(encoding='utf-8-sig'))
old=json.loads((R/'audits/hubei_cpp_time_impact_20261020/evidence/zjnu_snapshot.json').read_text())
ident={}
for k,v in old['receipts'].items():
 if not k.startswith('new_'):continue
 n=v['path'].split('/hubei_cloud_probability_20261020/')[-1]
 if n in remote['files']:ident[k]=remote['files'][n].get('sha256')==v['sha256']
assert all(ident.values())
assert remote['R_checkpoint']['sha256']=='8ad4dbd1328325b778308ca4f8a6398dc476043ad41230ed094b9ae4855877c9'
fp=D/'geometry_1054_trainval.npy'
assert hashlib.sha256(fp.read_bytes()).hexdigest()==remote['ghi_geometry']['sha256']
rows=pd.read_csv(D/'rows_1054_trainval.csv',dtype={'time_bjt':str})
g=np.load(fp);assert len(rows)==len(g)==25899
src=(R/'refine-logs/hubei_solar_geometry_repair_20261021/code/solar_geometry_noaa.py').read_text()
correct={};legacy={};exec(src,correct);exec(src.replace('4.0 * (180.0 / np.pi) *','4.0 *'),legacy)
coords={'Niushou':(32.08,112.04),'Wangzhai':(32.04,112.12),'Mizhuang':(32.12,112.20)}
cache={};a=[];b=[]
for row in rows.itertuples():
 k=(row.station,row.time_bjt)
 if k not in cache:
  t=datetime.strptime(row.time_bjt,'%Y%m%d%H%M')-timedelta(hours=8)
  la,lo=coords[row.station]
  cache[k]=[float(ns['compute_solar_angles'](np.array(la),np.array(lo),t)[0]) for ns in [correct,legacy]]
 x,y=cache[k];a.append(x);b.append(y)
saved=np.rad2deg(np.arccos(np.clip(g[:,0,1,1].astype(float),-1,1)))
a=np.array(a);b=np.array(b)
stats={}
for split in ['train','val']:
 m=rows.split.eq(split).to_numpy()
 stats[split]={'n':int(m.sum()),'abs_SOZ_error_vs_correct_NOAA_deg':{'median':float(np.median(abs(saved[m]-a[m]))),'max':float(np.max(abs(saved[m]-a[m])))},'abs_SOZ_error_vs_missing_factor_formula_deg':{'median':float(np.median(abs(saved[m]-b[m]))),'max':float(np.max(abs(saved[m]-b[m])))}}
audit={'conclusion':'10.54% is reproduced on correctly UTC-matched CPP but original erroneous solar geometry; NOT a corrected-geometry primary result','live_metadata_hash_matches_archived_source':ident,'R_checkpoint_sha256':remote['R_checkpoint']['sha256'],'geometry_array_sha256':remote['ghi_geometry']['sha256'],'station_center_comparisons':stats,'comparison_scope':'All retained GHI train/val rows, centre pixel, registered station coordinates; no test/buffer','formula_difference':'equation of time: 4*Q vs correct 4*(180/pi)*Q minutes','statistics_scope':'Not a re-training or a corrected GHI score; old result and output retained'}
(D/'AUDIT_1054_GEOMETRY.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')

e=(P/'main.tex').read_text(encoding='utf-8');z=(P/'main_zh.md').read_text(encoding='utf-8')
e=e.replace('Original-geometry estimation comparisons and their cloud-condition dependence are reported separately from the verified-geometry forecasting application. ','')
z=z.replace('原几何条件下的同期对照及云况依赖与已核验几何的预报应用分别报告。','')
e=e.replace('all six station--initialization pairs','all six station--random-seed pairs')
z=z.replace('六个站点—初始化配对','六个站点—随机种子配对')
# Remove a duplicated image caused by recovering the original map caption.
z=re.sub(r'(!\[[^\]]*\]\([^\n]*fig01_regions\.png\))\s*\1',r'\1',z)
# Caption map from actual document body, excluding the macro definition.
body=e[e.index(r'\begin{document}'):];section='';n=0;sn=0;fm={}
for m in re.finditer(r'\\section\*\{Supplementary material\}|\\begin\{figure\}.*?\\end\{figure\}',body,re.S):
 s=m.group()
 if s.startswith(r'\section*'):section='S';continue
 if section:sn+=1;num='S'+str(sn)
 else:n+=1;num=str(n)
 lab=re.search(r'\\label\{([^}]+)',s).group(1);img=re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)',s).group(1)
 fm[lab]={'number':num,'image':Path(img).stem+'.png'}
for lab,rec in fm.items():
 name=rec['image'];num=rec['number']
 match=re.search(r'!\[[^\]]*\]\([^\n]*'+re.escape(name)+r'\)',z)
 assert match,(lab,name)
 if 'verified_results_20261101' in match.group():
  ca=z.rfind('**图',0,match.start());cb=z.find('：',ca)
 else:
  ca=z.index('**图',match.end());cb=z.find('：',ca)
 assert ca>=0 and cb>ca
 z=z[:ca]+'**图'+num+z[cb:]
 # Positions may have shifted; find image again.
 match=re.search(r'!\[[^\]]*\]\([^\n]*'+re.escape(name)+r'\)',z)
 z=z[:match.start()]+re.sub(r'!\[[^\]]*\]',f'![图{num}]',match.group())+z[match.end():]
z=z.replace('分时效差异见图7。','分时效差异见图8。')
z=z.replace('其输入流见图3','其输入流见图3')
z=z.replace('**表2：模型结构及训练设置','**表3：模型结构及训练设置')
z=z.replace('最终结构与训练设置见表2。','最终结构与训练设置见表3。')
# References in the main event text/captions remain 7; forecast curve is 8.
z=z.replace('三站轨迹见图 S2','三站轨迹见图 S2')
z=z.replace('图 S2 保留图7事件','图 S2 保留图7事件')
z=z.replace('此开发样本不同于图8验证队列','此开发样本不同于图S5验证队列')
# Supplemental table order: model sizes S1, forecast inputs S2, legacy controls
# S3, four-arm forecast S4, decomposition S5.
idx=z.index('### S4 ')
z=z[:idx]+z[idx:].replace('**表3：','**表S3：').replace('**表5：','**表S4：').replace('**表6：','**表S5：').replace('（表3）','（表S3）')
# Chinese figure references inside the legacy score paragraphs.
idx=z.index('### S4 ');tail=z[idx:]
tail=tail.replace('（图4）','（图S3）').replace('（图5）','（图S4）')
z=z[:idx]+tail
for name,s in [('main.tex',e),('main_zh.md',z)]: (P/name).write_text(s,encoding='utf-8')
(D/'UNIFIED_FIGURE_MAP.json').write_text(json.dumps(fm,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(stats,indent=2));print('Unified captions repaired; actual main figures',n,'supplementary',sn)
