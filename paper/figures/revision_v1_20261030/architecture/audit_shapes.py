from pathlib import Path
import ast, hashlib, json, torch
from torch import nn
from torch.nn import functional as F
root=Path('solar-cot-ifs').resolve(); out=root/'paper/figures/revision_v1_20261030/architecture'
def classes(path, names):
    text=path.read_text(encoding='utf-8-sig'); tree=ast.parse(text)
    selected=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
    ns={'torch':torch,'nn':nn,'F':F};exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),ns);return ns
paths={
'hunan_r':'experiments/monthly_R_station16_20261013_v1/scripts/train_station16_R.py',
'hunan_head':'experiments/monthly_R_station16_20261013_v1/head_candidate_v0.py',
'hunan_al':'experiments/monthly_R_station16_20261013/head_r_al_candidate.py',
'hubei_r':'refine-logs/hubei_solar_geometry_repair_20261021/code/train_cot_repaired.py',
'hubei_head':'refine-logs/hubei_cloud_probability_20261020/train_hubei_ghi_fourarm_paired.py'}
record={'state':'CODE_ONLY_CPU_SHAPE_AND_PARAMETER_AUDIT','training_started':False,'experiment_payloads_opened':False,'sources':{},'models':{}}
for key,rel in paths.items():
    p=root/rel;record['sources'][key]={'path':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    if key.endswith('_r'):
        model=classes(p,['block','COTUNet'])['COTUNet']().eval(); y=model(torch.zeros(2,16,16,16))
        record['models'][key]={'parameters':sum(p.numel() for p in model.parameters()),'input':[2,16,16,16],'output':list(y.shape),'nonnegative':bool((y>=0).all()),'widths':[model.enc1[0].out_channels,model.enc2[0].out_channels,model.middle[0].out_channels]}
    elif key=='hunan_head':
        cls=classes(p,['LocalGHIHead'])['LocalGHIHead']
        for v in ['A','AC','AP']:
            m=cls(v,2).eval();y=m(torch.zeros(2,13,3,3),torch.zeros(2,2,3,3),torch.zeros(2,1,3,3) if v=='AC' else None)
            record['models']['hunan_'+v]={'parameters':sum(p.numel() for p in m.parameters()),'output':list(y.shape),'first_readout':m.readout[0].in_features}
    elif key=='hunan_al':
        m=classes(p,['LocalGHIHeadAL'])['LocalGHIHeadAL']().eval();y=m(torch.zeros(2,13,3,3),torch.zeros(2,2,3,3),torch.zeros(2,32,3,3))
        record['models']['hunan_AL']={'parameters':sum(p.numel() for p in m.parameters()),'output':list(y.shape),'first_readout':m.readout[0].in_features}
    else:
        m=classes(p,['PairedGHIHead'])['PairedGHIHead']().eval(); y=m(torch.zeros(2,15,3,3),torch.zeros(2,2,3,3))
        record['models']['hubei_fourarm']={'parameters':sum(p.numel() for p in m.parameters()),'output':list(y.shape),'first_readout':m.readout[0].in_features,'all_arms_identical_shape':True}
(out/'CODE_WITNESS.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record['models'],indent=2))
