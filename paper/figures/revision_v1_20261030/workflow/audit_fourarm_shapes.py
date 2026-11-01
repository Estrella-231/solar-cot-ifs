from pathlib import Path
import ast,json,hashlib,torch
from torch import nn
root=Path('solar-cot-ifs').resolve(); out=root/'paper/figures/revision_v1_20261030/workflow'
def load(rel,name):
 p=root/rel;t=ast.parse(p.read_text(encoding='utf-8-sig'));node=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name==name)
 env={'torch':torch,'nn':nn};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),env);return env[name],p
base,p1=load('refine-logs/hubei_forecast_cot_increment_20261022/code_attempt1/history_base.py','HistoryCOTGHIHead')
corr,p2=load('refine-logs/hubei_forecast_cot_increment_20261022/code_attempt1/train.py','ConditionalHead')
b=base().eval(); c=corr(308).eval()
with torch.no_grad():
 y=b(torch.zeros(2,15,3,3),torch.zeros(2,8,15,3,3),torch.zeros(2,8,1,3,3))
 z=torch.zeros(2,308);base_y=torch.tensor([0.2,1.]);initial=c(z,base_y,torch.ones(2));c.delta[-1].bias.fill_(0.3);fallback=c(z,base_y,torch.zeros(2))
 assert torch.equal(initial,base_y) and torch.equal(fallback,base_y)
record={'scope':'code-only CPU shape and zero-init/fallback witness; no experiment arrays read','base_parameters':sum(x.numel() for x in b.parameters()),'base_output_shape':list(y.shape),'conditional_feature_dimension':308,'conditional_parameters_each':sum(x.numel() for x in c.parameters()),'only_future_COT_value_slots_differ':18,'zero_init_exact_base':bool(torch.equal(initial,base_y)),'availability_zero_exact_base':bool(torch.equal(fallback,base_y)),'sources':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [p1,p2]}}
(out/'CODE_WITNESS.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record,indent=2))
