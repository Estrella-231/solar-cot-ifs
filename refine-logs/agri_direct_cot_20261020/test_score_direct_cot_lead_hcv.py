"""Synthetic regression test for paired H-CV delta and date-block bootstrap."""
import numpy as np
from score_direct_cot_lead_hcv import point_delta, bootstrap

DATA={}
for st in ('sili','zhujia'):
 for fold in range(5):
  keys=[];truth=np.asarray([100.,200.,300.,400.])
  for j,(day,lead) in enumerate((('2024-04-01',15),('2024-04-01',60),('2024-04-03',15),('2024-04-03',60))):
   keys.append({'station':st,'init_time_utc':f'2024-04-01 00:{j:02d}:00','target_time_utc':day+' 12:00:00','lead_minutes':lead})
  for seed in (42,43,44):
   pred={
    'none_lead':truth+1.,'pxc_lead':truth,
    'none_no_lead':truth+1.,'pxc_no_lead':truth+.5,
   }
   for arm,yhat in pred.items(): DATA[(st,fold,arm,seed)]={'keys':keys,'truth':truth,'pred':yhat}

assert abs(point_delta(DATA,'cot_effect_with_lead')+1.0)<1e-9
assert abs(point_delta(DATA,'cot_effect_without_lead')+.5)<1e-9
assert abs(point_delta(DATA,'lead_cot_interaction')+.5)<1e-9
assert abs(point_delta(DATA,'cot_effect_with_lead',True)+1.0)<1e-9
p,lo,hi=bootstrap(DATA,'cot_effect_with_lead',False,nboot=100,seed=1)
assert (abs(p+1)<1e-9 and abs(lo+1)<1e-9 and abs(hi+1)<1e-9),(p,lo,hi)
print('DIRECT_COT_LEAD_HCV_SCORER_TESTS_PASS')
