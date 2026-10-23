"""Create explicit trainer from the exact paired-control implementation."""
from pathlib import Path
P=Path(__file__).resolve().parent
source=(P.parent/'hubei_cot_calibration_coverage_20261023/train_station.py').read_text(encoding='utf-8')
def replace(old,new):
    global source
    assert old in source,old
    source=source.replace(old,new)
replace("from history_base import HistoryCOTGHIHead,sha256,array_sha256", "from history_base import HistoryCOTGHIHead,sha256,array_sha256\nfrom regularization import response_loss,engineering_checks\nG=P.parent/'hubei_cot_calibration_coverage_20261023'")
replace("(P/'METADATA_AUDIT.json')", "(G/'METADATA_AUDIT.json')")
replace("arms=['small_no_future_COT','small_future_COT','large_no_future_COT','large_future_COT']", "arms=['large_no_future_COT','large_future_COT','large_regularized_future_COT']")
replace("orders={'small':stream(small),'large':stream(large)}", "orders={'large':stream(large)}")
replace("predictions=[base_np[select]];run=[]", "predictions=[base_np[select]];run=[]\n    hv=torch.as_tensor(ff[:,270:279],device='cuda');sd=meta['future_cot_norm']['log1p_std']\n    check=engineering_checks(template,features,base,available,hv,sd)\n    (a.out/f'seed{seed}_ENGINEERING.json').write_text(json.dumps(check,indent=2))")
replace("model.train();ii=torch.as_tensor(order,device='cuda');q=model(features_arm[ii],base[ii],available[ii]);loss=(((q.double()*tclear[ii]-ttruth[ii])/1000)**2).mean();assert torch.isfinite(loss)", "model.train();ii=torch.as_tensor(order,device='cuda');omega=torch.zeros((),device='cuda')\n            if 'regularized' in arm:\n                x=features_arm[ii].detach().requires_grad_(True)\n                q,omega,direction=response_loss(model,x,base[ii],available[ii],hv[ii],sd)\n            else:q=model(features_arm[ii],base[ii],available[ii])\n            primary=(((q.double()*tclear[ii]-ttruth[ii])/1000)**2).mean();loss=primary+omega;assert torch.isfinite(loss)")
replace("fit_objective=float(loss.detach())", "fit_objective=float(primary.detach()),response_penalty=float(omega.detach())")
replace("run.append(dict(arm=arm", "if 'regularized' not in arm:\n            old_logs=json.loads((G/'replay'/a.station/f'seed{seed}_{arm}_steps.json').read_text())\n            score_diff=max(abs(x['select_rmse']-y['select_rmse']) for x,y in zip(logs,old_logs))\n            loss_diff=max(abs(x['fit_objective']-y['fit_objective']) for x,y in zip(logs[1:],old_logs[1:]))\n            assert score_diff<=.001 and loss_diff<=1e-8,(score_diff,loss_diff)\n            with np.load(G/'replay'/a.station/f'seed{seed}_Cselect.npz') as old:\n                old_index=3 if 'no_future' in arm else 4\n                prediction_diff=float(abs(q-old['prediction'][old_index]).max());assert prediction_diff<=.001,prediction_diff\n            (a.out/f'seed{seed}_{arm}_CONTROL_REPLAY.json').write_text(json.dumps(dict(score_curve_max=score_diff,loss_curve_max=loss_diff,prediction_max=prediction_diff)))\n        run.append(dict(arm=arm")
replace("state='CALIBRATION_COVERAGE_FOUR_ARM_COMPLETE'", "state='RESPONSE_REGULARIZATION_THREE_ARM_COMPLETE'")
(P/'train_station.py').write_text(source,encoding='utf-8',newline='\n')
print('Explicit paired trainer generated')
