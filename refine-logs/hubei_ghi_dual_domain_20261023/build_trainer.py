from pathlib import Path
P=Path(__file__).resolve().parent
text=(P.parent/'hubei_cot_calibration_coverage_20261023/train_station.py').read_text(encoding='utf-8')
def change(old,new):
    global text
    assert old in text,old;text=text.replace(old,new)
change('from history_base import HistoryCOTGHIHead,sha256,array_sha256','from history_base import HistoryCOTGHIHead,sha256,array_sha256\nG=P.parent/\'hubei_cot_calibration_coverage_20261023\'')
change("(P/'METADATA_AUDIT.json')","(G/'METADATA_AUDIT.json')")
change("tensors={k:torch.from_numpy(data[k]).cuda() for k in ['future','history','cot','fusion_features']}","""with np.load(P/'prepared_attempt1'/a.station/'actual_training_views.npz') as z:aux={k:z[k] for k in z.files}
assert np.array_equal(aux['prepared_index'],take[large]) and aux['target_time'].tolist()==[rr[i]['target_time_bjt'] for i in large]
assert np.all(role[large]!='C_select') and np.intersect1d(large,select).size==0
actual_future=data['future'].copy();actual_future[large]=aux['future']
actual_ff=ff.copy();actual_ff[large,:135]=aux['future'].reshape(len(large),-1);actual_ff[large,279:306]=aux['cot_features']
actual_available=solar.all(1).astype('float32');actual_available[large]=aux['available'];actual_available=torch.from_numpy(actual_available).cuda()
tensors={k:torch.from_numpy(data[k]).cuda() for k in ['future','history','cot','fusion_features']}
tensors['actual_future']=torch.from_numpy(actual_future).cuda();tensors['actual_features']=torch.from_numpy(actual_ff).cuda()""")
change("arms=['small_no_future_COT','small_future_COT','large_no_future_COT','large_future_COT']","arms=['forecast_no_COT','forecast_future_COT','dual_no_COT','dual_future_COT']")
change('base_np=base.cpu().numpy().astype(\'float64\')*clear;del base_model',"with torch.inference_mode():actual_base=torch.cat([base_model(tensors['actual_future'][j:j+1536],tensors['history'][j:j+1536],tensors['cot'][j:j+1536]) for j in range(0,len(take),1536)])\n    base_np=base.cpu().numpy().astype('float64')*clear;del base_model")
change("features=torch.cat([tensors['fusion_features'],base[:,None]],1).detach();zero=features.clone();zero[:,279:297]=0", "features=torch.cat([tensors['fusion_features'],base[:,None]],1).detach();zero=features.clone();zero[:,279:297]=0\n    actual_features=torch.cat([tensors['actual_features'],actual_base[:,None]],1).detach();actual_zero=actual_features.clone();actual_zero[:,279:297]=0")
change("scope=arm.split('_')[0];model=ConditionalHead(308).cuda()","scope='large';model=ConditionalHead(308).cuda()")
change("features_arm=zero if 'no_future' in arm else features", "features_arm=zero if 'no_COT' in arm else features\n        aux_features=(actual_zero if 'no_COT' in arm else actual_features) if arm.startswith('dual') else features_arm\n        aux_base=actual_base if arm.startswith('dual') else base\n        aux_available=actual_available if arm.startswith('dual') else available")
change("model.train();ii=torch.as_tensor(order,device='cuda');q=model(features_arm[ii],base[ii],available[ii]);loss=(((q.double()*tclear[ii]-ttruth[ii])/1000)**2).mean();assert torch.isfinite(loss)", "model.train();ii=torch.as_tensor(order,device='cuda');q=model(features_arm[ii],base[ii],available[ii]);aux_q=model(aux_features[ii],aux_base[ii],aux_available[ii])\n            primary=(((q.double()*tclear[ii]-ttruth[ii])/1000)**2).mean();auxiliary=(((aux_q.double()*tclear[ii]-ttruth[ii])/1000)**2).mean();loss=(primary+auxiliary)/2;assert torch.isfinite(loss)\n            if not arm.startswith('dual'):assert torch.equal(q,aux_q) and float(primary)==float(auxiliary)")
change("fit_objective=float(loss.detach())", "fit_objective=float(loss.detach()),forecast_objective=float(primary.detach()),auxiliary_objective=float(auxiliary.detach())")
change("run.append(dict(arm=arm", "if arm.startswith('forecast'):\n            old_arm='large_no_future_COT' if 'no_COT' in arm else 'large_future_COT'\n            old_logs=json.loads((G/'replay'/a.station/f'seed{seed}_{old_arm}_steps.json').read_text())\n            score_difference=max(abs(x['select_rmse']-y['select_rmse']) for x,y in zip(logs,old_logs));loss_difference=max(abs(x['fit_objective']-y['fit_objective']) for x,y in zip(logs[1:],old_logs[1:]))\n            with np.load(G/'replay'/a.station/f'seed{seed}_Cselect.npz') as old:\n                index=3 if 'no_COT' in arm else 4\n                pred_difference=float(abs(q-old['prediction'][index]).max())\n            assert score_difference<=.001 and loss_difference<=1e-8 and pred_difference<=.001,(score_difference,loss_difference,pred_difference)\n            (a.out/f'seed{seed}_{arm}_CONTROL.json').write_text(json.dumps(dict(score_max=score_difference,loss_max=loss_difference,prediction_max=pred_difference)))\n        run.append(dict(arm=arm")
change("state='CALIBRATION_COVERAGE_FOUR_ARM_COMPLETE'","state='PAIRED_DUAL_DOMAIN_FOUR_ARM_COMPLETE'")
(P/'train_station.py').write_text(text,encoding='utf-8',newline='\n')
print('Trainer generated with forecast-only predict closure')
