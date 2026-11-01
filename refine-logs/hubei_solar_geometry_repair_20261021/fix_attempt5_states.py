from pathlib import Path
base=Path(r'F:\CODE_Classify\辐照度预报\solar-cot-ifs\refine-logs\hubei_solar_geometry_repair_20261021\code_pipeline_attempt5')
p=base/'build_corrected_data.py';s=p.read_text(encoding='utf-8')
old="csvout(out/'indexes'/f'{role}_oof.csv',[dict(split=q['split'],data_paths='|'.join(str(a.cache/'data'/Path(f).name) for f in q['data_paths'].split('|')),aux_paths='|'.join(str(a.cache/'aux'/Path(f).name) for f in q['aux_paths'].split('|'))) for q in oo])"
new="csvout(out/'indexes'/f'{role}_oof.csv',[dict(split=q['split'],data_paths='|'.join(str(dataByUTC[Path(f).stem[:12]]) for f in q['data_paths'].split('|')),aux_paths='|'.join(str(auxByBJT[Path(f).stem[:12]]) for f in q['aux_paths'].split('|'))) for q in oo])"
assert old in s;s=s.replace(old,new,1);p.write_text(s,encoding='utf-8')
for name in ('audit_prepared_attempt5.py','queue_controller.py'):
 p=base/name;s=p.read_text(encoding='utf-8');s2=s.replace('ATTEMPT4','ATTEMPT5');assert s2!=s,name;p.write_text(s2,encoding='utf-8')
