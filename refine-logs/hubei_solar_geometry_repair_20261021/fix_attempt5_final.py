from pathlib import Path
base=Path(r'F:\CODE_Classify\辐照度预报\solar-cot-ifs\refine-logs\hubei_solar_geometry_repair_20261021\code_pipeline_attempt5')
for p in base.glob('*.py'):
 s=p.read_text(encoding='utf-8');s2=s.replace('PASS_PREPARED_ATTEMPT4','PASS_PREPARED_ATTEMPT5')
 if s2!=s:p.write_text(s2,encoding='utf-8')
