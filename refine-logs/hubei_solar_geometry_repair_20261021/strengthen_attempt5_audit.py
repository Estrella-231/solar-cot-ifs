from pathlib import Path
p=Path(r'F:\CODE_Classify\辐照度预报\solar-cot-ifs\refine-logs\hubei_solar_geometry_repair_20261021\code_pipeline_attempt5\build_corrected_data.py')
s=p.read_text(encoding='utf-8')
old="final=dict(solar_geometry_source_sha256=sha(Path(__file__).parent/'solar_geometry_noaa.py'),state='CORRECTED_SOLAR_TRAIN_VAL_DATA_READY',S_cache=str(a.cache),"
new="actual_cache_root=str(Path(manifest[0]['data_path']).parent.parent);final=dict(solar_geometry_source_sha256=sha(Path(__file__).parent/'solar_geometry_noaa.py'),state='CORRECTED_SOLAR_TRAIN_VAL_DATA_READY',S_cache=actual_cache_root,"
assert old in s;s=s.replace(old,new,1);p.write_text(s,encoding='utf-8')
p=Path(r'F:\CODE_Classify\辐照度预报\solar-cot-ifs\refine-logs\hubei_solar_geometry_repair_20261021\code_pipeline_attempt5\audit_prepared_attempt5.py')
s=p.read_text(encoding='utf-8')
old="assert sum(r['split'] == 'val' for r in manifest) == 913\n    result ="
new="assert sum(r['split'] == 'val' for r in manifest) == 913\n    # Independently verify every corrected S cache file referenced by the new indexes.\n    for row in manifest:\n        for path_key, sha_key in (('data_path', 'data_sha256'), ('aux_path', 'aux_sha256')):\n            path = Path(row[path_key])\n            assert path.is_file() and sha(path) == row[sha_key]\n    result ="
assert old in s;s=s.replace(old,new,1);p.write_text(s,encoding='utf-8')
