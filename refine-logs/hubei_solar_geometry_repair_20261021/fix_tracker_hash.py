from pathlib import Path
p=Path(r'F:\CODE_Classify\辐照度预报\solar-cot-ifs\refine-logs\hubei_solar_geometry_repair_20261021\EXPERIMENT_TRACKER.md')
s=p.read_text(encoding='utf-8').replace('0db3d74f0e50740b053e80395e70c1669e29062414126c30b976d300e6e030f2','8f815c7d87bcea8bcfe5991f408cf1d71d5168181b20e6bee9ffc1a59be39843')
p.write_text(s,encoding='utf-8')
