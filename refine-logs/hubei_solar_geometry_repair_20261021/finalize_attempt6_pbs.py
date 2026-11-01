from pathlib import Path
root=Path(r'F:\CODE_Classify\辐照度预报\solar-cot-ifs\refine-logs\hubei_solar_geometry_repair_20261021')
p=root/'prepare_data_attempt5.pbs'
s=p.read_text(encoding='utf-8-sig').replace('attempt5','attempt6').replace('hb_solar_data5','hb_solar_data6')
s=s.replace('--canonical-data-root /home/Data_Pool_3/chenyi/Auxiliary_data/HuBei_AGRI_Preprocessed/data',
            '--canonical-data-root /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021/r_only_data_attempt6')
anchor='set -euo pipefail\n'
insert='''set -euo pipefail
B=/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021
PREP=/tmp/hubei_solar_audit_20261021_attempt6
test "$(sha256sum "$B/preparation_attempt6.tar" | cut -d' ' -f1)" = 7169be51bff5a20d2cfa53543528cc2914ffa6e2d931833c6b64b5c8742d38e4
test "$(sha256sum "$B/r_only_sources_attempt6.tar" | cut -d' ' -f1)" = 379b4ab5b13c8af954d3d7ccf84cd9cb695ef07574d3bcb670edf6c0c46f3ed6
test ! -e "$PREP"
mkdir -p "$PREP"
tar -xf "$B/preparation_attempt6.tar" -C "$PREP"
'''
if anchor not in s: raise RuntimeError('PBS insertion anchor not found')
s=s.replace(anchor,insert,1)
(root/'prepare_data_attempt6.pbs').write_bytes(s.replace('\r\n','\n').replace('\r','\n').encode('utf-8'))
print('updated attempt6 PBS with preparation unpack and source archive hash checks')
