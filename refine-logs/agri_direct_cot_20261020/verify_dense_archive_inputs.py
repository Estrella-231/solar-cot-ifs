"""Compare extracted observed AGRI with every matching archived validation patch."""
import json
from pathlib import Path
import numpy as np
from audit_cot_event_timing import read_csv, sha, STATIONS

base = Path(__file__).resolve().parent
new = base/'data/dense_observed_v_reference_patches_v1'
old = base/'results/teacher_forecast_error_decomposition_v1_20261021'
archive = base.parents[1]/'experiments/monthly_R_station16_20261013_v1/data/pack_trainval_v1/x_raw.npy'
assert sha(archive) == '06d4606623c1495d63ab015d778b5691d6f0c072e58fad79c6e559c84d450a31'
receipt = json.loads((base/'results/dense_observed_v_r_reference_v1_20261021/complete.json').read_text())
for name in ('x_raw.npy','mask.npy','rows.csv'):
    assert sha(new/name) == receipt['source']['input_arrays_sha256'][name]
x = np.load(new/'x_raw.npy', mmap_mode='r')
z = np.load(archive, mmap_mode='r')
bytime = {r['target_time_utc']:int(r['frame_index']) for r in read_csv(new/'rows.csv')}
mask = np.load(old/'corrected_common_reference_mask.npy')[:,0]
count = n = pattern = 0
maximum = 0.
for r in read_csv(old/'reference_rows.csv'):
    if r['target_time_utc'] not in bytime: continue
    a = x[bytime[r['target_time_utc']], STATIONS.index(r['station']), :13]
    b = z[int(r['array_index']), :13]
    m = np.broadcast_to(mask[int(r['reference_index'])], a.shape)
    assert np.isfinite(a[m]).all() and np.isfinite(b[m]).all()
    difference = np.abs(a[m].astype(float)-b[m].astype(float))
    if len(difference): maximum = max(maximum, float(difference.max()))
    pattern += int(np.count_nonzero(np.isfinite(a) != np.isfinite(b)))
    count += 1; n += int(m.sum())
result = {'state':'PASS' if maximum <= 1e-6 else 'FAIL',
          'matching_station_patches':count, 'common_valid_channel_values':n,
          'max_abs_difference_on_common_valid_support':maximum,
          'finite_pattern_differences_full_patch':pattern,
          'scope':'13 raw AGRI channels on archived corrected common valid support',
          'script_sha256':sha(Path(__file__)), 'archive_x_sha256':sha(archive),
          'new_x_sha256':receipt['source']['input_arrays_sha256']['x_raw.npy']}
out = base/'results/dense_observed_v_r_reference_v1_20261021/full_archive_input_consistency.json'
out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
assert result['state']=='PASS'
