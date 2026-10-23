"""Review actual decoded CTH and independent scalar reads; no GHI scoring."""
import hashlib
import json
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parent
D = P/'height_extract_attempt1'
manifest = json.loads((D/'COMPLETE.json').read_text())
witness = json.loads((P/'SOURCE_SCALAR_WITNESS.json').read_text())
assert len(manifest['records']) == len(witness['records']) == 8
centers = {'Niushou': (111,124), 'Wangzhai': (112,126), 'Mizhuang': (110,128)}
rows = []; maximum = 0.; comparisons = 0
for rec, independent in zip(manifest['records'], witness['records']):
    assert rec['time_bjt'] == independent['time_bjt'] and rec['source'] == independent['source']
    path = D/(rec['time_bjt']+'.npz')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == rec['output_sha256']
    with np.load(path, allow_pickle=False) as z:
        h, c, t = z['CTH_raw'], z['CLP_raw'], z['COT_raw']
        assert h.shape == c.shape == t.shape == (256,256)
        attributes = rec['metadata']['variable_attributes']['CTH']
        assert attributes['Unit'] == 'km'
        finite = np.isfinite(h)
        assert not np.any(finite & ((h < 0) | (h > 17.5)))
        assert set(np.unique(c[np.isfinite(c)])).issubset({0.,1.,2.})
        assert np.array_equal(np.isfinite(c), finite) and np.array_equal(np.isfinite(t), finite)
        assert not np.any((c == 0) & (h != 0) & finite)
        for st, (y,x) in centers.items():
            for key in ['CTH','CLP','COT']:
                actual = float(z[key+'_raw'][y,x])
                expected = independent['station_pixels'][st]['values'][key]
                if expected is None:
                    assert np.isnan(actual)
                else:
                    error = abs(actual-expected); maximum = max(maximum,error)
                    assert error == 0
                comparisons += 1
        rows.append(dict(time_bjt=rec['time_bjt'], source_UTC=rec['UTC'],
                         finite_fraction=float(finite.mean()), all_fields_missing=bool(not finite.any()),
                         CTH_km_range=[float(h[finite].min()),float(h[finite].max())] if finite.any() else [],
                         QA_present=rec['metadata']['QA_present'],
                         embedded_time_variables=independent['time_named_variables'],
                         clear_zero_height_pixels=int(((c==0)&finite).sum()),
                         cloudy_zero_height_pixels=int(((c>0)&(h==0)).sum())))
result = dict(state='ACTUAL_CTH_READ_AND_SCALAR_MAPPING_PASS_PARTIAL_HISTORY_NO_QA',
              records=rows, scalar_comparisons=comparisons, scalar_max_abs_error=maximum,
              all_eight_history_frames_available=all(r['finite_fraction'] > 0 for r in rows),
              frames_with_finite_CTH=sum(r['finite_fraction']>0 for r in rows),
              CTH_units='km', QA_quality_certified=False, acquisition_time_verified=False,
              release_time_verified=False, operational_height_input_approved=False,
              forecast_COT_or_GHI_gain_evaluated=False, test_payloads_read=0, GHI_labels_read=0,
              extraction_manifest_sha256=hashlib.sha256((D/'COMPLETE.json').read_bytes()).hexdigest())
(P/'ACTUAL_HEIGHT_REVIEW.json').write_text(json.dumps(result,indent=2))
print(result['state'], comparisons, maximum)
