import hashlib,json,tarfile
from pathlib import Path
base=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021')
archive=base/'restoration_requests_attempt2.tar.gz'
with tarfile.open(archive,'r:gz') as tf:
    names=set(tf.getnames())
    candidates=[n for n in names if n.endswith('RESTORATION_REQUEST_WITNESS.json')]
    assert len(candidates)==1,candidates
    witness=json.load(tf.extractfile(candidates[0]))
    assert witness['state']=='ALL_RESTORATION_REQUESTS_HAVE_C13_WITNESS' and witness['frames']==255
    report=json.loads((base/'restored_attempt3'/'RESTORATION_REPORT.json').read_text())
    restored={r['utc']:r for r in report['records']}
    assert len(restored)==255 and len(witness['records'])==255
    mismatches=[]
    for r in witness['records']:
        u=r['utc']; member=f"restore_requests_with_witness/{u}.npz"
        b=tf.extractfile(member).read(); h=hashlib.sha256(b).hexdigest()
        if h!=r['augmented_request_sha256'] or restored[u]['request_sha256']!=h:
            mismatches.append(u)
    assert not mismatches,mismatches[:5]
    prep=base/'preparation_attempt6.tar'
    with tarfile.open(prep,'r:') as pt:
        prep_names=set(pt.getnames())
    result={'state':'PASS_RESTORATION_WITNESS_ARCHIVE_MATCHES_RESTORED_REPORT','witness_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'witness_member':candidates[0],'witness_frames':witness['frames'],'restored_frames':len(restored),'request_hash_mismatches':mismatches,'preparation_has_witness':any(n.endswith('RESTORATION_REQUEST_WITNESS.json') for n in prep_names),'test_payloads_read':0,'buffer_payloads_read':0}
    out=base/'RESTORATION_WITNESS_ARCHIVE_AUDIT_attempt7.json'
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
