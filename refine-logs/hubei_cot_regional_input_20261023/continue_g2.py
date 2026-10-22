"""Continue verified CPP source to one transfer and node-local CPU pack.

No qsub, GPU allocation, training, threshold selection or automatic retries of
failed transfers. Partial destination directories are preserved for inspection.
"""
import hashlib, json, shlex, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = '/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cot_regional_input_20261023'
SOURCE = '/tmp/hubei_cot_regional_input_20261023_ready'
CODE = '/home/Data_Pool/zjnu/hubei_cot_regional_input_code_20261023'
DEST = BASE+'/source_ready_attempt1'
PACK = '/dev/shm/hubei_regional_r_pack_20261023_attempt1'
PY = '/public/home/slfu/miniconda3/envs/swc/bin/python'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12']


def remote(host, cmd, **kwargs):
    return subprocess.run(SSH+[host, cmd], check=True, **kwargs)


def main():
    receipt = HERE/'G2_CONTINUATION.json'
    assert not receipt.exists(), 'Do not overwrite an earlier continuation'
    while True:
        try:
            s = json.loads(remote('FD-201', f'python3 {CODE}/probe_ready_state.py', capture_output=True, text=True, timeout=35).stdout)
        except (subprocess.SubprocessError, json.JSONDecodeError) as e:
            print('PROBE_RETRY', str(e), flush=True)
            time.sleep(30)
            continue
        (HERE/'G2_WAIT_STATE.json').write_text(json.dumps(s, indent=2))
        if s.get('REPORT.json') == 'PASS_REGIONAL_CPP_SOURCE_AND_REFERENCE' and s.get('INDEPENDENT_STATISTICS.json') == 'PASS_INDEPENDENT_PACKAGE_AND_TRAIN_STATISTICS':
            break
        if not any(s[k] for k in ('extract_live', 'finalizer_live', 'audit_live')):
            raise RuntimeError(f'Source preparation terminal without passing audits: {s}')
        time.sleep(30)
    start = time.monotonic(); size = 0; digest = hashlib.sha256()
    # -h is essential: ready payload links refer to preserved attempt1 files.
    command = f'set -o pipefail; tar -h -C {SOURCE} -cf - payload REPORT.json frames.json INDEPENDENT_STATISTICS.json REFERENCE_COMPARISON.json | gzip -1'
    with (HERE/'g2_source.log').open('wb') as se, (HERE/'g2_destination.log').open('wb') as de:
        src = subprocess.Popen(SSH+['FD-201', command], stdout=subprocess.PIPE, stderr=se)
        dst = subprocess.Popen(SSH+['zjnu-hpc', f'test ! -e {DEST} && mkdir {DEST} && tar -xzf - -C {DEST}'], stdin=subprocess.PIPE, stdout=de, stderr=de)
        try:
            for block in iter(lambda: src.stdout.read(1 << 20), b''):
                dst.stdin.write(block); digest.update(block); size += len(block)
            dst.stdin.close(); a = src.wait(); b = dst.wait()
        except Exception:
            src.terminate(); dst.terminate()
            raise
    transfer = dict(state='TRANSFER_COMPLETE' if a == b == 0 else 'FAILED_PRESERVE_PARTIAL',
                    source_exit=a, destination_exit=b, compressed_bytes=size,
                    compressed_stream_sha256=digest.hexdigest(), seconds=time.monotonic()-start,
                    source=SOURCE, destination=DEST)
    (HERE/'G2_TRANSFER_REPORT.json').write_text(json.dumps(transfer, indent=2))
    assert a == b == 0
    remote('zjnu-hpc', f'{PY} {BASE}/audit_ready_statistics.py --ready {DEST} --manifest {BASE}/selected_cpp_agri.csv --out {DEST}/DESTINATION_AUDIT.json', stdout=(HERE/'g2_audit.log').open('w'), stderr=subprocess.STDOUT)
    # Keep data preparation entirely on node22's existing local AGRI cache.
    command = f"test ! -e {PACK} && test $(df -B1 --output=avail /dev/shm | tail -1) -gt 20000000000 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 {PY} {BASE}/build_regional_pack.py --manifest {BASE}/selected_cpp_agri.csv --cpp-ready {DEST} --out {PACK}"
    with (HERE/'g2_pack.log').open('w') as log:
        remote('zjnu-hpc', 'ssh -o BatchMode=yes -o ConnectTimeout=12 node22 '+shlex.quote(command), stdout=log, stderr=subprocess.STDOUT)
    result = json.loads(remote('zjnu-hpc', 'ssh -o BatchMode=yes node22 '+shlex.quote(f'cat {PACK}/REPORT.json'), capture_output=True, text=True).stdout)
    assert result['state'] == 'REGIONAL_R_PACK_BUILT_NOT_TRAINED'
    receipt.write_text(json.dumps(dict(state='G2_CPU_PACK_COMPLETE', pack=PACK, host='node22', report=result,
                                       auto_training=False, test_payloads_read=0), indent=2))
    print('G2_CPU_PACK_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
