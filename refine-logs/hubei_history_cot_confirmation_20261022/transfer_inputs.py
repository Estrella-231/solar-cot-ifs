"""Stream compressed new-date crops between SSH hosts without local raw copies."""
import hashlib,json,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
REMOTE='/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_history_cot_confirmation_20261022'
SRC='/tmp/hubei_history_cot_new_period_ready_20261022_attempt2'
def main():
    prefix=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12']
    start=time.monotonic();n=0;h=hashlib.sha256();last=start
    with (HERE/'transfer_source.log').open('wb') as se,(HERE/'transfer_destination.log').open('wb') as de:
        source=subprocess.Popen(prefix+['FD-201',f'set -o pipefail; tar -h -C {SRC} -cf - data aux frame_manifest.csv PREPROCESS_REPORT.json | gzip -1'],stdout=subprocess.PIPE,stderr=se)
        dest=subprocess.Popen(prefix+['zjnu-hpc',f'test ! -e {REMOTE}/stage_attempt2 && mkdir {REMOTE}/stage_attempt2 && tar -xzf - -C {REMOTE}/stage_attempt2'],stdin=subprocess.PIPE,stdout=de,stderr=de)
        try:
            while True:
                b=source.stdout.read(1<<20)
                if not b:break
                dest.stdin.write(b);h.update(b);n+=len(b)
                if time.monotonic()-last>25:
                    print(json.dumps(dict(compressed_MiB=round(n/2**20,1),seconds=round(time.monotonic()-start,1))),flush=True);last=time.monotonic()
            dest.stdin.close();src_exit=source.wait();dst_exit=dest.wait()
        except Exception:
            source.terminate();dest.terminate();raise
    result=dict(state='TRANSFER_COMPLETE' if src_exit==dst_exit==0 else 'FAILED_RETAIN_PARTIAL',
        source_exit=src_exit,destination_exit=dst_exit,compressed_bytes=n,compressed_stream_sha256=h.hexdigest(),
        seconds=time.monotonic()-start,destination=REMOTE+'/stage_attempt2',source=SRC,
        old_test_payloads_read=0,old_buffer_payloads_read=0)
    (HERE/'TRANSFER_REPORT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    assert src_exit==dst_exit==0
if __name__=='__main__':main()
