"""Wait for the named finalizer; audit its completed package once."""
import subprocess, sys, time
from pathlib import Path

P = Path(__file__).resolve().parent
ready = Path('/tmp/hubei_cot_regional_input_20261023_ready')
while True:
    try:
        live = b'finalize_source.py' in Path('/proc/2347947/cmdline').read_bytes()
    except FileNotFoundError:
        live = False
    if not live:
        break
    time.sleep(30)
assert (ready/'REPORT.json').exists(), 'Finalizer exited without report; inspect original logs'
subprocess.run([sys.executable, str(P/'audit_ready_statistics.py'), '--ready', str(ready),
                '--manifest', str(P/'selected_cpp_agri.csv'),
                '--out', str(ready/'INDEPENDENT_STATISTICS.json')], check=True)
