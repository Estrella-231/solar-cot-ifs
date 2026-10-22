"""Small read-only process/package probe used by the transfer continuation."""
import json
from pathlib import Path

root = Path('/tmp/hubei_cot_regional_input_20261023_ready')
state = {}
for name, pid, script in [('extract', 2318421, 'extract_cpp_region.py'),
                           ('finalizer', 2347947, 'finalize_source.py'),
                           ('audit', 2359683, 'run_ready_audit.py')]:
    try:
        state[name+'_live'] = script.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:
        state[name+'_live'] = False
for name in ('REPORT.json', 'INDEPENDENT_STATISTICS.json'):
    p = root/name
    if p.exists():
        try:
            state[name] = json.loads(p.read_text())['state']
        except json.JSONDecodeError:
            state[name] = 'WRITE_IN_PROGRESS'
print(json.dumps(state))
