"""Freeze common new-period rows after source-only quality checks, no scores."""
import csv,hashlib,json
from collections import Counter
from datetime import datetime,timedelta
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=HERE/'final_inputs_attempt2';out.mkdir(exist_ok=False)
    original=json.loads((HERE/'input_selection_attempt1/INPUT_SELECTION.json').read_text())
    frames={datetime.fromisoformat(r['time_BJT']) for r in csv.DictReader((HERE/'frame_manifest.csv').open())}
    available=[];needs=set()
    for s in original['initializations_BJT']:
        init=datetime.fromisoformat(s);times={init-timedelta(minutes=k) for k in range(105,-1,-15)}
        if times.issubset(frames):available.append(s);needs.update(times)
    allowed=set(available);labels={(r['station'],r['target_time_BJT']):r for r in csv.DictReader((HERE/'labels_attempt1/labels_with_clear.csv').open())}
    keys=[r for r in csv.DictReader((HERE/'input_selection_attempt1/candidate_keys.csv').open())
          if r['init_time_BJT'] in allowed and labels[(r['station'],r['target_time_BJT'])]['valid_daylight_clear_sky_label']=='1']
    assert len({(r['station'],r['init_time_BJT'],r['target_time_BJT']) for r in keys})==len(keys)
    with (out/'keys.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(keys[0]));w.writeheader();w.writerows(keys)
    rep=dict(state='FROZEN_NEW_PERIOD_INPUT_KEYS_NO_SCORES',initializations_BJT=available,
        history_times_BJT=[t.isoformat(' ') for t in sorted(needs)],initializations=len(available),history_frames=len(needs),
        forecast_rows=len(keys),removed_initializations_for_unreadable_HDF=original['initializations']-len(available),
        counts=dict(Counter(r['station']+'/'+r['analysis'] for r in keys)),
        dates={st:sorted({r['target_time_BJT'][:10] for r in keys if r['station']==st}) for st in ['Niushou','Wangzhai','Mizhuang']},
        keys_sha256=sha(out/'keys.csv'),labels_with_clear_sha256=sha(HERE/'labels_attempt1/labels_with_clear.csv'),
        model_freeze_sha256=sha(HERE/'MODEL_FREEZE.json'),source_manifest_sha256=sha(HERE/'frame_manifest.csv'),
        old_test_payloads_read=0,old_buffer_payloads_read=0,new_period_scores_computed=0)
    (out/'INPUT_SELECTION.json').write_text(json.dumps(rep,indent=2)+'\n')
    print(json.dumps({k:v for k,v in rep.items() if k not in ['initializations_BJT','history_times_BJT','dates']},indent=2))
if __name__=='__main__':main()
