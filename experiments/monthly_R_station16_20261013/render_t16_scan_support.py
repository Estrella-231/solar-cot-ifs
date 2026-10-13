#!/usr/bin/env python3
"""Raw scan-metadata vs observed interval-support diagnosis, selected val keys."""
import csv
import hashlib
import json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'experiments/monthly_R_station16_20261013_v1/results/t16_deterioration_forensics_20261014'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    raw=json.loads((OUT/'scan_raw_FD107.json').read_bytes())
    cached=json.loads((OUT/'scan_metadata_audit.json').read_bytes())
    assert len(raw['records'])==51 and all(r['state']=='READ_TIMING_METADATA' for r in raw['records'])
    key=lambda r:(r['station'],r['target_time_utc'])
    rm={key(r):r for r in raw['records']};cm={key(r):r for r in cached['records']}
    assert set(rm)==set(cm)
    offset_error=max(abs(r['raw_scan_midpoint_offset_s']-cm[k]['cached_scan_midpoint_offset_s']) for k,r in rm.items())
    assert offset_error<.001
    rows=json.loads((OUT/'per_target_diagnostics.json').read_bytes())
    chosen=[next(r for r in rows if r['case']==ci and r['lead_min']==lead) for ci,lead in [(1,60),(3,165)]]
    nextkeys={(r['station'],(datetime.fromisoformat(r['target_time_utc'])+timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S')) for r in chosen}
    labelpath=REPO/'refine-logs/cot_value_first_20261013/tmp/unique_target_labels_train_val.csv'
    next_values={}
    for r in csv.DictReader(labelpath.open(encoding='utf-8-sig',newline='')):
        if r['new_split']=='val' and r['label_status']=='complete_ghi' and key(r) in nextkeys:
            next_values[key(r)]=[float(v) for v in r['raw_ghi_5min_wm2'].split('|')]
    fig,axs=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
    selected=[]
    for ax,r in zip(axs,chosen):
        k=key(r);scan=rm[k];dt=datetime.fromisoformat(r['target_time_utc'])
        nextkey=(r['station'],(dt+timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S'))
        five=r['raw_5min']+next_values[nextkey]
        ax.bar([-12.5,-7.5,-2.5,2.5,7.5,12.5],five,width=4.8,color=['#4777ad']*3+['0.65']*3,
               label='observed mean within each 5-min interval')
        for label,val,col in [('15-min observation',r['ghi'],'black'),('A prediction',r['A'],'#187dad'),('AC prediction',r['AC'],'#d47d12')]:
            ax.hlines(val,-15,0,color=col,lw=2.3,label=f'{label}: {val:.1f}')
        ax.axvline(0,color='0.3',ls='--',label='end of scored 15-min label')
        offset=scan['raw_scan_midpoint_offset_s']/60
        ax.axvline(offset,color='#a52149',lw=2,label=f'actual AGRI row scan: +{offset:.2f} min')
        ax.set_ylim(0,1250);ax.set_xlim(-16,16);ax.set_xticks([-15,-10,-5,0,5,10,15])
        ax.set_xlabel('Minutes relative to nominal target time / label end')
        ax.set_ylabel('GHI (W m$^{-2}$)');ax.legend(fontsize=8,loc='upper right')
        ax.set_title(f"{r['station'].title()} {(dt+timedelta(hours=8)).strftime('%Y-%m-%d %H:%M')} BJT\n"
                     f"R/CPP centre COT: {r['R_center_COT']:.2f}/{r['CPP_center_COT']:.2f}")
        selected.append({'case':r['case'],'station':r['station'],'target_time_utc':r['target_time_utc'],
                         'scan_midpoint_UTC':scan['scan_line_midpoint_UTC'],'observed_5min_intervals':five,
                         'nominal_label_GHI':r['ghi'],'A':r['A'],'AC':r['AC']})
    fig.suptitle('Original HDF scan times: a support mismatch, but not a complete explanation\n'
                 'Left: rapid transition after label end. Right: observed GHI stays high despite AC underestimation.\n'
                 'Selected post-hoc targets within frozen validation cases; no labels moved; later observations are diagnostic only.',fontsize=11)
    names=['03_actual_scan_vs_interval_support.png','03_actual_scan_vs_interval_support.pdf']
    fig.savefig(OUT/names[0],dpi=150)
    fig.savefig(OUT/names[1],metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    receipt={'raw_scan_success_n':51,'cached_offset_max_abs_error_s':offset_error,
             'station_offset_range_s':{s:[min(r['raw_scan_midpoint_offset_s'] for r in raw['records'] if r['station']==s),max(r['raw_scan_midpoint_offset_s'] for r in raw['records'] if r['station']==s)] for s in ('sili','zhujia')},
             'raw_scan_sha256':sha(OUT/'scan_raw_FD107.json'),'cached_scan_sha256':sha(OUT/'scan_metadata_audit.json'),
             'labels_source_sha256':sha(labelpath),'source_sha256':sha(Path(__file__)),
             'output_sha256':{n:sha(OUT/n) for n in names},'selected_posthoc_targets':selected,'test_payload_opened':False}
    (OUT/'scan_support_audit.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({k:v for k,v in receipt.items() if k!='selected_posthoc_targets'},indent=2))


if __name__=='__main__':main()
