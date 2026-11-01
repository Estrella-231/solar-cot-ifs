"""Authoritative BJT keys for legacy Hubei UTC-data/BJT-aux sequence indices."""
from datetime import datetime, timedelta
from pathlib import Path

def canonical_bjt_stamps(row):
    data = row['data_paths'].split('|')
    aux = row['aux_paths'].split('|')
    if len(data) != 24 or len(aux) != 24:
        raise ValueError('Expected eight history plus sixteen future frames')
    utc = [datetime.strptime(Path(p).stem[:12], '%Y%m%d%H%M') for p in data]
    bjt = [datetime.strptime(Path(p).stem[:12], '%Y%m%d%H%M') for p in aux]
    if any(g-u != timedelta(hours=8) for u,g in zip(utc,bjt)):
        raise ValueError('Source UTC + 8h must equal auxiliary BJT for every frame')
    if any(bjt[i+1]-bjt[i] != timedelta(minutes=15) for i in range(23)):
        raise ValueError('Sequence is not consecutive fifteen-minute frames')
    return [t.strftime('%Y%m%d%H%M') for t in bjt]
