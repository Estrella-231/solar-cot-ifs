"""Bounded causal reader witness using synthetic aux, no observations on disk."""
import tempfile
from pathlib import Path
from datetime import datetime,timedelta,timezone
import numpy as np
from cache_frozen_forecast import ForecastInputs,identity

with tempfile.TemporaryDirectory() as d:
    root=Path(d);start=datetime(2024,4,2,tzinfo=timezone.utc)
    rels=[(start+timedelta(minutes=15*k)).strftime('%Y/%m/%d/%Y%m%d%H%M.npy') for k in range(24)]
    row=dict(data_relpaths='|'.join(rels),BJT_start='2024-04-02 08:00:00')
    assert identity(row)==start+timedelta(minutes=105)
    data=object.__new__(ForecastInputs);data.rows=[row];data.root=root;data.cache=root/'cache'
    data.mean=np.zeros((13,1,1),np.float32);data.std=np.ones((13,1,1),np.float32)
    calls=[]
    def history(rel):
        assert rel in rels[:8], 'future science read'
        calls.append(rel)
        return np.zeros((13,2,2),np.float32),np.ones((13,2,2),bool),np.ones((3,2,2),np.float32)
    data.frame=history
    for rel in rels[8:]:
        p=data.cache/(rel+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
        # Only geometry exists: accidentally accessing future AGRI raises KeyError.
        np.savez(p,g=np.ones((3,2,2),np.float32))
    idx,x,g=data[0]
    assert calls==rels[:8] and x.shape==(8,13,2,2) and g.shape==(24,3,2,2)
print('PASS: history-only AGRI read, all 16 target geometries, canonical init/lead')
