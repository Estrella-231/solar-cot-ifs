# A/AC frozen-feature pilot

Server project `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907`, unchanged swc runtime. No upstream training or test. Cache is frozen; pack full acceptance is required. Pack format/audit is described in HEAD_PACK_20260908.md.

Model decisions frozen before pilot: 13→16→32 CNN with two3×3/pad1 convolutions and SiLU/global mean pooling; lead8 and station4 learned embeddings shared in design; 4 COT slots and 1 IFS slot; 3 common target geometry inputs; MLP52→64→64→1 with SiLU/softplus. A's COT slots and both groups' IFS slot are zeros. Same parameter shape, seed42 initialization/shuffle, AdamW/wd1e-4, learning rates3e-4/1e-3, max50epochs/patience8. Independent model reset after profile and each experiment. Station embeddings are an implementation choice unspecified by earlier plan, common to both arms. No clipping kt, no added robust loss, no extra gradient clipping.

Batch unit is a labeled (station,init,target) row. Uniform random rows with fixed Ntrain/(32*n_station_lead) weights yield the exact station/lead macro MSE objective in expectation, including uneven label coverage. Start profile64 and test up to2048; select smallest batch within5% best speed with20% allocated-memory reserve; use same frozen batch across all arms/rates. BF16 forward, FP32 loss, FP64 GHI evaluation. Full trainval input pack is GPU resident (~8GB) if verified to fit. No S/R parameters enter optimizer.

Smoke: seed42 uniformly select8 training sequence ids (before reading their loss), use all their valid labels, both A/AC 800 updates, final weighted kt MSE <=0.5initial. This is an engineering trainability gate, not evidence of fit quality or generalization. Default PBS only runs profile/smoke. Source audit of dawn-label outliers must be reviewed before pilot: max kt≈478; one Sili target appears repeatedly across leads and dominates target energy. Preserve cohort and log issue; no hidden filtering or loss changes.

CPU checks, execute verbatim:

```bash
cd /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907
python_bin=/public/home/slfu/miniconda3/envs/swc/bin/python
"$python_bin" -m py_compile scripts/train_head_pilot.py scripts/test_head_contract.py
bash -n scripts/run_head_pilot.pbs
"$python_bin" scripts/test_head_contract.py
```

After pack audit COMPLETE and CPU tests, submit once and record returned PBS id:

```bash
/opt/gridview/pbs/dispatcher/bin/qsub scripts/run_head_pilot.pbs
```

After smoke success and dawn source audit disposition, the existing authorization permits seed42 validation pilot (two arms × two rates), using a fresh PBS job:

```bash
/opt/gridview/pbs/dispatcher/bin/qsub -v HEAD_STAGE=pilot scripts/run_head_pilot.pbs
```

Output `runs/head_A_AC_seed42_20260908_v1`; log `head_pilot.log`. Resume only unchanged contract and complete epoch state; failed smoke stops automatically. All four candidate checkpoints and predictions retained; select each arm's LR by validation RMSE only, keep negative effects. Saved prediction NumPy recomputation gate before COMPLETE; independent audit and go/no-go follow, no automatic test or formal claims.
