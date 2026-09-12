# Frozen SimVP and COT feature cache

This stage reloads the validation-selected Hunan S and R. It performs full-grid S validation, profiles one A100, verifies 32 forecast sequences, then caches all 27,675 train and 4,258 validation sequences. No test payloads or test metrics. Downstream label join and eight-sequence head overfit are separate prerequisites to A/AC pilot.

Server project: `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907`.
Output: `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/data/frozen_forecast_trainval_20260907_v1`.
Log: `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/forecast_cache.log`.

Use unchanged swc environment and imported model/data code from `/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanSimVPDDP_20260907/scripts`. No package installation. GPU must be allocated by PBS; this script requests one card and profiles inference batches 4/8/16/24/32, choosing measured throughput under 80% allocated memory. More GPUs require measured need and remain capped by the user's eight-card limit.

CPU checks (execute on login, no GPU work):

```bash
cd /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907
python_bin=/public/home/slfu/miniconda3/envs/swc/bin/python
"$python_bin" -m py_compile scripts/cache_frozen_forecast.py
bash -n scripts/run_forecast_cache.pbs
PYTHONPATH=/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanSimVPDDP_20260907/scripts:scripts "$python_bin" scripts/test_forecast_inputs.py
```

After these pass, submit exactly once:

```bash
/opt/gridview/pbs/dispatcher/bin/qsub scripts/run_forecast_cache.pbs
```

Gates: source/normalization/grid SHA, strict weight load, validation RMSE within 0.001 of recorded 0.555505797061281 (BF16 batch-shape tolerance), finite GPU outputs, frozen gradients, serialized readback, valid sequence time axes. Validation saved sufficient statistics allow independent aggregation but are not full-grid prediction archives. No scientific full-grid improvement claim is made from these statistics.

Production feature reads: eight real historical AGRI frames plus deterministic nominal solar geometry. Future AGRI/masks are only used by the separate validation pass. CPP/GHI are never feature inputs. Physical AGRI is obtained by inverting S normalization before applying R normalization. Cache stores float32 physical predicted AGRI, target geometry, nonnegative log1p COT maps/statistics, and frozen R 16D pooled features; AL projection seeds remain 11/22/33, to be applied downstream. Center pixel [8,8] matches both M0 nearest station pixels. Patch bounds are S-config locked and grid hash bound.

On any failed gate, preserve output/log and stop; do not submit downstream training. Rerunning unchanged code/contract may resume existing shards; different code/config requires a separately named output. Check complete.json and all shard hashes before consuming cache. Expected storage roughly 16 GB; label cohort transfer and exact UTC-init mapping are independently audited.
